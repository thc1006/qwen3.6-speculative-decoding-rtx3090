#!/usr/bin/env python3
"""A stand-in for llama-server, so the runner can be tested end to end.

Every guard in `retest_runner.py` that fires *after* the arm loop - the
completeness validation, the RUN_COMPLETE / RUN_FAILED decision, the per-file
writes - was unreachable without a GPU and a 20 GiB model, which meant a
mutation that deleted the validation call survived the whole suite. This serves
just enough of the llama.cpp HTTP surface to reach those guards in about a
second: `/health`, `/v1/chat/completions` with a `timings` block, and a startup
banner in the format `server_identity()` parses.

It accepts and ignores every llama.cpp flag except `--port`.

Environment knobs, used by the tests to drive failure paths:
  FAKE_EXIT_BEFORE_HEALTH=1  exit immediately, so the arm never becomes healthy
  FAKE_FAIL_ON_TAG=<n>       return HTTP 500 on the n-th completion request
  FAKE_PREDICTED_N=<n>       tokens to report (default 300); 0 drives the
                             "produced no tokens" branch
  FAKE_SHORT_UNLESS_IGNORE_EOS=1
                             stop early unless the request sets ignore_eos, so
                             the hard-cap guard has something to catch
  FAKE_BUILD=<n> FAKE_COMMIT=<sha>
  FAKE_READY_FILE=<path>     write the parent pid there once the orphan
                             watchdog is armed, and only then
  FAKE_RECORD_REQUESTS=<path>
                             append every request body to this file as JSONL,
                             so a test can assert what treatment was sent
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8080
MODEL = ""
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])
    if a in ("-m", "--model") and i + 1 < len(sys.argv):
        MODEL = sys.argv[i + 1]

N_PREDICT = int(os.environ.get("FAKE_PREDICTED_N", "300"))
FAIL_ON = int(os.environ.get("FAKE_FAIL_ON_TAG", "0"))
_seen = {"n": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the log parseable
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            req = {}
        _seen["n"] += 1
        # Record the body so a test can assert on the TREATMENT the runner sent,
        # not merely on what came back. `hardcap` was dropped on the concurrent
        # submission path and every guard downstream still passed, because a
        # freerun request that reaches the cap looks exactly like a capped one.
        rec = os.environ.get("FAKE_RECORD_REQUESTS")
        if rec:
            with open(rec, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"path": self.path, "body": req}) + "\n")
        if FAIL_ON and _seen["n"] == FAIL_ON:
            self._send(500, {"error": "fake failure"})
            return
        n_pred = N_PREDICT
        if os.environ.get("FAKE_SHORT_UNLESS_IGNORE_EOS") == "1" \
                and not req.get("ignore_eos"):
            n_pred = max(1, N_PREDICT // 3)
        ms = 3000.0
        self._send(200, {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "x" * 32, "reasoning_content": ""},
                "logprobs": {"content": [{"token": "x"} for _ in range(n_pred)]},
            }],
            "usage": {"completion_tokens": n_pred},
            "timings": {
                "predicted_n": n_pred, "predicted_ms": ms,
                "predicted_per_second": (n_pred / ms * 1000) if ms else 0,
                "prompt_n": 40, "prompt_ms": 100.0,
                "draft_n": 0, "draft_n_accepted": 0,
            },
        })


if os.environ.get("FAKE_EXIT_BEFORE_HEALTH") == "1":
    print("fake llama-server: exiting before health, as instructed", flush=True)
    sys.exit(3)

print(f"build {os.environ.get('FAKE_BUILD', '9999')} "
      f"({os.environ.get('FAKE_COMMIT', 'deadbee')}) with cc (fake) for x86_64-linux-gnu",
      flush=True)
# A real llama-server always prints which file it loaded, and the runner now
# requires a healthy arm-run to have observed the target from somewhere. This
# fake used to print no loader line at all, so every end-to-end test exercised
# the path where provenance is simply absent - which was exactly the fail-open
# the fourth review found.
if MODEL and os.environ.get("FAKE_NO_LOADER_LINE") != "1":
    print(f"llama_model_loader: loaded meta data with 45 key-value pairs and "
          f"579 tensors from {MODEL} (version GGUF V3 (latest))", flush=True)

_PARENT = os.getppid()


def _exit_when_orphaned() -> None:
    """Exit if the process that launched this one goes away.

    `retest_runner.py` kills the server in a `finally`, which does not run when
    the runner itself is killed - and then this outlives it, holding its port
    with nothing left to talk to. Two of these were found alive after two days
    and one day, orphaned by interrupted test runs, which is the same shape as
    the mutation suite's interrupted restore in ERRATA. One second of polling
    is enough: nothing here needs to survive its parent.

    Two conditions, not one. Comparing against the parent recorded at start
    covers a reparented orphan, which lands on the nearest subreaper and only
    on init when there is not one closer. Checking for init as well covers the
    window before this line runs: if the parent dies first, the recorded parent
    is already init and the comparison alone would never fire. The test found
    that, by killing the parent as soon as the pid existed rather than once the
    port was open.
    """
    while True:
        _now = os.getppid()
        if _now != _PARENT or _now == 1:
            os._exit(0)
        time.sleep(1)


threading.Thread(target=_exit_when_orphaned, daemon=True).start()
# A readiness signal for the one test that needs to know the watchdog is armed,
# written only when asked so that no other test sees a byte of difference. The
# open port is not that signal: it says the socket bound, which is neither
# necessary nor sufficient for the parent to have been recorded, and a test that
# waited for it passed locally and failed on CI in seven milliseconds.
if os.environ.get("FAKE_READY_FILE"):
    with open(os.environ["FAKE_READY_FILE"], "w") as _rf:
        _rf.write(str(_PARENT))
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
