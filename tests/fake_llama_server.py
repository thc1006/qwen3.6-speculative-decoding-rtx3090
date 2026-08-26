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
  FAKE_BUILD=<n> FAKE_COMMIT=<sha>
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8080
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])

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
        self.rfile.read(n)
        _seen["n"] += 1
        if FAIL_ON and _seen["n"] == FAIL_ON:
            self._send(500, {"error": "fake failure"})
            return
        ms = 3000.0
        self._send(200, {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "x" * 32, "reasoning_content": ""},
                "logprobs": {"content": [{"token": "x"} for _ in range(N_PREDICT)]},
            }],
            "usage": {"completion_tokens": N_PREDICT},
            "timings": {
                "predicted_n": N_PREDICT, "predicted_ms": ms,
                "predicted_per_second": (N_PREDICT / ms * 1000) if ms else 0,
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
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
