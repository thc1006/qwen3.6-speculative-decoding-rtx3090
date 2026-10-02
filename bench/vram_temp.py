#!/usr/bin/env python3
"""The GDDR6X memory temperature NVML does not expose, read off the card's BAR0.

    sudo python3 bench/vram_temp.py              # one reading, in degrees
    sudo python3 bench/vram_temp.py 1 60         # every second for a minute, as csv
    sudo python3 bench/vram_temp.py --probe      # what it would read, and from where

ERRATA A16 said the memory-junction temperature "needs a sensor the platform does
not offer". It does not. NVML does not expose it, which is a different statement:
`nvidia-smi --query-gpu=temperature.memory` returns `N/A` on this card and the
sensor is a thirty-two bit MMIO register whose low twelve bits are the temperature
in thirty-seconds of a degree.

This file exists because A16's addendum publishes readings taken with it -- idle
against a core reading, and a rise to ninety degrees under a bandwidth load -- and
for a day those readings rested on a script in `/tmp` on one host. A published
measurement whose instrument is not in the tree cannot be reproduced by anyone,
which is the defect this repository has now removed from its probe launcher, its
CI reproduction and its push guard.

The offset and the scaling come from `olealgoritme/gddr6`, which is a published
hardware fact rather than code: that project carries no licence, so nothing of it
is copied here. Two choices differ from it deliberately. It opens `/dev/mem`,
which maps all of physical memory; this maps the card's own `resource0`, so the
mapping is scoped to one device and a wrong offset cannot reach anything else. And
it maps read-write; this maps `PROT_READ`, so there is no path by which it can
write to the card.

Root is required: mapping a PCI BAR needs `CAP_SYS_RAWIO`. The reading is
therefore an instrument the benchmark runner may not be able to reach, and
`reading()` returns the reason rather than raising, in the shape
`bench/retest_runner.py` already uses for `nvidia-smi`.
"""
from __future__ import annotations

import mmap
import os
import pathlib
import sys
import time

# Per ARCHITECTURE, not per card: the register lives at a different offset on each
# memory controller layout, and reading this one on another device returns a
# number that means something else. Keyed by PCI device id, so a card this file
# was not written for is refused rather than silently misread.
OFFSETS = {
    # 10de:2204, GA102, RTX 3090, GDDR6X
    "0x2204": 0x0000E2A8,
}
VENDOR_NVIDIA = "0x10de"
# the low twelve bits, in thirty-seconds of a degree Celsius
MASK, SCALE = 0x00000FFF, 32.0
DEFAULT_PCI = "0000:01:00.0"


def celsius(raw: int) -> float:
    """Degrees from the raw register word.

    Pure, so the arithmetic is tested where there is no card: the mask matters
    because the upper twenty bits are not temperature, and the divisor matters
    because a reading of 1344 is forty-two degrees and not 1344.
    """
    return (raw & MASK) / SCALE


def _sysfs(pci: str) -> pathlib.Path:
    return pathlib.Path("/sys/bus/pci/devices") / pci


def identify(pci: str) -> tuple[str, str]:
    """The vendor and device id sysfs reports, as lowercase hex strings."""
    d = _sysfs(pci)
    return ((d / "vendor").read_text().strip().lower(),
            (d / "device").read_text().strip().lower())


def offset_for(pci: str) -> int:
    """The offset for this device, or a refusal naming what it found instead."""
    vendor, device = identify(pci)
    if vendor != VENDOR_NVIDIA or device not in OFFSETS:
        raise SystemExit(
            f"{pci} is {vendor}:{device}, and this file knows "
            f"{VENDOR_NVIDIA}:{{{', '.join(sorted(OFFSETS))}}}. The offset is per "
            f"architecture: reading it on another device returns a number that "
            f"means something else, so this refuses rather than reports one.")
    return OFFSETS[device]


def read_raw(pci: str = DEFAULT_PCI) -> int:
    """The register word, through a read-only mapping of this card's BAR0."""
    off = offset_for(pci)
    page = mmap.PAGESIZE
    base = (off // page) * page
    with open(_sysfs(pci) / "resource0", "rb") as fh:
        m = mmap.mmap(fh.fileno(), page, offset=base, prot=mmap.PROT_READ)
        try:
            return int.from_bytes(m[off - base:off - base + 4], "little")
        finally:
            m.close()


def _why_refused(pci: str, err: Exception) -> str:
    """A refused mapping, explained, because the errno is not the reason.

    `CONFIG_IO_STRICT_DEVMEM=y` makes the kernel refuse to map any IO region a
    driver has CLAIMED, and the `nvidia` driver claims BAR0. Both routes then fail
    and they look like two unrelated problems: `resource0` gives `EINVAL` and
    `/dev/mem` gives `EPERM`. They are one policy.

    Measured on 2026-10-02: the bench host reads this register and the box these
    sessions run on cannot, with the same card generation and the same offset. An
    instrument that reports `[Errno 22] Invalid argument` sends a reader looking at
    the offset, which is the one thing that was right.
    """
    bits = [f"{type(err).__name__}: {err}"]
    d = pathlib.Path("/sys/bus/pci/devices") / pci
    try:
        bits.append(f"driver={(d / 'driver').resolve().name}")
    except OSError:
        bits.append("driver=none")
    for path, label in (("/sys/kernel/security/lockdown", "lockdown"),):
        try:
            bits.append(f"{label}={pathlib.Path(path).read_text().strip()}")
        except OSError:
            pass
    try:
        rel = os.uname().release
        cfg = pathlib.Path(f"/boot/config-{rel}").read_text(encoding="utf-8")
        strict = "CONFIG_IO_STRICT_DEVMEM=y" in cfg
        bits.append(f"CONFIG_IO_STRICT_DEVMEM={'y' if strict else 'n'}")
        if strict:
            bits.append("the kernel refuses to map an IO region a driver has "
                        "claimed; `iomem=relaxed` on the kernel command line "
                        "disables that check, needs a reboot, and weakens a "
                        "hardening. The offset is not the problem")
    except OSError:
        pass
    return "unavailable: " + "; ".join(bits)


def reading(pci: str = DEFAULT_PCI):
    """Degrees, or the reason there are none.

    The same shape `bench/retest_runner.py` uses for `nvidia-smi`: an instrument
    the run could not reach is recorded as the reason it could not, because a
    missing field and a field that says why are not the same evidence.
    """
    try:
        return celsius(read_raw(pci))
    except SystemExit as e:                                  # a refusal is a reason
        return f"unavailable: {e}"
    except (OSError, ValueError) as e:
        # the mapping, which is the case worth explaining
        return _why_refused(pci, e)
    except Exception as e:                                   # noqa: BLE001
        return f"unavailable: {e}"


def _probe(pci: str) -> None:
    vendor, device = identify(pci)
    res = _sysfs(pci) / "resource0"
    print(f"device      {pci} is {vendor}:{device}")
    print(f"offset      {OFFSETS.get(device, 'unknown for this device')}"
          if device not in OFFSETS else
          f"offset      BAR0 + {OFFSETS[device]:#07x}")
    print(f"mapping     {res}, PROT_READ, one page")
    print(f"privileged  euid {os.geteuid()}, "
          f"{'root' if os.geteuid() == 0 else 'NOT root, so the mapping will fail'}")
    print(f"reading     {reading(pci)}")


def main() -> None:
    argv = sys.argv[1:]
    pci = os.environ.get("GPU_PCI", DEFAULT_PCI)
    if argv and argv[0] == "--probe":
        if len(argv) != 1:
            sys.exit("--probe takes no other argument")
        _probe(pci)
        return
    for a in argv:
        if a.startswith("-"):
            sys.exit(f"unknown option {a!r}; the options are --probe, or "
                     f"INTERVAL [COUNT]")
    if not argv:
        v = reading(pci)
        if isinstance(v, str):
            sys.exit(v)
        print(f"{v:.1f}")
        return
    try:
        interval = float(argv[0])
        count = int(argv[1]) if len(argv) > 1 else 0
    except ValueError:
        sys.exit(f"INTERVAL [COUNT] wants numbers, got {argv!r}")
    if interval <= 0 or count < 0:
        sys.exit(f"INTERVAL must be positive and COUNT may not be negative, "
                 f"got {interval} and {count}")
    print("wall_iso,vram_c")
    n = 0
    while not count or n < count:
        v = reading(pci)
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},"
              f"{v if isinstance(v, str) else f'{v:.1f}'}", flush=True)
        n += 1
        if count and n >= count:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
