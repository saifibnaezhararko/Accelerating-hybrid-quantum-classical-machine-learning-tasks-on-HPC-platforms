"""Report which quantum/ML backends this machine can run — and prove it.

Run this first on any new machine, and after every simulator install on the HPC
node. ``--verify`` is the evidence for "GPU simulator operational": it builds real
circuits and pushes a forward (and backward, where supported) pass through each
available backend.

::

    python scripts/check_backends.py                 # capability + backend table
    python scripts/check_backends.py --verify        # actually run each backend
    python scripts/check_backends.py --json report.json
    python scripts/check_backends.py --require pennylane-qiskit-aer   # exit 1 if unusable
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from qnlp_hpc.backends import (
    BACKENDS,
    available_backends,
    gpu_simulation_available,
    probe_all,
    verify_all,
)
from qnlp_hpc.paths import resolve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe and verify the quantum simulation backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a real forward/backward pass through every available backend.",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="BACKEND",
        help="Exit non-zero unless this backend is available. Repeatable.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit non-zero unless a GPU-backed simulator is usable.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="Write the full report as JSON (for the benchmark log).",
    )
    return parser


def _table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    line = "  ".join("-" * width for width in widths)
    out = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=True)), line]
    out.extend(
        "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Platform: {platform.platform()}")
    print(f"Python:   {sys.version.split()[0]}\n")

    capabilities = probe_all()

    print("Capabilities")
    print(
        _table(
            [
                (
                    name,
                    capability.status,
                    capability.version or "-",
                    capability.detail,
                )
                for name, capability in capabilities.items()
            ],
            ("component", "found", "version", "detail"),
        )
    )

    availability = available_backends(capabilities)
    print("\nBackends")
    print(
        _table(
            [
                (
                    name,
                    BACKENDS[name].kind,
                    BACKENDS[name].trainer,
                    BACKENDS[name].device,
                    "yes" if ok else "no",
                    reason,
                )
                for name, (ok, reason) in availability.items()
            ],
            ("backend", "kind", "trainer", "device", "usable", "reason"),
        )
    )

    verifications = []
    if args.verify:
        print("\nVerification (real forward/backward pass)")
        verifications = verify_all(capabilities)
        print(
            _table(
                [
                    (
                        result.backend,
                        result.status,
                        f"{result.seconds:.2f}s" if result.seconds else "-",
                        result.detail,
                    )
                    for result in verifications
                ],
                ("backend", "status", "time", "detail"),
            )
        )

    has_gpu = gpu_simulation_available(capabilities)
    print(f"\nGPU quantum simulation: {'available' if has_gpu else 'NOT available'}")

    if args.json:
        destination = resolve(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                    "capabilities": {
                        name: asdict(capability) for name, capability in capabilities.items()
                    },
                    "backends": {
                        name: {"available": ok, "reason": reason}
                        for name, (ok, reason) in availability.items()
                    },
                    "verifications": [asdict(result) for result in verifications],
                    "gpu_simulation": has_gpu,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Wrote report -> {destination}")

    exit_code = 0
    for name in args.require:
        if name not in availability:
            print(f"\nUnknown backend {name!r}. Known: {sorted(BACKENDS)}", file=sys.stderr)
            exit_code = 1
            continue
        ok, reason = availability[name]
        if not ok:
            print(f"\nRequired backend {name!r} unavailable: {reason}", file=sys.stderr)
            exit_code = 1

    if args.require_gpu and not has_gpu:
        print("\nRequired GPU simulation is unavailable.", file=sys.stderr)
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
