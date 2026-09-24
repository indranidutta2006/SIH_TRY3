"""Container smoke test for SIH26138 Green Fleet Navigator.

Replicates the Docker container's startup sequence without requiring Docker:
  1. Verifies artifacts exist (build phase already done by ci_generate_fixtures.py
     or run_full_pipeline.py — matches what build_for_deploy.sh produces).
  2. Starts Streamlit in headless mode on an ephemeral port (matches the
     Dockerfile CMD: streamlit run app.py --server.headless true).
  3. Polls the HTTP health endpoint until it responds 200 or times out.
  4. Asserts the response body contains expected Streamlit content.
  5. Terminates the server process cleanly.

Exit codes
----------
  0  — smoke test passed
  1  — smoke test failed (reason printed to stderr)

Usage
-----
  python scripts/smoke_test_container.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_PORT = 8502          # Use 8502 locally to avoid conflicts with any running dashboard
STARTUP_TIMEOUT_S = 60        # Max seconds to wait for Streamlit to become responsive
POLL_INTERVAL_S = 1.0


def _banner(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {msg}\n{bar}")


def _step(label: str, ok: bool, detail: str = "") -> None:
    sym = "[OK]  " if ok else "[FAIL]"
    line = f"  {sym} {label}"
    if detail:
        line += f"  =>  {detail}"
    print(line)
    if not ok:
        sys.exit(1)


def check_required_artifacts() -> None:
    """Stage 1 — verify build artifacts exist (mirrors Dockerfile RUN build_for_deploy.sh)."""
    required = {
        "data/raw/voyages_sample.csv":              Path("data/raw/voyages_sample.csv"),
        "artifacts/metrics/baseline_metrics.json":  Path("artifacts/metrics/baseline_metrics.json"),
        "artifacts/models/hist_gradient_boosting.pkl": Path("artifacts/models/hist_gradient_boosting.pkl"),
    }
    all_ok = True
    for label, p in required.items():
        exists = p.exists() and p.stat().st_size > 0
        if not exists:
            print(f"  [FAIL] Missing artifact: {label}")
            all_ok = False
        else:
            print(f"  [OK]   {label}  ({p.stat().st_size:,} bytes)")
    if not all_ok:
        print("\nRun 'python scripts/ci_generate_fixtures.py' first to generate artifacts.")
        sys.exit(1)


def start_streamlit() -> subprocess.Popen:
    """Stage 2 — start Streamlit headless (mirrors Dockerfile CMD)."""
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"  [OK]   Streamlit process started  (pid={proc.pid}, port={STREAMLIT_PORT})")
    return proc


def poll_until_ready(proc: subprocess.Popen) -> float:
    """Stage 3 — poll HTTP until Streamlit responds 200 or timeout."""
    url = f"http://127.0.0.1:{STREAMLIT_PORT}/"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    attempt = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            print(f"\nStreamlit exited unexpectedly (returncode={proc.returncode}):\n{out}")
            sys.exit(1)
        try:
            t0 = time.monotonic()
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    elapsed = time.monotonic() - t0
                    return elapsed
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        attempt += 1
        print(f"  [wait] Attempt {attempt:>2}  —  not ready yet...", end="\r", flush=True)
        time.sleep(POLL_INTERVAL_S)

    print(f"\n  [FAIL] Streamlit did not respond within {STARTUP_TIMEOUT_S}s")
    proc.terminate()
    sys.exit(1)


def verify_response_body() -> None:
    """Stage 4 — assert response body looks like a Streamlit page."""
    url = f"http://127.0.0.1:{STREAMLIT_PORT}/"
    with urllib.request.urlopen(url, timeout=5) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    # Streamlit always serves an HTML shell with these markers
    assert "streamlit" in body.lower(), "Response body does not contain 'streamlit'"
    assert "<html" in body.lower(), "Response body is not HTML"
    print(f"  [OK]   Response body contains expected Streamlit HTML markers")
    print(f"  [OK]   Content-Length: {len(body):,} bytes")


def main() -> None:
    os.chdir(PROJECT_ROOT)

    _banner("SIH26138 — Container Smoke Test")
    print(f"  Replicating: Dockerfile CMD (streamlit run app.py --server.headless true)")
    print(f"  Port: {STREAMLIT_PORT}  •  Startup timeout: {STARTUP_TIMEOUT_S}s\n")

    # Stage 1
    print("Stage 1/4  Check required build artifacts")
    check_required_artifacts()

    # Stage 2
    print("\nStage 2/4  Start Streamlit headless server")
    proc = start_streamlit()

    try:
        # Stage 3
        print(f"\nStage 3/4  Poll HTTP until ready (max {STARTUP_TIMEOUT_S}s)")
        response_time_s = poll_until_ready(proc)
        print()  # clear the \r line
        _step("HTTP 200 received", True, f"first response in {response_time_s*1000:.0f} ms")

        # Stage 4
        print("\nStage 4/4  Verify response body")
        verify_response_body()

    finally:
        # Always terminate cleanly
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    _banner("SMOKE TEST PASSED")
    print(f"  Streamlit serves HTTP 200 with valid HTML on port {STREAMLIT_PORT}.")
    print(f"  Container entrypoint is healthy.\n")


if __name__ == "__main__":
    main()
