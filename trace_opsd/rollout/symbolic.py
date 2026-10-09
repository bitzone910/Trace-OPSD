"""Thread-safe math-verify calls with a reusable, killable CPU subprocess."""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

VERIFIER_REVISION = "trace_opsd.math-verify.process.v1"


class SymbolicBackend:
    def __init__(self, *, timeout: float = 15, command: list[str] | None = None):
        self.timeout = timeout
        self.command = command or [sys.executable, "-u", str(Path(__file__).resolve())]
        self._lock = threading.Lock()
        self._process = None
        self._responses = None

    @staticmethod
    def _read_output(stream, responses):
        try:
            for line in stream:
                responses.put(line)
        except (OSError, ValueError):
            pass  # The watchdog may close the pipe while the reader is active.
        finally:
            responses.put(None)

    def _stop(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        self._responses = None

    def close(self):
        with self._lock:
            self._stop()

    def _receive(self, timeout):
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            self._stop()
            raise TimeoutError(
                "math-verify subprocess exceeded its wall-time limit"
            ) from exc
        if line is None:
            self._stop()
            raise RuntimeError("math-verify subprocess exited without a response")
        try:
            response = json.loads(line)
        except ValueError:
            self._stop()
            raise RuntimeError("invalid math-verify subprocess response") from None
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def compare(self, reference: str, prediction: str) -> bool:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._stop()
                self._responses = queue.Queue()
                self._process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    ),
                )
                threading.Thread(
                    target=self._read_output,
                    args=(self._process.stdout, self._responses),
                    daemon=True,
                ).start()
                if not self._receive(30).get("ready"):
                    self._stop()
                    raise RuntimeError("math-verify subprocess did not initialize")
            request = json.dumps({"reference": reference, "prediction": prediction})
            try:
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
            except (OSError, ValueError):
                self._stop()
                raise RuntimeError("cannot send math-verify request") from None
            return bool(self._receive(self.timeout)["correct"])


_BACKEND = SymbolicBackend()
atexit.register(_BACKEND.close)


@lru_cache(maxsize=16384)
def compare_math_expressions(reference: str, prediction: str) -> bool:
    return _BACKEND.compare(reference, prediction)


def worker_main():
    try:
        from math_verify import LatexExtractionConfig, parse, verify
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            values = []
            for key in ("reference", "prediction"):
                text = request[key]
                if not isinstance(text, str) or len(text) > 16384:
                    raise ValueError(
                        "symbolic answer must be text of at most 16384 characters"
                    )
                # Parse the whole extracted mathematical answer; otherwise a
                # bare 2\\sqrt{5} can be misread as the prefix number 2.
                values.append(
                    parse(
                        "$" + text + "$",
                        extraction_config=(LatexExtractionConfig(),),
                        fallback_mode="no_fallback",
                        parsing_timeout=None,
                        raise_on_error=True,
                    )
                )
            correct = bool(
                verify(values[0], values[1], timeout_seconds=None, raise_on_error=True)
            )
            response = {"correct": correct}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    worker_main()
