"""Bounded subprocess execution for trusted mathematical Python snippets.

This is defense in depth for trusted training traffic, not an isolation
boundary for hostile code. Production deployments should use containers or
another OS sandbox when inputs are adversarial.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from threading import BoundedSemaphore, Thread
from typing import Any

from .config import ToolConfig


def _effective_uid() -> int | None:
    """Return the effective UID on POSIX without exposing it on Windows."""

    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is None:
        return None
    return get_effective_uid()


DENIED_MODULE_ROOTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "glob",
    "http",
    "importlib",
    "io",
    "multiprocessing",
    "nt",
    "os",
    "pathlib",
    "posix",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "urllib",
    "webbrowser",
}
DENIED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
DENIED_ATTRIBUTES = {
    "chmod",
    "chown",
    "fork",
    "kill",
    "open",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "spawn",
    "system",
    "terminate",
    "unlink",
}
DENIED_REFLECTION_ATTRIBUTES = {
    "__bases__",
    "__builtins__",
    "__class__",
    "__code__",
    "__func__",
    "__globals__",
    "__mro__",
    "__subclasses__",
}


class _DangerVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reason: str | None = None

    def _deny(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in DENIED_MODULE_ROOTS:
                self._deny(f"import {root}")
                return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        root = (node.module or "").split(".", 1)[0]
        if root in DENIED_MODULE_ROOTS:
            self._deny(f"import from {root}")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
            self._deny(f"call {node.func.id}")
            return
        if isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute in DENIED_ATTRIBUTES or attribute.startswith("spawn"):
                self._deny(f"call .{attribute}")
                return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in DENIED_REFLECTION_ATTRIBUTES:
            self._deny(f"attribute {node.attr}")
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in DENIED_REFLECTION_ATTRIBUTES:
            self._deny(f"name {node.id}")


def find_dangerous_operation(code: str) -> str | None:
    """Return a reason for rejected code while allowing syntax errors to run."""

    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return None
    visitor = _DangerVisitor()
    visitor.visit(tree)
    return visitor.reason


PYTHON_ERROR_CATEGORIES = frozenset(
    {
        "syntax_error",
        "name_error",
        "attribute_error",
        "runtime_error",
        "timeout",
        "unsafe_code",
        "invalid_input",
        "sandbox_error",
    }
)

RECOVERABLE_PYTHON_ERROR_CATEGORIES = frozenset(
    {
        "syntax_error",
        "name_error",
        "attribute_error",
        "runtime_error",
        "timeout",
    }
)

PYTHON_EXECUTION_RESULT_VERSION = "trace_opsd.python-execution-result.v1"


@dataclass(frozen=True)
class PythonExecutionResult:
    """Executor-owned outcome; observations never determine execution status."""

    observation: str
    status: str = "ok"
    error_category: str | None = None
    error_origin: str | None = None
    execution_result_version: str = PYTHON_EXECUTION_RESULT_VERSION

    @classmethod
    def failure(
        cls, category: str, detail: object, *, origin: str
    ) -> PythonExecutionResult:
        return cls(_python_error(category, detail), "error", category, origin)


def classify_python_error(observation: object) -> str | None:
    """Read the stable category emitted by :class:`PythonExecutor`."""

    if not isinstance(observation, str):
        return None
    prefix = "Error [python:"
    start = observation.find(prefix)
    if start < 0:
        return None
    end = observation.find("]", start + len(prefix))
    if end < 0:
        return None
    category = observation[start + len(prefix) : end]
    return category if category in PYTHON_ERROR_CATEGORIES else None


def is_recoverable_python_error(category: object) -> bool:
    return category in RECOVERABLE_PYTHON_ERROR_CATEGORIES


def _python_error(category: str, detail: object) -> str:
    """Emit a compact error that is useful as the next-turn observation.

    Do not repeat XML tags or protocol feedback here: base models were copying
    those strings into the next Python snippet.
    """

    normalized = str(detail).strip() or "execution failed"
    return (
        f"Error [python:{category}]: {normalized}. "
        "Submit self-contained executable Python only."
    )


_CHILD_INITIALIZATION = r"""
import json
import os
import sys

import cmath
import fractions
import itertools
import math
import random
import numpy as np
import scipy
import sympy as sp
from fractions import Fraction
from itertools import combinations, product
from math import comb, factorial, gcd, log10
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad
from scipy.optimize import fsolve, minimize
from sympy import (
    Abs,
    Eq,
    Function,
    Matrix,
    Point,
    Poly,
    Rational,
    Symbol,
    binomial,
    cbrt,
    cos,
    divisors,
    exp,
    expand,
    factor,
    integrate,
    limit,
    log,
    roots,
    simplify,
    sin,
    solve,
    sqrt,
    symbols,
)
from sympy.ntheory.modular import crt

code_path = sys.argv[1]
limits = json.loads(sys.argv[2])
with open(code_path, "r", encoding="utf-8") as source:
    user_code = source.read()

if os.name == "posix":
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu"], limits["cpu"]))
    memory = limits["memory"]
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    output = limits["output"]
    resource.setrlimit(resource.RLIMIT_FSIZE, (output, output))
    nofile = limits["nofile"]
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    uid = limits.get("uid")
    gid = limits.get("gid")
    if uid is not None and gid is not None:
        if os.geteuid() == 0:
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
        elif uid != os.geteuid() or gid != os.getegid():
            raise PermissionError("executor UID/GID requires root privileges")

x, y, z = symbols("x y z")

namespace = {
    "__name__": "__main__",
    "__file__": "<python-tool>",
    "math": math,
    "np": np,
    "scipy": scipy,
    "sp": sp,
    "sympy": sp,
    "cmath": cmath,
    "fractions": fractions,
    "itertools": itertools,
    "random": random,
    "x": x,
    "y": y,
    "z": z,
    "symbols": symbols,
    "Symbol": Symbol,
    "Eq": Eq,
    "solve": solve,
    "Rational": Rational,
    "simplify": simplify,
    "factor": factor,
    "expand": expand,
    "Matrix": Matrix,
    "sqrt": sqrt,
    "comb": comb,
    "factorial": factorial,
    "Fraction": Fraction,
    "combinations": combinations,
    "product": product,
    "gcd": gcd,
    "log10": log10,
    "binomial": binomial,
    "cbrt": cbrt,
    "cos": cos,
    "sin": sin,
    "integrate": integrate,
    "exp": exp,
    "log": log,
    "Point": Point,
    "Poly": Poly,
    "Polynomial": Polynomial,
    "Abs": Abs,
    "Function": Function,
    "divisors": divisors,
    "limit": limit,
    "roots": roots,
    "crt": crt,
    "quad": quad,
    "fsolve": fsolve,
    "minimize": minimize,
}
""".strip()

# This file is opened before privilege dropping and remains separate from the
# snippet's stdout/stderr and working directory. A printed error cannot forge an
# execution result. This is instrumentation for trusted snippets, not a hostile
# code isolation boundary.
_CHILD_BOOTSTRAP = (
    r"""
import json
import os
import sys

_control = open(sys.argv[3], "a", encoding="utf-8")

def _report(phase, status=None, exc=None):
    value = {"phase": phase, "status": status}
    if exc is not None:
        value["exception_type"] = type(exc).__name__
        value["detail"] = str(exc)[:8192]
    _control.write(json.dumps(value) + "\n")
    _control.flush()
    os.fsync(_control.fileno())

_report("bootstrap")
try:
""".lstrip()
    + indent(_CHILD_INITIALIZATION, "    ")
    + r"""
except BaseException as _exc:
    _report("bootstrap", "error", _exc)
    raise

_report("user_code")
try:
    exec(compile(user_code, "<python-tool>", "exec"), namespace, namespace)
except BaseException as _exc:
    _report("user_code", "error", _exc)
    raise
else:
    _report("user_code", "ok")
"""
)


class PythonExecutor:
    def __init__(self, config: ToolConfig) -> None:
        self.config = config
        self._slots = BoundedSemaphore(config.python_max_concurrency)

    def capabilities(self) -> dict[str, Any]:
        posix = os.name == "posix"
        is_root = posix and _effective_uid() == 0
        return {
            "platform": os.name,
            "posix_resource_limits": posix,
            "process_group_cleanup": posix,
            "privilege_drop_configured": (
                self.config.executor_uid is not None
                and self.config.executor_gid is not None
            ),
            "service_is_root": is_root,
            "adversarial_sandbox": False,
            "execution_result_version": PYTHON_EXECUTION_RESULT_VERSION,
        }

    def _root_configuration_error(self) -> str | None:
        if (
            os.name == "posix"
            and _effective_uid() == 0
            and (self.config.executor_uid is None or self.config.executor_gid is None)
        ):
            return _python_error(
                "sandbox_error",
                "Python Executor is running as root; configure executor_uid and "
                "executor_gid before executing code",
            )
        return None

    def execute(self, code: str) -> str:
        """Execute validated source and always return observation content."""

        return self.execute_result(code).observation

    def execute_result(self, code: str) -> PythonExecutionResult:
        """Return trusted status, failure category, and execution phase."""

        root_error = self._root_configuration_error()
        if root_error is not None:
            return PythonExecutionResult(
                root_error, "error", "sandbox_error", "sandbox"
            )
        acquired = self._slots.acquire(timeout=self.config.python_queue_timeout_seconds)
        if not acquired:
            return PythonExecutionResult.failure(
                "timeout", "Python Executor queue wait timed out", origin="queue"
            )
        try:
            return self._execute_in_subprocess(code)
        except BaseException as exc:
            return PythonExecutionResult.failure(
                "sandbox_error",
                f"sandbox failed ({type(exc).__name__}: {exc})",
                origin="sandbox",
            )
        finally:
            self._slots.release()

    def _execute_in_subprocess(self, code: str) -> PythonExecutionResult:
        try:
            ast.parse(code, mode="exec")
        except (SyntaxError, ValueError, TypeError) as exc:
            detail = getattr(exc, "msg", None) or str(exc)
            return PythonExecutionResult.failure(
                "syntax_error", detail, origin="user_code"
            )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            control_path = tmp_path / "control"
            work_path = tmp_path / "work"
            control_path.mkdir()
            work_path.mkdir()
            code_path = control_path / "user_code.py"
            code_path.write_text(code, encoding="utf-8")
            outcome_path = control_path / "execution.jsonl"
            outcome_path.touch(mode=0o600)
            if (
                os.name == "posix"
                and _effective_uid() == 0
                and self.config.executor_uid is not None
                and self.config.executor_gid is not None
            ):
                os.chown(
                    work_path,
                    self.config.executor_uid,
                    self.config.executor_gid,
                )
                work_path.chmod(0o700)

            limits = json.dumps(
                {
                    "cpu": self.config.python_cpu_seconds,
                    "memory": self.config.python_memory_limit_mb * 1024 * 1024,
                    "output": self.config.python_output_limit_bytes,
                    "nofile": self.config.python_nofile_limit,
                    "uid": self.config.executor_uid,
                    "gid": self.config.executor_gid,
                },
                separators=(",", ":"),
            )
            command = [
                sys.executable,
                "-I",
                "-u",
                "-c",
                _CHILD_BOOTSTRAP,
                str(code_path),
                limits,
                str(outcome_path),
            ]
            environment = self._subprocess_environment(str(work_path))
            creationflags = 0
            popen_kwargs: dict[str, Any] = {"start_new_session": True}
            if os.name == "nt":
                popen_kwargs = {}
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

            try:
                process = subprocess.Popen(
                    command,
                    cwd=work_path,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                    creationflags=creationflags,
                    **popen_kwargs,
                )
            except OSError as exc:
                return PythonExecutionResult.failure(
                    "sandbox_error",
                    f"Python process startup failed ({exc})",
                    origin="bootstrap",
                )
            stdout_capture = _CappedPipeCapture(
                process.stdout,
                self.config.python_output_limit_bytes,
            )
            stderr_capture = _CappedPipeCapture(
                process.stderr,
                self.config.python_output_limit_bytes,
            )
            stdout_capture.start()
            stderr_capture.start()
            try:
                return_code = process.wait(timeout=self.config.python_timeout_seconds)
            except subprocess.TimeoutExpired:
                self._kill_process_tree(process)
                process.wait()
                stdout_capture.finish()
                stderr_capture.finish()
                outcome = self._read_child_outcome(outcome_path)
                return PythonExecutionResult.failure(
                    "timeout",
                    "execution timed out after "
                    f"{self.config.python_timeout_seconds:g}s",
                    origin="execution_limit"
                    if outcome.get("phase") == "user_code"
                    else "bootstrap",
                )

            stdout, stdout_truncated = stdout_capture.finish()
            stderr, _ = stderr_capture.finish()
            outcome = self._read_child_outcome(outcome_path)
            if (
                return_code == 0
                and outcome.get("status") == "ok"
                and outcome.get("phase") == "user_code"
            ):
                output = stdout.strip()
                if not output:
                    return PythonExecutionResult("(no output)")
                if stdout_truncated:
                    output += "\n...[raw output truncated]"
                return PythonExecutionResult(output)

            last_error = self._last_nonempty_line(stderr)
            if return_code < 0:
                return PythonExecutionResult.failure(
                    "runtime_error",
                    f"execution terminated by signal {-return_code}",
                    origin="unknown",
                )
            if outcome.get("status") == "error":
                phase = outcome.get("phase")
                exception_name = str(outcome.get("exception_type", "Exception"))
                detail = f"{exception_name}: {outcome.get('detail', '')}"
                category = (
                    self._exception_category(exception_name)
                    if phase == "user_code"
                    else "sandbox_error"
                )
                return PythonExecutionResult.failure(
                    category,
                    detail,
                    origin="user_code" if phase == "user_code" else "bootstrap",
                )
            return PythonExecutionResult.failure(
                "runtime_error",
                last_error or f"execution failed with exit code {return_code}",
                origin="unknown",
            )

    @staticmethod
    def _read_child_outcome(path: Path) -> dict[str, Any]:
        outcome: dict[str, Any] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return outcome
        for line in lines:
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                break  # A kill may leave the final record partially written.
            if not isinstance(value, dict) or value.get("phase") not in {
                "bootstrap",
                "user_code",
            }:
                return {}
            outcome = value
        return outcome

    @staticmethod
    def _exception_category(last_error: str) -> str:
        if "NameError" in last_error or "UnboundLocalError" in last_error:
            return "name_error"
        if "AttributeError" in last_error:
            return "attribute_error"
        if "SyntaxError" in last_error or "IndentationError" in last_error:
            return "syntax_error"
        return "runtime_error"

    @staticmethod
    def _last_nonempty_line(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[Any]) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass

    @staticmethod
    def _subprocess_environment(tmp_dir: str) -> dict[str, str]:
        allowed = {
            "LANG",
            "LC_ALL",
            "PATH",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "WINDIR",
        }
        environment = {
            key: value for key, value in os.environ.items() if key in allowed
        }
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TEMP": tmp_dir,
                "TMP": tmp_dir,
                "TMPDIR": tmp_dir,
            }
        )
        return environment


class _CappedPipeCapture:
    """Continuously drain a child pipe while retaining only a bounded prefix."""

    def __init__(self, stream: Any, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self._thread = Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> tuple[str, bool]:
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            try:
                self.stream.close()
            except OSError:
                pass
            self._thread.join(timeout=1)
        return self.data.decode("utf-8", errors="replace"), self.truncated

    def _read(self) -> None:
        try:
            while True:
                chunk = self.stream.read(8192)
                if not chunk:
                    break
                remaining = self.limit - len(self.data)
                if remaining > 0:
                    self.data.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.stream.close()
            except OSError:
                pass
