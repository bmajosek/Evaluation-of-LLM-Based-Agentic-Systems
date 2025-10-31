import os
import sys
import tempfile
import subprocess
from typing import Tuple


def _compose_program(imports: str, func_src: str, tests: str) -> str:
    """Builds a single Python file for sandboxed execution."""
    parts = []
    if (imports or "").strip():
        parts.append(imports.strip())
    parts.append(func_src.strip())
    parts.append("ns = globals()")
    parts.append(tests.strip())
    return "\n\n".join(parts)


def _write_temp_program(code: str) -> str:
    """Writes code to a temporary file."""
    fd, path = tempfile.mkstemp(prefix="prog_", suffix=".py", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _limit_resources_on_posix():
    """Applies resource limits on POSIX systems."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_DATA, (64 * 1024 * 1024, 64 * 1024 * 1024))
    except Exception:
        pass


def run_code_with_tests(func_src: str, tests: str, imports: str) -> Tuple[bool, str]:
    """Executes code with tests in an isolated subprocess."""
    program = _compose_program(imports, func_src, tests)
    path = _write_temp_program(program)

    try:
        cmd = [sys.executable, "-I", path]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={},
            start_new_session=True,
            preexec_fn=_limit_resources_on_posix if os.name == "posix" else None,
        )
        out, err = proc.communicate(timeout=8)
        ok = (proc.returncode == 0)
        return ok, ("" if ok else (err or out))
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        return False, "TimeoutExpired: execution exceeded time limit"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
