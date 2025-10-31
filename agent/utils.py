import re
from agent.tools import run_code_with_tests

FENCE = re.compile(r"```(?:python)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
UNBOUND_LOCAL = re.compile(r"UnboundLocalError:.*local variable '(\w+)'", re.IGNORECASE)


def first_fenced_python(text: str) -> str:
    """Extracts the first Python code block from a fenced block."""
    m = FENCE.search(text)
    return m.group(1).strip() if m else ""


def extract_def_by_name(text: str, fn_name: str) -> str:
    """Extracts a function definition by its name."""
    pat = re.compile(rf"(def\s+{re.escape(fn_name)}\s*\([^)]*\)\s*:[\s\S]+?)(?=\n\S|$)")
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def extract_or_fallback(raw: str, entry_point: str) -> str:
    """Extracts code from a fenced block, falling back to function name."""
    code = first_fenced_python(raw)
    if not code:
        code = extract_def_by_name(raw, entry_point)
    return code


def autopatch_typing(code: str, tests: str, imports: str):
    """Auto-patches missing typing imports."""
    ok, err = run_code_with_tests(code, tests, imports)
    if ok or not err or "NameError: name" not in err:
        return ok, err, code

    missing_typing = ("List", "Tuple", "Dict", "Set", "Optional")
    if any(f"name '{t}' is not defined" in err for t in missing_typing):
        patched = "from typing import *\n" + code
        ok2, err2 = run_code_with_tests(patched, tests, imports)
        return ok2, err2, patched if ok2 else code
    return ok, err, code


def autopatch_unbound_local(code: str, tests: str, imports: str, err: str):
    """Auto-patches UnboundLocalError by initializing the variable to 0."""
    m = UNBOUND_LOCAL.search(err or "")
    if not m:
        return False, err, code
    var = m.group(1)
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            header_idx = i
            break
    else:
        return False, err, code

    indent = " " * (len(lines[header_idx]) - len(lines[header_idx].lstrip()))
    insert_line = indent + "    " + f"{var} = 0"
    lines.insert(header_idx + 1, insert_line)
    patched = "\n".join(lines)
    ok2, err2 = run_code_with_tests(patched, tests, imports)
    return ok2, err2, patched if ok2 else code
