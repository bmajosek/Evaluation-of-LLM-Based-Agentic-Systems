SYSTEM_PROMPT = (
    "You are a Python bug-fixing assistant. Your job is to fix exactly one function.\n"
    "Return ONLY a single fenced Python code block that contains the full corrected function.\n"
    "Keep the same function name and parameters. Do not add prints or extra top-level code.\n"
    "If you use typing features (e.g., List, Tuple), include the necessary imports.\n"
    "Respect the intended return type: if a float is expected, do not return booleans, etc."
)

USER_TEMPLATE = """\
Fix the function `{entry_point}` so all tests pass.

You may use these imports (optional):
{imports}

Function declaration (for reference):
{declaration}

Buggy function body (for reference):
{buggy_body}

Return the full corrected function in ONE fenced Python code block, with any needed imports.
Do not output any explanations.
"""

CRITIC_TEMPLATE = """\
Your previous attempt did not pass the tests.

Here is the error trace:
{error}

Here is your previous function:
{prev_code}

Please return an improved, fully corrected version of the SAME function (same name & signature).
Return ONLY one fenced Python code block. Include any needed imports inside the block.
"""