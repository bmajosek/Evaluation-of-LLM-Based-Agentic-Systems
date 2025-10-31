from __future__ import annotations

import json
import os
import random
from typing import Callable, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.agent import create_agent, run_agent
from agent.prompts import SYSTEM_PROMPT

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def set_seed(seed: int = 123) -> None:
    """Make runs more reproducible (still uses light sampling)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 384,
) -> Callable[[str], str]:
    """
    Returns a callable(user_prompt: str) -> str using chat templates when available.
    """
    print(f"🔹 Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def generate(user_prompt: str) -> str:
        if hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tok(text, return_tensors="pt").to(mdl.device)
        else:
            inputs = tok(SYSTEM_PROMPT + "\n\n" + user_prompt, return_tensors="pt").to(mdl.device)

        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                eos_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def evaluate_on_humanevalfix(
    model_name: str = DEFAULT_MODEL,
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
    retries: int = 4,
    save_dir: str = "runs",
    seed: int = 2002,
) -> float:
    """
    Evaluate the agent on HumanEvalFix (Python split of HumanEvalPack) and compute pass@1.
    """
    set_seed(seed)

    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    print("Dataset downloaded.")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    model = load_model(model_name, max_new_tokens)
    agent = create_agent(model)

    total = 0
    passed = 0

    for ex in ds:
        total += 1
        task_id: str = ex["task_id"]
        imports: str = ex.get("import") or ""
        entry_point: str = ex["entry_point"]
        buggy_body: str = ex["buggy_solution"]
        declaration: str = ex["declaration"]
        tests: str = ex["test"]

        print("#" * 60)
        print(f"Task: {task_id}")

        result = run_agent(
            agent,
            imports=imports,
            buggy_body=buggy_body,
            entry_point=entry_point,
            declaration=declaration,
            tests=tests,
            max_retries=retries,  # internal repair loop;
        )
        ok = bool(result["passed"])
        if ok:
            passed += 1

        print("#" * 60)
        print(f"Task: {task_id} | Passed: {ok}")
        if not ok and (result.get("error") or "").strip():
            print("Error:\n", result["error"])
        print("Model code:\n", result["program"])
        print("#" * 60)
        print(f"[{total}] {task_id} – passed {passed}/{total}")

        rec = {
            "task_id": task_id,
            "passed": ok,
            "model": model_name,
            "program": result.get("program", ""),
            "error": result.get("error", ""),
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    score = passed / total if total else 0.0
    print(f"\n pass@1: {score:.2%} (on {total} tasks)")
    print(f" Per-task results saved to: {out_path}")
    return score
