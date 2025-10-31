import argparse
from evaluation.humaneval_eval import evaluate_on_humanevalfix, DEFAULT_MODEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (default: Qwen/Qwen3-0.6B)")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N tasks")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--retries", type=int, default=4, help="Internal reflection attempts")
    ap.add_argument("--save-dir", type=str, default="runs", help="Directory for results.jsonl")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    evaluate_on_humanevalfix(
        model_name=args.model,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        retries=args.retries,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
