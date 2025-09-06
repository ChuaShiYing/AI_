# main.py — single-sentence CLI, loads from selected model's best_model via factory
from __future__ import annotations
import os, sys, argparse
from translator import load_translator

MODEL_TYPE_MAP = {
    "smt": "smt/run/best_model",
    "nmt": "nmt/run/best_model",
    "hybrid": "hybrid/run/best_model"
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CN→EN Translator (choose model type: smt, nmt, hybrid)")
    ap.add_argument("--model_type", type=str, choices=["smt", "nmt", "hybrid"], default="nmt",
                    help="Which model type to use: smt, nmt, or hybrid")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("text", nargs="*", help="Chinese text (single sentence).")
    args = ap.parse_args()

    # Compose the path to the correct best_model folder
    run_dir = MODEL_TYPE_MAP[args.model_type]
    if not os.path.isdir(run_dir):
        print(f"Model folder '{run_dir}' not found. Please train or place the model first.", file=sys.stderr)
        sys.exit(1)

    tr_load = load_translator(run_dir)
    txt = " ".join(args.text).strip()
    if not txt:
        print("Example:\n  python main.py --model_type nmt 我爱学习。", file=sys.stderr)
        sys.exit(1)

    if args.model_type == "Hybrid":
        out, info, table = tr_load.translator.translate(
        txt,
        num_beams=args.beams,
        max_new_tokens=args.max_new,
        no_repeat_ngram_size=args.no_repeat_ngram,
        length_penalty=args.length_penalty,
        return_info=True,
        return_table=True,   # ✅ 新增
        model_type=args.model_type,
    )
        print("输出:", out)
        print(info)
        print(table)             # ✅ 新增打印表格
    else:
        out, info = tr_load.translator.translate(
        txt,
        num_beams=args.beams,
        max_new_tokens=args.max_new,
        no_repeat_ngram_size=args.no_repeat_ngram,
        length_penalty=args.length_penalty,
        return_info=True,
        model_type=args.model_type,
    )
        print("输出:", out)
        print(info)
