"""
Convert tiktoken encodings to the C++ tokenizer format.

Default: nanochat's tokenizer.pkl (pickled tiktoken Encoding) ->
- tokenizer.json: {"pattern", "mergeable_ranks": [base64 bytes, index = rank], "special_tokens": {name: id}}
- token_bytes.bin: int32 little-endian byte length per token id (0 for special tokens)

With --pretrained gpt2 cl100k_base: tiktoken's pretrained encodings -> <name>.json (for tok_eval baselines).
"""
import argparse
import base64
import json
import os
import pickle

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")


def encoding_to_json(enc):
    ranks = sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1])
    assert [r for _, r in ranks] == list(range(len(ranks))), "ranks must be contiguous from 0"
    return {
        "pattern": enc._pat_str,
        "mergeable_ranks": [base64.b64encode(b).decode("ascii") for b, _ in ranks],
        "special_tokens": dict(enc._special_tokens),
    }


def token_bytes(enc):
    special = set(enc._special_tokens.values())
    return np.array([0 if i in special else len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)],
                    dtype="<i4")


def export_pretrained(name, out_dir):
    import tiktoken
    enc = tiktoken.get_encoding(name)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(encoding_to_json(enc), f)
    print(f"exported {name} ({enc.n_vocab} tokens) to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer-dir", default=os.path.join(CACHE_DIR, "tokenizer"))
    parser.add_argument("--out-dir", help="default: --tokenizer-dir")
    parser.add_argument("--pretrained", nargs="+", help="tiktoken encodings to export, e.g. gpt2 cl100k_base")
    args = parser.parse_args()
    out_dir = args.out_dir or args.tokenizer_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.pretrained:
        for name in args.pretrained:
            export_pretrained(name, out_dir)
        return

    with open(os.path.join(args.tokenizer_dir, "tokenizer.pkl"), "rb") as f:
        enc = pickle.load(f)
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(encoding_to_json(enc), f)
    token_bytes(enc).tofile(os.path.join(out_dir, "token_bytes.bin"))
    print(f"exported {enc.n_vocab} tokens to {out_dir}")


if __name__ == "__main__":
    main()
