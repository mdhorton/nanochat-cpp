"""
Convert the C++ tokenizer format to Python nanochat's (inverse of export_tokenizer.py).

- tokenizer.json -> tokenizer.pkl (pickled tiktoken Encoding)
- token_bytes.bin -> token_bytes.pt (int32 tensor, for bits per byte)

Run after tok_train: Python evals (e.g. base_eval CORE) need these to match the tokenizer the model was trained with.
"""
import argparse
import base64
import json
import os
import pickle

import numpy as np
import tiktoken
import torch

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer-dir", default=os.path.join(CACHE_DIR, "tokenizer"))
    args = parser.parse_args()

    with open(os.path.join(args.tokenizer_dir, "tokenizer.json")) as f:
        d = json.load(f)
    enc = tiktoken.Encoding(name="rustbpe", pat_str=d["pattern"],
                            mergeable_ranks={base64.b64decode(b): i for i, b in enumerate(d["mergeable_ranks"])},
                            special_tokens=d["special_tokens"])
    with open(os.path.join(args.tokenizer_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(enc, f)
    token_bytes = np.fromfile(os.path.join(args.tokenizer_dir, "token_bytes.bin"), dtype="<i4")
    torch.save(torch.from_numpy(token_bytes), os.path.join(args.tokenizer_dir, "token_bytes.pt"))
    print(f"imported {enc.n_vocab} tokens in {args.tokenizer_dir}")


if __name__ == "__main__":
    main()
