"""
Export golden tokenizer data from Python nanochat for the C++ parity tests.

Writes to --out-dir:
- train_tiny.json, train_medium.json: rustbpe training inputs/settings and resulting mergeable ranks
- tokenizer.json, token_bytes.bin: the trained nanochat tokenizer (see export_tokenizer.py)
- encode.jsonl: {"text", "chunks", "ids"} for val docs and edge cases
- conversations.jsonl: {"conversation", "ids", "mask", "completion_ids"}
- gpt2.json, cl100k_base.json, encode_<name>.jsonl: pretrained tiktoken encodings and {"text", "ids"}
"""
import argparse
import base64
import json
import os
import pickle
import sys

sys.dont_write_bytecode = True  # keep the nanochat tree untouched

from export_tokenizer import CACHE_DIR


def ranks_b64(tok):
    ranks = sorted(tok.enc._mergeable_ranks.items(), key=lambda kv: kv[1])
    return [base64.b64encode(b).decode("ascii") for b, _ in ranks]


def train_iter(max_chars, doc_cap):
    # same semantics as scripts/tok_train.py text_iterator
    from nanochat.dataset import parquets_iter_batched
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc = doc[:doc_cap]
            nchars += len(doc)
            yield doc
            if nchars > max_chars:
                return


EDGE_CASES = [
    "", " ", "  ", "\n", "\r\n", "\r\n\r\n", "a\r\n\r\n  b", "\t\tx", "x   ", "   x",
    "Hello world! This is a test.", "I'm you're it's we'll they've I'd", "I'M YOU'RE", "don’t",
    "Numbers: 123, 4567, 89, 0.5, 1e10, ٣٤٥ ①②",
    "Unicode: 你好世界 🌍 naïve café Ålborg ÆØÅ ß ﬁ", "🙂🙂🙂 👩‍👩‍👧 🇬🇧",
    "def f(x):\n    return x + 1\n\n\n", "<|bos|> <|user_start|>", "a" * 1000, " " * 100 + "z",
    " nbsp em　ideo​zwsp\u0085nel", "tab\ttab\vvt\fff",
    "ſ ſs K Kelvin", "x́̂ combining",
    "a  \n", "b\n  ", "c \n\n", "d\t \r\n",  # \s++$ in the GPT-2/cl100k patterns
]

CONVERSATIONS = [
    {"messages": [{"role": "user", "content": "Hi there!"}, {"role": "assistant", "content": "Hello! How can I help?"}]},
    {"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "2+2?"},
                  {"role": "assistant", "content": "4"}]},
    {"messages": [{"role": "user", "content": "How many r in strawberry?"},
                  {"role": "assistant", "content": [
                      {"type": "text", "text": "Let me count. "},
                      {"type": "python", "text": "'strawberry'.count('r')"},
                      {"type": "python_output", "text": "3"},
                      {"type": "text", "text": "There are 3."}]},
                  {"role": "user", "content": "thanks"}, {"role": "assistant", "content": "You're welcome 🙂"}]},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=CACHE_DIR, help="nanochat data directory")
    parser.add_argument("--out-dir", help="default: <base-dir>/golden")
    parser.add_argument("--nanochat-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "nanochat"))
    parser.add_argument("--num-docs", type=int, default=1000, help="val docs in encode.jsonl")
    parser.add_argument("--num-pretrained-docs", type=int, default=300, help="val docs in encode_<pretrained>.jsonl")
    parser.add_argument("--medium-max-chars", type=int, default=20_000_000)
    parser.add_argument("--medium-doc-cap", type=int, default=10_000)
    parser.add_argument("--medium-vocab-size", type=int, default=4096)
    args = parser.parse_args()
    args.out_dir = args.out_dir or os.path.join(args.base_dir, "golden")

    os.environ["NANOCHAT_BASE_DIR"] = args.base_dir  # nanochat reads it at import time
    sys.path.insert(0, args.nanochat_dir)
    import regex
    from nanochat.common import get_base_dir
    from nanochat.dataset import parquets_iter_batched
    from nanochat.tokenizer import RustBPETokenizer, SPLIT_PATTERN, SPECIAL_TOKENS
    from export_tokenizer import encoding_to_json, export_pretrained, token_bytes

    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)

    # 1) training parity: tiny in-memory corpus (from tests/test_tokenizer.py) and a medium climbmix slice
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "hello world, hello tokenizer, hello hello hello",
        "Numbers like 12345 and unicode like naïve café 你好 🙂 should survive.",
        "def f(x):\n    return x + 1\n",
    ] * 8
    vocab_size = 256 + len(SPECIAL_TOKENS) + 35
    tok = RustBPETokenizer.train_from_iterator(iter(corpus), vocab_size)
    with open(out("train_tiny.json"), "w") as f:
        json.dump({"corpus": corpus, "vocab_size": vocab_size, "mergeable_ranks": ranks_b64(tok)}, f)

    tok = RustBPETokenizer.train_from_iterator(train_iter(args.medium_max_chars, args.medium_doc_cap),
                                               args.medium_vocab_size)
    with open(out("train_medium.json"), "w") as f:
        json.dump({"max_chars": args.medium_max_chars, "doc_cap": args.medium_doc_cap,
                   "vocab_size": args.medium_vocab_size, "mergeable_ranks": ranks_b64(tok)}, f)
    print("wrote training goldens")

    # 2) the real trained tokenizer
    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
        enc = pickle.load(f)
    with open(out("tokenizer.json"), "w") as f:
        json.dump(encoding_to_json(enc), f)
    token_bytes(enc).tofile(out("token_bytes.bin"))
    tok = RustBPETokenizer(enc, "<|bos|>")

    # 3) encoding parity
    texts = list(EDGE_CASES)
    for batch in parquets_iter_batched(split="val"):
        texts.extend(batch[:args.num_docs - (len(texts) - len(EDGE_CASES))])
        if len(texts) - len(EDGE_CASES) >= args.num_docs:
            break
    pat = regex.compile(SPLIT_PATTERN)
    with open(out("encode.jsonl"), "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text, "chunks": pat.findall(text), "ids": tok.encode(text)}) + "\n")
    print(f"wrote {len(texts)} encode goldens")

    # 4) conversation rendering
    with open(out("conversations.jsonl"), "w") as f:
        for conv in CONVERSATIONS:
            ids, mask = tok.render_conversation(conv)
            completion_ids = tok.render_for_completion(conv)
            f.write(json.dumps({"conversation": conv, "ids": ids, "mask": mask,
                                "completion_ids": completion_ids}) + "\n")
    print(f"wrote {len(CONVERSATIONS)} conversation goldens")

    # 5) pretrained encodings used as tok_eval baselines
    for name in ["gpt2", "cl100k_base"]:
        export_pretrained(name, args.out_dir)
        tok = RustBPETokenizer.from_pretrained(name)
        with open(out(f"encode_{name}.jsonl"), "w") as f:
            for text in texts[:len(EDGE_CASES) + args.num_pretrained_docs]:
                f.write(json.dumps({"text": text, "ids": tok.encode(text)}) + "\n")
    print("wrote pretrained encode goldens")


if __name__ == "__main__":
    main()
