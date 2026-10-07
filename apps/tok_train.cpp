// Train the BPE tokenizer on the pretraining data. Port of scripts/tok_train.py.
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <thread>

#include "nanochat/common.h"
#include "nanochat/data/dataset.h"
#include "nanochat/flags.h"
#include "nanochat/tokenizer/bpe.h"
#include "nanochat/tokenizer/tokenizer.h"

using namespace nanochat;
namespace fs = std::filesystem;

template <typename... Args>
static void println(std::format_string<Args...> fmt, Args&&... args) {
  std::cout << std::format(fmt, std::forward<Args>(args)...) << std::endl;
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static int run(int argc, char** argv) {
  Flags flags(argc, argv, "Train a BPE tokenizer");
  const fs::path base_dir = flags.str("base-dir", default_base_dir().string(), "nanochat data directory");
  const auto max_chars = flags.i64("max-chars", 2'000'000'000, "maximum characters to train on");
  const auto doc_cap = flags.i64("doc-cap", 10'000, "maximum characters per document");
  const auto vocab_size = flags.i64("vocab-size", 32768, "vocabulary size, including special tokens");
  const auto threads = flags.i64("threads", std::thread::hardware_concurrency(), "splitting threads");
  const fs::path out_dir = flags.str("out-dir", (base_dir / "tokenizer").string(), "output directory");
  flags.done();
  println("max_chars: {}, doc_cap: {}, vocab_size: {}, threads: {}", max_chars, doc_cap, vocab_size, threads);

  // 1) count split chunks
  auto t0 = std::chrono::steady_clock::now();
  ParquetBatches batches(base_dir / "base_data_climbmix", Split::Train);
  CappedTexts texts(batches, max_chars, doc_cap);
  Pcre2Splitter splitter;
  ChunkCounter counter(splitter, static_cast<int>(threads));
  std::vector<std::string> docs;
  int64_t num_docs = 0;
  while (texts.next(docs)) {
    counter.add(docs);
    num_docs += static_cast<int64_t>(docs.size());
  }
  println(
        "counted {} docs, {} chars, {} unique chunks in {:.2f}s", num_docs, texts.num_chars(), counter.counts().size(),
        seconds_since(t0));

  // 2) merges
  auto t1 = std::chrono::steady_clock::now();
  auto ranks = train_bpe(counter.counts(), static_cast<uint32_t>(vocab_size - SPECIAL_TOKENS.size()));
  println("learned {} merges in {:.2f}s", ranks.size() - 256, seconds_since(t1));
  println("Training time: {:.2f}s", seconds_since(t0));
  auto tokenizer = Tokenizer::from_ranks(std::move(ranks));

  // 3) save tokenizer and per-token byte lengths (for bits per byte)
  tokenizer.save(out_dir / "tokenizer.json");
  println("Saved tokenizer to {}", (out_dir / "tokenizer.json").string());
  const auto token_bytes = tokenizer.token_bytes();
  std::ofstream(out_dir / "token_bytes.bin", std::ios::binary)
        .write(
              reinterpret_cast<const char*>(token_bytes.data()),
              static_cast<std::streamsize>(token_bytes.size() * sizeof(int32_t)));
  println("Saved token_bytes to {}", (out_dir / "token_bytes.bin").string());

  // sanity check
  const std::string test_text = "Hello world! This is a test.\nNumbers: 123, 4567, 89\n"
                                "Contractions: I'm, you're, it's\nSpecial chars: @#$%^&*()\nUnicode: 你好世界 🌍";
  if (tokenizer.decode(tokenizer.encode(test_text)) != test_text) {
    std::cerr << "round trip failed\n";
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
