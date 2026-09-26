// Evaluate compression ratio of the tokenizer against GPT-2 and GPT-4. Port of scripts/tok_eval.py.
#include <format>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "nanochat/common.h"
#include "nanochat/dataset.h"
#include "nanochat/flags.h"
#include "nanochat/tokenizer/tokenizer.h"
#include "tok_eval_texts.h"

using namespace nanochat;
namespace fs = std::filesystem;

template <typename... Args>
static void println(std::format_string<Args...> fmt, Args&&... args) {
  std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

struct Result {
  size_t bytes, tokens;
  double ratio;
};

using Results = std::map<std::string, Result>;
using Texts = std::vector<std::pair<std::string, std::string>>;

static const char *GREEN = "\033[92m", *RED = "\033[91m", *RESET = "\033[0m";

static std::string first_row_group_joined(const fs::path& data_dir, Split split) {
  ParquetBatches batches(data_dir, split);
  std::vector<std::string> docs;
  std::string joined;
  if (batches.next(docs))
    for (size_t i = 0; i < docs.size(); ++i)
      joined += (i ? "\n" : "") + docs[i];
  return joined;
}

static Results evaluate(const Tokenizer& tok, const Texts& texts) {
  Results results;
  for (const auto& [name, text] : texts) {
    const auto ids = tok.encode(text);
    if (tok.decode(ids) != text)
      throw std::runtime_error("decode mismatch on " + name);
    results[name] = {text.size(), ids.size(), static_cast<double>(text.size()) / static_cast<double>(ids.size())};
  }
  return results;
}

static void print_comparison(
      const std::string& baseline_name, const Results& baseline, const Results& ours, const Texts& texts) {
  println("\nComparison with {}:", baseline_name);
  println("{}", std::string(95, '='));
  println(
        "{:<10} {:<8} {:<15} {:<15} {:<12} {:<10}", "Text Type", "Bytes", baseline_name, "Ours", "Relative", "Better");
  println("{:10} {:8} {:<7} {:<7} {:<7} {:<7} {:<12}", "", "", "Tokens", "Ratio", "Tokens", "Ratio", "Diff %");
  println("{}", std::string(95, '-'));
  for (const auto& [name, _] : texts) {
    const auto &b = baseline.at(name), &o = ours.at(name);
    // positive means ours uses fewer tokens
    const double relative_diff = (static_cast<double>(b.tokens) - static_cast<double>(o.tokens)) /
                                 static_cast<double>(b.tokens) * 100;
    const char *baseline_color = "", *ours_color = "", *diff_color = "";
    std::string better = "Tie";
    if (b.ratio > o.ratio) {
      baseline_color = GREEN, ours_color = RED, diff_color = RED;
      better = baseline_name;
    }
    else if (o.ratio > b.ratio) {
      baseline_color = RED, ours_color = GREEN, diff_color = GREEN;
      better = "Ours";
    }
    println(
          "{:<10} {:<8} {}{:<7}{} {}{:<7.2f}{} {}{:<7}{} {}{:<7.2f}{} {}{:+7.1f}%{}     {:<10}", name, b.bytes,
          baseline_color, b.tokens, RESET, baseline_color, b.ratio, RESET, ours_color, o.tokens, RESET, ours_color,
          o.ratio, RESET, diff_color, relative_diff, RESET, better);
  }
}

static int run(int argc, char** argv) {
  Flags flags(argc, argv, "Evaluate compression ratio of the tokenizer");
  const fs::path base_dir = flags.str("base-dir", default_base_dir().string(), "nanochat data directory");
  const fs::path tok_dir = base_dir / "tokenizer";
  const fs::path ours_path = flags.str("tokenizer", (tok_dir / "tokenizer.json").string(), "our tokenizer");
  const fs::path gpt2_path = flags.str(
        "gpt2", (tok_dir / "gpt2.json").string(), "GPT-2 encoding (tools/export_tokenizer.py --pretrained gpt2)");
  const fs::path gpt4_path = flags.str(
        "gpt4", (tok_dir / "cl100k_base.json").string(),
        "GPT-4 encoding (tools/export_tokenizer.py --pretrained cl100k_base)");
  flags.done();

  // the tokenizer was trained on earlier train shards, so it has seen the train text
  const fs::path data_dir = base_dir / "base_data_climbmix";
  Texts texts = {
        {"news", std::string(news_text)},       {"korean", std::string(korean_text)},
        {"code", std::string(code_text)},       {"math", std::string(math_text)},
        {"science", std::string(science_text)}, {"climbmix-train", first_row_group_joined(data_dir, Split::Train)}};
  if (auto val_text = first_row_group_joined(data_dir, Split::Val); !val_text.empty())
    texts.emplace_back("climbmix-val", std::move(val_text));

  const auto gpt2 = Tokenizer::load(gpt2_path, "<|endoftext|>");
  const auto gpt4 = Tokenizer::load(gpt4_path, "<|endoftext|>");
  const auto ours = Tokenizer::load(ours_path);
  const Results gpt2_results = evaluate(gpt2, texts), gpt4_results = evaluate(gpt4, texts),
                ours_results = evaluate(ours, texts);

  println("\nVocab sizes:");
  println("GPT-2: {}", gpt2.vocab_size());
  println("GPT-4: {}", gpt4.vocab_size());
  println("Ours: {}", ours.vocab_size());
  print_comparison("GPT-2", gpt2_results, ours_results, texts);
  print_comparison("GPT-4", gpt4_results, ours_results, texts);
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
