// Port of nanochat tests/test_tokenizer.py: trains a tiny tokenizer in-process, no data needed.
#include <gtest/gtest.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <set>

#include "nanochat/tokenizer/tokenizer.h"

using namespace nanochat;
using json = nlohmann::json;
using Ids = std::vector<token_t>;

static std::vector<std::string> tiny_corpus() {
  std::vector<std::string> corpus;
  for (int i = 0; i < 8; ++i) {
    corpus.insert(
          corpus.end(),
          {"The quick brown fox jumps over the lazy dog.", "hello world, hello tokenizer, hello hello hello",
           "Numbers like 12345 and unicode like naïve café 你好 🙂 should survive.", "def f(x):\n    return x + 1\n"});
  }
  return corpus;
}

static const uint32_t kTinyVocab = 256 + SPECIAL_TOKENS.size() + 35;

static const Tokenizer& tok() {
  static const Tokenizer t = [] {
    bool given = false;
    return Tokenizer::train(
          [&](std::vector<std::string>& docs) {
            if (given)
              return false;
            docs = tiny_corpus();
            return given = true;
          },
          kTinyVocab, 4);
  }();
  return t;
}

static Ids concat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

TEST(Tokenizer, VocabSize) {
  EXPECT_EQ(tok().vocab_size(), kTinyVocab);
}

TEST(Tokenizer, EncodeDecodeRoundtrip) {
  for (std::string text : {"hello world", "naïve café 你好 🙂", "unseen tokens: zqxjkv"})
    EXPECT_EQ(tok().decode(tok().encode(text)), text);
}

TEST(Tokenizer, SpecialTokens) {
  std::set<token_t> ids;
  for (auto name : SPECIAL_TOKENS)
    ids.insert(tok().encode_special(name));
  EXPECT_EQ(ids.size(), SPECIAL_TOKENS.size());
  EXPECT_EQ(tok().bos_token_id(), tok().encode_special("<|bos|>"));
  // specials are not special-cased in ordinary text
  EXPECT_GT(tok().encode("<|bos|>").size(), 1);
}

TEST(Tokenizer, EncodePrependAppend) {
  auto ids = tok().encode("hello", tok().bos_token_id(), tok().encode_special("<|user_end|>"));
  EXPECT_EQ(ids.front(), tok().bos_token_id());
  EXPECT_EQ(ids.back(), tok().encode_special("<|user_end|>"));
}

TEST(Tokenizer, EncodeBatch) {
  std::vector<std::string> texts = {"hello", "world", "naïve", ""};
  auto batch = tok().encode_batch(texts);
  ASSERT_EQ(batch.size(), texts.size());
  for (size_t i = 0; i < texts.size(); ++i)
    EXPECT_EQ(batch[i], tok().encode(texts[i]));
}

TEST(Tokenizer, SaveLoadRoundtrip) {
  auto path = std::filesystem::temp_directory_path() / ("nanochat_tok_" + std::to_string(::getpid()) + ".json");
  tok().save(path);
  auto loaded = Tokenizer::load(path);
  std::filesystem::remove(path);
  EXPECT_EQ(loaded.vocab_size(), tok().vocab_size());
  EXPECT_EQ(loaded.token_bytes(), tok().token_bytes());
  EXPECT_EQ(loaded.encode("hello naïve world"), tok().encode("hello naïve world"));
}

TEST(Tokenizer, RenderConversationMasks) {
  json conv = {
        {"messages",
         {{{"role", "user"}, {"content", "hi"}},
          {{"role", "assistant"}, {"content", "hello!"}},
          {{"role", "user"}, {"content", "bye"}},
          {{"role", "assistant"}, {"content", "later"}}}}};
  auto [ids, mask] = tok().render_conversation(conv);
  ASSERT_EQ(ids.size(), mask.size());
  EXPECT_EQ(ids[0], tok().bos_token_id());
  EXPECT_EQ(mask[0], 0);
  const token_t assistant_end = tok().encode_special("<|assistant_end|>");
  Ids supervised;
  for (size_t i = 0; i < ids.size(); ++i)
    if (mask[i])
      supervised.push_back(ids[i]);
  EXPECT_EQ(
        supervised,
        concat(concat(concat(tok().encode("hello!"), {assistant_end}), tok().encode("later")), {assistant_end}));
  const token_t user_start = tok().encode_special("<|user_start|>");
  for (size_t i = 0; i < ids.size(); ++i)
    if (ids[i] == user_start)
      EXPECT_EQ(mask[i], 0);
}

TEST(Tokenizer, RenderConversationSystemMessageMerged) {
  json without_system = {
        {"messages",
         {{{"role", "user"}, {"content", "sys prompt\n\nhi"}}, {{"role", "assistant"}, {"content", "yo"}}}}};
  json with_system = {
        {"messages",
         {{{"role", "system"}, {"content", "sys prompt"}},
          {{"role", "user"}, {"content", "hi"}},
          {{"role", "assistant"}, {"content", "yo"}}}}};
  auto a = tok().render_conversation(with_system), b = tok().render_conversation(without_system);
  EXPECT_EQ(a.ids, b.ids);
  EXPECT_EQ(a.mask, b.mask);
}

TEST(Tokenizer, RenderConversationToolParts) {
  json conv = {
        {"messages",
         {{{"role", "user"}, {"content", "add"}},
          {{"role", "assistant"},
           {"content",
            {{{"type", "text"}, {"text", "sure"}},
             {{"type", "python"}, {"text", "1+1"}},
             {{"type", "python_output"}, {"text", "2"}},
             {{"type", "text"}, {"text", "it is 2"}}}}}}}};
  auto [ids, mask] = tok().render_conversation(conv);
  auto index = [&](std::string_view name) {
    return std::ranges::find(ids, tok().encode_special(name)) - ids.begin();
  };
  EXPECT_EQ(mask[index("<|python_start|>")], 1);
  for (auto i = index("<|output_start|>"); i <= index("<|output_end|>"); ++i)
    EXPECT_EQ(mask[i], 0);
}

TEST(Tokenizer, RenderConversationTruncation) {
  std::string hello, world;
  for (int i = 0; i < 100; ++i) {
    hello += "hello ";
    world += "world ";
  }
  json conv = {{"messages", {{{"role", "user"}, {"content", hello}}, {{"role", "assistant"}, {"content", world}}}}};
  auto [ids, mask] = tok().render_conversation(conv, 32);
  EXPECT_EQ(ids.size(), 32);
  EXPECT_EQ(mask.size(), 32);
}

TEST(Tokenizer, RenderForCompletion) {
  json conv = {
        {"messages",
         {{{"role", "user"}, {"content", "hi"}}, {{"role", "assistant"}, {"content", "this gets stripped"}}}}};
  auto ids = tok().render_for_completion(conv);
  EXPECT_EQ(ids.back(), tok().encode_special("<|assistant_start|>"));
  auto stripped = tok().encode("this gets stripped");
  EXPECT_TRUE(std::ranges::search(ids, stripped).empty());
}
