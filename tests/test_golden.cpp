// Parity with Python nanochat, using data from tools/export_golden.py. Skipped when the golden dir is missing.
#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "nanochat/base64.h"
#include "nanochat/dataset.h"
#include "nanochat/tokenizer.h"
#include "test_env.h"

using namespace nanochat;
using json = nlohmann::json;
namespace fs = std::filesystem;

static fs::path golden(const std::string &name) { return test_env().golden_dir / name; }

#define REQUIRE_GOLDEN(name)                                                                                           \
    if (!fs::exists(golden(name)))                                                                                     \
    GTEST_SKIP() << "missing " << golden(name) << " (run tools/export_golden.py)"

static json read_json(const fs::path &path) {
    std::ifstream in(path);
    return json::parse(in);
}

static std::vector<json> read_jsonl(const fs::path &path) {
    std::ifstream in(path);
    std::vector<json> rows;
    for (std::string line; std::getline(in, line);)
        rows.push_back(json::parse(line));
    return rows;
}

// Compares trained ranks to golden base64 ranks, reporting the first mismatch.
static void expect_same_ranks(const Tokenizer &trained, const json &golden_ranks) {
    ASSERT_EQ(trained.vocab_size() - SPECIAL_TOKENS.size(), golden_ranks.size());
    for (token_t id = 0; id < golden_ranks.size(); ++id)
        ASSERT_EQ(trained.decode_single_token_bytes(id), base64_decode(golden_ranks[id].get<std::string>()))
                << "first mismatch at rank " << id;
}

TEST(Golden, TrainTiny) {
    REQUIRE_GOLDEN("train_tiny.json");
    const json g = read_json(golden("train_tiny.json"));
    bool given = false;
    auto tok = Tokenizer::train(
            [&](std::vector<std::string> &docs) {
                if (given)
                    return false;
                docs = g.at("corpus").get<std::vector<std::string>>();
                return given = true;
            },
            g.at("vocab_size").get<uint32_t>(), 4);
    expect_same_ranks(tok, g.at("mergeable_ranks"));
}

TEST(Golden, TrainMedium) {
    REQUIRE_GOLDEN("train_medium.json");
    const json g = read_json(golden("train_medium.json"));
    ParquetBatches batches(test_env().base_dir / "base_data_climbmix", Split::Train);
    CappedTexts texts(batches, g.at("max_chars").get<int64_t>(), g.at("doc_cap").get<int64_t>());
    auto tok = Tokenizer::train([&](std::vector<std::string> &docs) { return texts.next(docs); },
                                g.at("vocab_size").get<uint32_t>(), 16);
    expect_same_ranks(tok, g.at("mergeable_ranks"));
}

class GoldenTokenizer : public testing::Test {
protected:
    static void SetUpTestSuite() {
        if (fs::exists(golden("tokenizer.json")))
            tok_ = std::make_unique<Tokenizer>(Tokenizer::load(golden("tokenizer.json")));
    }
    static void TearDownTestSuite() { tok_.reset(); }
    void SetUp() override {
        if (!tok_)
            GTEST_SKIP() << "missing " << golden("tokenizer.json");
    }
    static inline std::unique_ptr<Tokenizer> tok_;
};

TEST_F(GoldenTokenizer, SplitMatchesPythonRegex) {
    REQUIRE_GOLDEN("encode.jsonl");
    Pcre2Splitter splitter;
    int mismatches = 0;
    for (const auto &row: read_jsonl(golden("encode.jsonl"))) {
        const auto text = row.at("text").get<std::string>();
        std::vector<std::string_view> chunks;
        splitter.split(text, chunks);
        const auto expected = row.at("chunks").get<std::vector<std::string>>();
        if (std::vector<std::string>(chunks.begin(), chunks.end()) != expected && ++mismatches <= 5)
            ADD_FAILURE() << "split mismatch for: " << text.substr(0, 200);
    }
    EXPECT_EQ(mismatches, 0);
}

TEST_F(GoldenTokenizer, EncodeMatchesTiktoken) {
    REQUIRE_GOLDEN("encode.jsonl");
    int mismatches = 0, rows = 0;
    for (const auto &row: read_jsonl(golden("encode.jsonl"))) {
        ++rows;
        const auto text = row.at("text").get<std::string>();
        const auto ids = tok_->encode(text);
        if (ids != row.at("ids").get<std::vector<token_t>>() && ++mismatches <= 5)
            ADD_FAILURE() << "encode mismatch for: " << text.substr(0, 200);
        if (tok_->decode(ids) != text && ++mismatches <= 5)
            ADD_FAILURE() << "decode mismatch for: " << text.substr(0, 200);
    }
    EXPECT_EQ(mismatches, 0) << "out of " << rows << " rows";
}

TEST_F(GoldenTokenizer, RenderConversationMatches) {
    REQUIRE_GOLDEN("conversations.jsonl");
    for (const auto &row: read_jsonl(golden("conversations.jsonl"))) {
        auto [ids, mask] = tok_->render_conversation(row.at("conversation"));
        EXPECT_EQ(ids, row.at("ids").get<std::vector<token_t>>());
        EXPECT_EQ(mask, row.at("mask").get<std::vector<uint8_t>>());
        EXPECT_EQ(tok_->render_for_completion(row.at("conversation")),
                  row.at("completion_ids").get<std::vector<token_t>>());
    }
}

class GoldenPretrained : public testing::TestWithParam<std::string> {};

TEST_P(GoldenPretrained, EncodeMatchesTiktoken) {
    const std::string name = GetParam();
    REQUIRE_GOLDEN(name + ".json");
    REQUIRE_GOLDEN("encode_" + name + ".jsonl");
    const auto tok = Tokenizer::load(golden(name + ".json"), "<|endoftext|>");
    int mismatches = 0, rows = 0;
    for (const auto &row: read_jsonl(golden("encode_" + name + ".jsonl"))) {
        ++rows;
        const auto text = row.at("text").get<std::string>();
        const auto ids = tok.encode(text);
        if (ids != row.at("ids").get<std::vector<token_t>>() && ++mismatches <= 5)
            ADD_FAILURE() << "encode mismatch for: " << text.substr(0, 200);
        if (tok.decode(ids) != text && ++mismatches <= 5)
            ADD_FAILURE() << "decode mismatch for: " << text.substr(0, 200);
    }
    EXPECT_EQ(mismatches, 0) << "out of " << rows << " rows";
}

INSTANTIATE_TEST_SUITE_P(Tiktoken, GoldenPretrained, testing::Values("gpt2", "cl100k_base"));

TEST_F(GoldenTokenizer, TokenBytesMatch) {
    REQUIRE_GOLDEN("token_bytes.bin");
    std::ifstream in(golden("token_bytes.bin"), std::ios::binary);
    std::vector<int32_t> expected(tok_->vocab_size());
    in.read(reinterpret_cast<char *>(expected.data()), static_cast<std::streamsize>(expected.size() * 4));
    ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(expected.size() * 4));
    EXPECT_EQ(tok_->token_bytes(), expected);
}
