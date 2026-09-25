// BOS best-fit data loader: batches and states must be identical to Python's (tools/export_train_golden.py).
#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "nanochat/train/dataloader.h"
#include "nanochat/train/safetensors.h"
#include "test_env.h"

using namespace nanochat;
namespace fs = std::filesystem;

namespace {

    fs::path golden(const std::string &name) { return test_env().golden_dir / "train" / name; }

    DataLoaderState state_from_json(const nlohmann::json &j) {
        return {.pq_idx = j["pq_idx"], .rg_idx = j["rg_idx"], .epoch = j["epoch"]};
    }

    class LoaderGolden : public testing::TestWithParam<std::string> {
    protected:
        static void SetUpTestSuite() {
            const auto tok_path = test_env().golden_dir / "tokenizer.json";
            if (fs::exists(golden("loader_cases.json")) && fs::exists(tok_path)) {
                tokenizer_ = std::make_unique<Tokenizer>(Tokenizer::load(tok_path));
                std::ifstream in(golden("loader_cases.json"));
                cases_ = nlohmann::json::parse(in);
            }
        }

        void SetUp() override {
            if (!tokenizer_)
                GTEST_SKIP() << "missing " << golden("loader_cases.json") << " (run tools/export_train_golden.py)";
        }

        static inline std::unique_ptr<Tokenizer> tokenizer_;
        static inline nlohmann::json cases_;
    };

} // namespace

TEST_P(LoaderGolden, MatchesPython) {
    const auto &c = cases_.at(GetParam());
    DataLoaderOptions options;
    options.rank = c.value("rank", 0);
    options.world_size = c.value("world_size", 1);
    if (c.contains("resume") && !c["resume"].is_null())
        options.resume = state_from_json(c["resume"]);
    const auto split = c["split"] == "train" ? Split::Train : Split::Val;
    DataLoader loader(*tokenizer_, c["B"], c["T"], split, test_env().base_dir / "base_data_climbmix", options);

    const auto want = safetensors::load(golden("loader_" + GetParam() + ".safetensors"));
    for (int64_t i = 0; i < c["num_batches"].get<int64_t>(); ++i) {
        auto [x, y] = loader.next();
        EXPECT_TRUE(torch::equal(x.cpu(), want.at("inputs")[i])) << "inputs of batch " << i;
        EXPECT_TRUE(torch::equal(y.cpu(), want.at("targets")[i])) << "targets of batch " << i;
        EXPECT_EQ(loader.state(), state_from_json(c["states"][i])) << "state after batch " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(Cases, LoaderGolden,
                         testing::Values("train", "resume", "resume_next_file", "rank1", "val"));

TEST(DataLoader, RowsStartWithBos) {
    const auto tok_path = test_env().golden_dir / "tokenizer.json";
    if (!fs::exists(tok_path))
        GTEST_SKIP() << "missing " << tok_path;
    const auto tokenizer = Tokenizer::load(tok_path);
    DataLoader loader(tokenizer, 4, 128, Split::Val, test_env().base_dir / "base_data_climbmix");
    for (int i = 0; i < 3; ++i) {
        auto [x, y] = loader.next();
        EXPECT_TRUE((x.select(1, 0) == tokenizer.bos_token_id()).all().item<bool>());
        EXPECT_TRUE(torch::equal(x.slice(1, 1), y.slice(1, 0, -1))); // targets are inputs shifted by one
    }
}
