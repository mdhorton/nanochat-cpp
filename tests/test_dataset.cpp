#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <unistd.h>

#include "nanochat/data/dataset.h"

namespace fs = std::filesystem;
using nanochat::ParquetBatches;
using nanochat::Split;

// Writes rows as a "text" column, rows_per_group rows per row group.
static void write_parquet(const fs::path& path, const std::vector<std::string>& rows, int64_t rows_per_group) {
  arrow::StringBuilder builder;
  PARQUET_THROW_NOT_OK(builder.AppendValues(rows));
  auto array = builder.Finish().ValueOrDie();
  auto table = arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8())}), {array});
  auto out = arrow::io::FileOutputStream::Open(path.string()).ValueOrDie();
  PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, rows_per_group));
}

class DatasetTest : public testing::Test {
protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() / ("nanochat_test_" + std::to_string(::getpid()));
    fs::create_directories(dir_);
    write_parquet(dir_ / "shard_00000.parquet", {"a0", "a1", "a2", "a3"}, 2);
    write_parquet(dir_ / "shard_00001.parquet", {"b0", "b1", "b2", "b3"}, 2);
    write_parquet(dir_ / "shard_00002.parquet", {"v0", "v1"}, 2);
    std::ofstream(dir_ / "shard_00003.parquet.tmp") << "partial download";
  }

  void TearDown() override {
    fs::remove_all(dir_);
  }

  std::vector<std::vector<std::string>> collect(Split split, int start = 0, int step = 1) {
    ParquetBatches batches(dir_, split, start, step);
    std::vector<std::vector<std::string>> out;
    std::vector<std::string> texts;
    while (batches.next(texts))
      out.push_back(texts);
    return out;
  }

  fs::path dir_;
};

using Batches = std::vector<std::vector<std::string>>;

TEST_F(DatasetTest, ListIgnoresTmp) {
  auto files = nanochat::list_parquet_files(dir_);
  ASSERT_EQ(files.size(), 3);
  EXPECT_EQ(files.back().filename(), "shard_00002.parquet");
}

TEST_F(DatasetTest, TrainIsAllButLast) {
  EXPECT_EQ(collect(Split::Train), (Batches{{"a0", "a1"}, {"a2", "a3"}, {"b0", "b1"}, {"b2", "b3"}}));
}

TEST_F(DatasetTest, ValIsLast) {
  EXPECT_EQ(collect(Split::Val), (Batches{{"v0", "v1"}}));
}

TEST_F(DatasetTest, StartStepStridesRowGroups) {
  EXPECT_EQ(collect(Split::Train, 1, 2), (Batches{{"a2", "a3"}, {"b2", "b3"}}));
  EXPECT_EQ(collect(Split::Val, 1, 2), Batches{});
}
