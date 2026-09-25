// Pretraining dataset: parquet shards with a "text" column. Port of nanochat/dataset.py.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace parquet::arrow {
    class FileReader;
}

namespace nanochat {

    enum class Split { Train, Val };

    // Sorted paths of all *.parquet files in data_dir.
    std::vector<std::filesystem::path> list_parquet_files(const std::filesystem::path &data_dir);

    // The "text" column of one parquet file, read one row group at a time.
    class ParquetTextFile {
    public:
        explicit ParquetTextFile(const std::filesystem::path &path);
        ~ParquetTextFile();

        int num_row_groups() const { return num_row_groups_; }
        // Replaces texts with row group rg.
        void read_row_group(int rg, std::vector<std::string> &texts);

    private:
        std::unique_ptr<parquet::arrow::FileReader> reader_;
        int text_col_ = -1;
        int num_row_groups_ = 0;
    };

    // Yields the texts of one row group per next() call. The last file is val, the rest train.
    // start/step stride over row groups within each file (e.g. start=rank, step=world_size).
    class ParquetBatches {
    public:
        ParquetBatches(const std::filesystem::path &data_dir, Split split, int start = 0, int step = 1);
        ~ParquetBatches();

        // Fills texts with the next row group; returns false when exhausted.
        bool next(std::vector<std::string> &texts);

    private:
        bool open_next_file();

        std::vector<std::filesystem::path> files_;
        int start_, step_;
        size_t file_idx_ = 0;
        std::unique_ptr<ParquetTextFile> file_;
        int rg_idx_ = 0;
    };

    // Documents cropped to doc_cap code points, stopping after the doc that takes the total past max_chars.
    // Same semantics as text_iterator in scripts/tok_train.py.
    class CappedTexts {
    public:
        CappedTexts(ParquetBatches &batches, int64_t max_chars, int64_t doc_cap, size_t batch_size = 8192);

        // Fills docs with up to batch_size documents; returns false when exhausted.
        bool next(std::vector<std::string> &docs);
        int64_t num_chars() const { return nchars_; }

    private:
        ParquetBatches &batches_;
        int64_t max_chars_, doc_cap_;
        size_t batch_size_;
        std::vector<std::string> pending_;
        size_t pos_ = 0;
        int64_t nchars_ = 0;
        bool done_ = false;
    };

} // namespace nanochat
