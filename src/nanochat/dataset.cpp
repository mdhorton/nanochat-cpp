#include "nanochat/dataset.h"

#include <algorithm>
#include <stdexcept>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include "nanochat/tokenizer/utf8.h"

namespace nanochat {

    namespace fs = std::filesystem;

    std::vector<fs::path> list_parquet_files(const fs::path &data_dir) {
        std::vector<fs::path> paths;
        for (const auto &entry: fs::directory_iterator(data_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".parquet")
                paths.push_back(entry.path());
        }
        std::ranges::sort(paths);
        return paths;
    }

    ParquetBatches::ParquetBatches(const fs::path &data_dir, Split split, int start, int step) :
        start_(start), step_(step) {
        if (start < 0 || step < 1)
            throw std::invalid_argument("ParquetBatches: need start >= 0 and step >= 1");
        auto paths = list_parquet_files(data_dir);
        if (split == Split::Train) {
            if (!paths.empty())
                paths.pop_back();
            files_ = std::move(paths);
        } else if (!paths.empty()) {
            files_ = {paths.back()};
        }
    }

    ParquetBatches::~ParquetBatches() = default;

    bool ParquetBatches::open_next_file() {
        if (file_idx_ >= files_.size())
            return false;
        file_ = std::make_unique<ParquetTextFile>(files_[file_idx_++]);
        rg_idx_ = start_;
        return true;
    }

    ParquetTextFile::ParquetTextFile(const fs::path &path) {
        auto infile = arrow::io::ReadableFile::Open(path.string()).ValueOrDie();
        reader_ = parquet::arrow::OpenFile(infile, arrow::default_memory_pool()).ValueOrDie();
        std::shared_ptr<arrow::Schema> schema;
        PARQUET_THROW_NOT_OK(reader_->GetSchema(&schema));
        text_col_ = schema->GetFieldIndex("text");
        if (text_col_ < 0)
            throw std::runtime_error("no 'text' column in " + path.string());
        num_row_groups_ = reader_->num_row_groups();
    }

    ParquetTextFile::~ParquetTextFile() = default;

    template<typename ArrayT>
    static void append_texts(const arrow::Array &chunk, std::vector<std::string> &texts) {
        const auto &arr = static_cast<const ArrayT &>(chunk);
        for (int64_t i = 0; i < arr.length(); ++i) {
            if (arr.IsNull(i))
                throw std::runtime_error("null text in parquet row");
            texts.emplace_back(arr.GetView(i));
        }
    }

    bool ParquetBatches::next(std::vector<std::string> &texts) {
        while (!file_ || rg_idx_ >= file_->num_row_groups()) {
            if (!open_next_file())
                return false;
        }
        file_->read_row_group(rg_idx_, texts);
        rg_idx_ += step_;
        return true;
    }

    void ParquetTextFile::read_row_group(int rg, std::vector<std::string> &texts) {
        auto table = reader_->ReadRowGroup(rg, {text_col_}).ValueOrDie();
        texts.clear();
        for (const auto &chunk: table->column(0)->chunks()) {
            switch (chunk->type_id()) {
                case arrow::Type::STRING:
                    append_texts<arrow::StringArray>(*chunk, texts);
                    break;
                case arrow::Type::LARGE_STRING:
                    append_texts<arrow::LargeStringArray>(*chunk, texts);
                    break;
                case arrow::Type::STRING_VIEW:
                    append_texts<arrow::StringViewArray>(*chunk, texts);
                    break;
                default:
                    throw std::runtime_error("unsupported 'text' column type: " + chunk->type()->ToString());
            }
        }
    }

    CappedTexts::CappedTexts(ParquetBatches &batches, int64_t max_chars, int64_t doc_cap, size_t batch_size) :
        batches_(batches), max_chars_(max_chars), doc_cap_(doc_cap), batch_size_(batch_size) {}

    bool CappedTexts::next(std::vector<std::string> &docs) {
        docs.clear();
        while (docs.size() < batch_size_ && !done_) {
            if (pos_ >= pending_.size()) {
                if (!batches_.next(pending_)) {
                    done_ = true;
                    break;
                }
                pos_ = 0;
                continue;
            }
            std::string &doc = pending_[pos_++];
            if (auto cropped = utf8_prefix(doc, doc_cap_); cropped.size() < doc.size())
                doc.resize(cropped.size());
            nchars_ += static_cast<int64_t>(utf8_length(doc));
            docs.push_back(std::move(doc));
            if (nchars_ > max_chars_)
                done_ = true;
        }
        return !docs.empty();
    }

} // namespace nanochat
