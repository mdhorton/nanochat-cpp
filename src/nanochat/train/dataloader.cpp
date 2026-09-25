#include "nanochat/train/dataloader.h"

#include <algorithm>
#include <stdexcept>

namespace nanochat {

    namespace fs = std::filesystem;

    DocumentBatches::DocumentBatches(const fs::path &data_dir, Split split, const DataLoaderOptions &options) :
        rank_(options.rank), world_size_(options.world_size), batch_size_(options.tokenizer_batch_size),
        resume_(options.resume) {
        auto paths = list_parquet_files(data_dir);
        if (paths.empty())
            throw std::runtime_error("no parquet files in " + data_dir.string());
        if (split == Split::Train)
            paths.pop_back();
        else
            paths.erase(paths.begin(), paths.end() - 1);
        files_ = std::move(paths);
        if (resume_) {
            pq_idx_ = resume_->pq_idx;
            epoch_ = resume_->epoch;
        }
    }

    DocumentBatches::~DocumentBatches() = default;

    // Mirrors the Python generator _document_batches as a state machine.
    void DocumentBatches::next(std::vector<std::string> &docs, DataLoaderState &where) {
        while (true) {
            if (!file_) {
                if (pq_idx_ >= static_cast<int64_t>(files_.size())) { // end of an epoch
                    first_pass_ = false;
                    ++epoch_;
                    pq_idx_ = 0;
                    continue;
                }
                auto file = std::make_unique<ParquetTextFile>(files_[pq_idx_]);
                if (first_pass_ && resume_ && pq_idx_ == resume_->pq_idx) {
                    // resume on the same file: advance one stride so data isn't repeated
                    const int64_t base = resume_->rg_idx / world_size_ + 1;
                    const int64_t rg = base * world_size_ + rank_;
                    if (rg >= file->num_row_groups()) {
                        ++pq_idx_;
                        continue;
                    }
                    resume_.reset();
                    rg_idx_ = rg;
                }
                else {
                    rg_idx_ = rank_;
                }
                file_ = std::move(file);
                rg_loaded_ = false;
            }
            if (!rg_loaded_) {
                if (rg_idx_ >= file_->num_row_groups()) {
                    file_.reset();
                    ++pq_idx_;
                    continue;
                }
                file_->read_row_group(static_cast<int>(rg_idx_), texts_);
                text_pos_ = 0;
                rg_loaded_ = true;
            }
            if (text_pos_ < texts_.size()) {
                const size_t end = std::min(texts_.size(), text_pos_ + batch_size_);
                docs.assign(std::make_move_iterator(texts_.begin() + static_cast<std::ptrdiff_t>(text_pos_)),
                            std::make_move_iterator(texts_.begin() + static_cast<std::ptrdiff_t>(end)));
                text_pos_ = end;
                where = {pq_idx_, rg_idx_, epoch_};
                return;
            }
            rg_idx_ += world_size_;
            rg_loaded_ = false;
        }
    }

    DataLoader::DataLoader(const Tokenizer &tokenizer, int64_t B, int64_t T, Split split, const fs::path &data_dir,
                           DataLoaderOptions options) :
        tokenizer_(tokenizer), B_(B), T_(T), options_(std::move(options)), batches_(data_dir, split, options_) {
        const auto cpu = torch::TensorOptions().dtype(torch::kInt64);
        row_buffer_ = torch::empty({B, T + 1}, cpu);
        cpu_buffer_ = torch::empty({2 * B * T}, cpu.pinned_memory(options_.device.is_cuda()));
        gpu_buffer_ = torch::empty({2 * B * T}, cpu.device(options_.device));
    }

    void DataLoader::refill() {
        batches_.next(docs_, state_);
        auto token_lists = tokenizer_.encode_batch(docs_, tokenizer_.bos_token_id(), std::nullopt,
                                                   options_.tokenizer_threads);
        for (auto &tokens: token_lists)
            doc_buffer_.push_back(std::move(tokens));
    }

    std::pair<torch::Tensor, torch::Tensor> DataLoader::next() {
        const int64_t row_capacity = T_ + 1;
        auto rows = row_buffer_.accessor<int64_t, 2>();
        for (int64_t row = 0; row < B_; ++row) {
            int64_t pos = 0;
            while (pos < row_capacity) {
                while (doc_buffer_.size() < options_.buffer_size)
                    refill();
                const int64_t remaining = row_capacity - pos;

                // largest doc that fits entirely (first one on ties)
                int64_t best_idx = -1, best_len = 0;
                for (size_t i = 0; i < doc_buffer_.size(); ++i) {
                    const auto len = static_cast<int64_t>(doc_buffer_[i].size());
                    if (len <= remaining && len > best_len) {
                        best_idx = static_cast<int64_t>(i);
                        best_len = len;
                    }
                }
                size_t take_idx;
                int64_t take_len;
                if (best_idx >= 0) {
                    take_idx = static_cast<size_t>(best_idx);
                    take_len = best_len;
                }
                else { // nothing fits: crop the shortest (first one on ties) to fill the row exactly
                    take_idx = static_cast<size_t>(std::ranges::min_element(doc_buffer_, {}, &std::vector<token_t>::size) -
                                                   doc_buffer_.begin());
                    take_len = remaining;
                }
                const auto &doc = doc_buffer_[take_idx];
                for (int64_t j = 0; j < take_len; ++j)
                    rows[row][pos + j] = doc[j];
                pos += take_len;
                doc_buffer_.erase(doc_buffer_.begin() + static_cast<std::ptrdiff_t>(take_idx));
            }
        }

        // stage [inputs | targets] in pinned memory, then a single host-to-device copy
        if (options_.device.is_cuda())
            copied_.synchronize(); // the previous async copy may still be reading cpu_buffer_
        const int64_t n = B_ * T_;
        cpu_buffer_.slice(0, 0, n).view({B_, T_}).copy_(row_buffer_.slice(1, 0, T_));
        cpu_buffer_.slice(0, n, 2 * n).view({B_, T_}).copy_(row_buffer_.slice(1, 1, T_ + 1));
        gpu_buffer_.copy_(cpu_buffer_, /*non_blocking=*/options_.device.is_cuda());
        if (options_.device.is_cuda())
            copied_.record();
        return {gpu_buffer_.slice(0, 0, n).view({B_, T_}), gpu_buffer_.slice(0, n, 2 * n).view({B_, T_})};
    }

} // namespace nanochat
