// Distributed pretraining data loader with BOS-aligned best-fit packing. Port of nanochat/dataloader.py.
//  - every row starts with BOS
//  - documents are packed largest-fit-first; when nothing fits, the shortest is cropped to fill the row exactly
//  - 100% utilization (no padding)
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

#include "nanochat/dataset.h"
#include "nanochat/tokenizer/tokenizer.h"

namespace nanochat {

    // Position of the last document batch read, for (approximate) resume.
    struct DataLoaderState {
        int64_t pq_idx = 0, rg_idx = 0, epoch = 1;
        bool operator==(const DataLoaderState &) const = default;
    };

    struct DataLoaderOptions {
        int rank = 0, world_size = 1; // row groups are strided across ranks
        std::optional<DataLoaderState> resume;
        int tokenizer_threads = 4;
        size_t tokenizer_batch_size = 128;
        size_t buffer_size = 1000; // documents kept for best-fit search
        torch::Device device = torch::kCUDA;
    };

    // Infinite iterator over document batches of the split, one tokenizer batch at a time, epoch after epoch.
    class DocumentBatches {
    public:
        DocumentBatches(const std::filesystem::path &data_dir, Split split, const DataLoaderOptions &options);
        ~DocumentBatches();

        // Fills docs and where (the batch's file, row group and epoch).
        void next(std::vector<std::string> &docs, DataLoaderState &where);

    private:
        std::vector<std::filesystem::path> files_;
        int rank_, world_size_;
        size_t batch_size_;
        std::optional<DataLoaderState> resume_; // cleared once the resume row group is reached
        bool first_pass_ = true;
        int64_t pq_idx_ = 0, rg_idx_ = 0, epoch_ = 1;
        std::unique_ptr<ParquetTextFile> file_;
        bool rg_loaded_ = false;
        std::vector<std::string> texts_; // current row group
        size_t text_pos_ = 0;
    };

    class DataLoader {
    public:
        DataLoader(const Tokenizer &tokenizer, int64_t B, int64_t T, Split split, const std::filesystem::path &data_dir,
                   DataLoaderOptions options = {});

        // (inputs, targets), each (B, T) int64 on the device. Views into one persistent buffer, overwritten by the
        // next call; the copy is stream-ordered, so using them before the next call is safe.
        std::pair<torch::Tensor, torch::Tensor> next();

        // Position of the last document batch that went into the most recent next().
        const DataLoaderState &state() const { return state_; }

    private:
        void refill();

        const Tokenizer &tokenizer_;
        int64_t B_, T_;
        DataLoaderOptions options_;
        DocumentBatches batches_;
        std::vector<std::vector<token_t>> doc_buffer_;
        DataLoaderState state_;
        torch::Tensor row_buffer_, cpu_buffer_, gpu_buffer_; // rows (B, T+1); [inputs | targets] staging and device
        at::cuda::CUDAEvent copied_; // the last host-to-device copy, before cpu_buffer_ is reused
        std::vector<std::string> docs_;
    };

} // namespace nanochat
