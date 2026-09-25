// BPE training. Port of rustbpe (github.com/karpathy/rustbpe).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanochat/tokenizer/splitter.h"

namespace nanochat {

    using ChunkCounts = std::unordered_map<std::string, int64_t>;

    // Counts split chunks over batches of documents, splitting in parallel.
    class ChunkCounter {
    public:
        ChunkCounter(const Splitter &splitter, int num_threads);

        void add(std::span<const std::string> docs);
        const ChunkCounts &counts() const { return counts_; }
        ChunkCounts take() { return std::move(counts_); }

    private:
        const Splitter &splitter_;
        int num_threads_;
        ChunkCounts counts_;
    };

    // Learns merges from chunk counts. Returns token bytes indexed by rank: 256 single bytes, then one per merge.
    // Merge order matches rustbpe: highest pair count first, ties broken by the smallest (left, right) id pair.
    // Stops early if no pairs remain.
    std::vector<std::string> train_bpe(const ChunkCounts &chunk_counts, uint32_t vocab_size);

} // namespace nanochat
