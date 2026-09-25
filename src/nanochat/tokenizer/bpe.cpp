#include "nanochat/tokenizer/bpe.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace nanochat {

    ChunkCounter::ChunkCounter(const Splitter &splitter, int num_threads) :
        splitter_(splitter), num_threads_(std::max(1, num_threads)) {}

    void ChunkCounter::add(std::span<const std::string> docs) {
        const int nt = std::min<int>(num_threads_, static_cast<int>(docs.size()));
        if (nt == 0)
            return;
        std::vector<std::unordered_map<std::string_view, int64_t>> locals(nt);
        {
            std::vector<std::jthread> threads;
            for (int t = 0; t < nt; ++t) {
                threads.emplace_back([&, t] {
                    std::vector<std::string_view> chunks;
                    for (size_t i = t; i < docs.size(); i += nt) {
                        chunks.clear();
                        splitter_.split(docs[i], chunks);
                        for (auto c: chunks)
                            ++locals[t][c];
                    }
                });
            }
        }
        for (const auto &local: locals) {
            for (const auto &[chunk, n]: local)
                counts_[std::string(chunk)] += n;
        }
    }

    namespace {

        using Pair = uint64_t; // (left << 32) | right, so integer order == lexicographic (left, right) order

        Pair make_pair(uint32_t a, uint32_t b) { return (static_cast<uint64_t>(a) << 32) | b; }
        uint32_t left(Pair p) { return static_cast<uint32_t>(p >> 32); }
        uint32_t right(Pair p) { return static_cast<uint32_t>(p); }

        struct Delta {
            Pair pair;
            int delta;
        };

        // Merges all non-overlapping occurrences of pair in ids (left to right); appends pair-count deltas.
        void merge_word(std::vector<uint32_t> &ids, Pair pair, uint32_t new_id, std::vector<Delta> &deltas) {
            const uint32_t a = left(pair), b = right(pair);
            const size_t n = ids.size();
            if (n < 2)
                return;
            size_t w = 0; // write index; ids[0, w) is the output so far
            size_t i = 0;
            while (i < n) {
                if (i + 1 < n && ids[i] == a && ids[i + 1] == b) {
                    if (w > 0) {
                        deltas.push_back({make_pair(ids[w - 1], a), -1});
                        deltas.push_back({make_pair(ids[w - 1], new_id), 1});
                    }
                    deltas.push_back({pair, -1});
                    if (i + 2 < n) {
                        deltas.push_back({make_pair(b, ids[i + 2]), -1});
                        deltas.push_back({make_pair(new_id, ids[i + 2]), 1});
                    }
                    ids[w++] = new_id;
                    i += 2;
                } else {
                    ids[w++] = ids[i++];
                }
            }
            ids.resize(w);
        }

        struct MergeJob {
            int64_t count;
            Pair pair;
            std::vector<uint32_t> words; // indices of words that may contain pair

            // heap order: max count, then smallest pair
            bool operator<(const MergeJob &o) const { return count != o.count ? count < o.count : pair > o.pair; }
        };

        // Appends word index i unless it was the last one added (callers visit each word contiguously).
        void add_word(std::vector<uint32_t> &words, uint32_t i) {
            if (words.empty() || words.back() != i)
                words.push_back(i);
        }

    } // namespace

    std::vector<std::string> train_bpe(const ChunkCounts &chunk_counts, uint32_t vocab_size) {
        if (vocab_size < 256)
            throw std::invalid_argument("vocab_size must be at least 256");

        std::vector<std::vector<uint32_t>> words;
        std::vector<int64_t> counts;
        words.reserve(chunk_counts.size());
        counts.reserve(chunk_counts.size());
        for (const auto &[chunk, n]: chunk_counts) {
            words.emplace_back(chunk.begin(), chunk.end());
            for (auto &id: words.back())
                id &= 0xFF; // char may be signed
            counts.push_back(n);
        }

        std::unordered_map<Pair, int64_t> pair_counts;
        std::unordered_map<Pair, std::vector<uint32_t>> where;
        for (uint32_t i = 0; i < words.size(); ++i) {
            const auto &w = words[i];
            if (w.size() < 2 || counts[i] == 0)
                continue;
            for (size_t j = 0; j + 1 < w.size(); ++j) {
                Pair p = make_pair(w[j], w[j + 1]);
                pair_counts[p] += counts[i];
                add_word(where[p], i);
            }
        }

        std::vector<MergeJob> heap;
        heap.reserve(where.size());
        for (auto &[pair, ws]: where) {
            if (int64_t c = pair_counts[pair]; c > 0)
                heap.push_back({c, pair, std::move(ws)});
        }
        where.clear();
        std::make_heap(heap.begin(), heap.end());

        std::vector<std::string> vocab(256);
        for (int b = 0; b < 256; ++b)
            vocab[b] = std::string(1, static_cast<char>(b));
        vocab.reserve(vocab_size);

        std::vector<Delta> deltas;
        std::unordered_map<Pair, std::vector<uint32_t>> updates;
        while (vocab.size() < vocab_size && !heap.empty()) {
            std::pop_heap(heap.begin(), heap.end());
            MergeJob top = std::move(heap.back());
            heap.pop_back();

            // lazy refresh: requeue if the count changed since this job was queued
            const auto it = pair_counts.find(top.pair);
            const int64_t current = it == pair_counts.end() ? 0 : it->second;
            if (current <= 0)
                continue;
            if (top.count != current) {
                top.count = current;
                heap.push_back(std::move(top));
                std::push_heap(heap.begin(), heap.end());
                continue;
            }

            const auto new_id = static_cast<uint32_t>(vocab.size());
            vocab.push_back(vocab[left(top.pair)] + vocab[right(top.pair)]);

            updates.clear();
            for (uint32_t wi: top.words) {
                deltas.clear();
                merge_word(words[wi], top.pair, new_id, deltas);
                for (const auto &[pair, delta]: deltas) {
                    const int64_t total = delta * counts[wi];
                    if (total == 0)
                        continue;
                    pair_counts[pair] += total;
                    if (delta > 0)
                        add_word(updates[pair], wi);
                }
            }
            for (auto &[pair, ws]: updates) {
                if (int64_t c = pair_counts[pair]; c > 0) {
                    heap.push_back({c, pair, std::move(ws)});
                    std::push_heap(heap.begin(), heap.end());
                }
            }
        }
        return vocab;
    }

} // namespace nanochat
