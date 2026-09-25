// Pre-tokenization: splits text into chunks that BPE merges never cross.
#pragma once

#include <memory>
#include <string_view>
#include <vector>

namespace nanochat {

    // GPT-4 split pattern with \p{N}{1,2} instead of \p{N}{1,3}. Copied verbatim from nanochat/tokenizer.py.
    inline constexpr std::string_view SPLIT_PATTERN =
            R"PAT('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)PAT";

    class Splitter {
    public:
        virtual ~Splitter() = default;

        // Appends the chunks of text (UTF-8) to out; chunks view into text. Throws on invalid UTF-8.
        virtual void split(std::string_view text, std::vector<std::string_view> &out) const = 0;
    };

    // PCRE2 (JIT) implementation. Thread-safe: split() may be called concurrently.
    class Pcre2Splitter final : public Splitter {
    public:
        explicit Pcre2Splitter(std::string_view pattern = SPLIT_PATTERN);
        ~Pcre2Splitter() override;
        Pcre2Splitter(const Pcre2Splitter &) = delete;
        Pcre2Splitter &operator=(const Pcre2Splitter &) = delete;

        void split(std::string_view text, std::vector<std::string_view> &out) const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace nanochat
