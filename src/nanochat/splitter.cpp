#include "nanochat/splitter.h"

#include <stdexcept>
#include <string>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace nanochat {

    static std::string pcre2_error(int code) {
        PCRE2_UCHAR buf[256];
        pcre2_get_error_message(code, buf, sizeof(buf));
        return reinterpret_cast<char *>(buf);
    }

    // Per-thread match state; pcre2_code itself is shareable across threads.
    struct MatchState {
        pcre2_match_data *md = pcre2_match_data_create(1, nullptr);
        pcre2_jit_stack *jit_stack = pcre2_jit_stack_create(32 * 1024, 4 * 1024 * 1024, nullptr);
        pcre2_match_context *mctx = pcre2_match_context_create(nullptr);

        MatchState() { pcre2_jit_stack_assign(mctx, nullptr, jit_stack); }
        ~MatchState() {
            pcre2_match_context_free(mctx);
            pcre2_jit_stack_free(jit_stack);
            pcre2_match_data_free(md);
        }
    };

    struct Pcre2Splitter::Impl {
        pcre2_code *code = nullptr;
    };

    Pcre2Splitter::Pcre2Splitter(std::string_view pattern) : impl_(std::make_unique<Impl>()) {
        int err;
        PCRE2_SIZE err_off;
        // DOLLAR_ENDONLY: $ matches only at the very end, as in Rust's regex (tiktoken's GPT-2/cl100k patterns use it)
        impl_->code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
                                    PCRE2_UTF | PCRE2_UCP | PCRE2_DOLLAR_ENDONLY, &err, &err_off, nullptr);
        if (!impl_->code)
            throw std::invalid_argument("pcre2_compile at offset " + std::to_string(err_off) + ": " + pcre2_error(err));
        if (int rc = pcre2_jit_compile(impl_->code, PCRE2_JIT_COMPLETE); rc != 0)
            throw std::runtime_error("pcre2_jit_compile: " + pcre2_error(rc));
    }

    Pcre2Splitter::~Pcre2Splitter() { pcre2_code_free(impl_->code); }

    void Pcre2Splitter::split(std::string_view text, std::vector<std::string_view> &out) const {
        thread_local MatchState st;
        const auto subject = reinterpret_cast<PCRE2_SPTR>(text.data());
        const size_t len = text.size();
        size_t off = 0;
        uint32_t opts = 0; // first match validates UTF-8 for the whole subject
        while (off < len) {
            int rc = pcre2_match(impl_->code, subject, len, off, opts, st.md, st.mctx);
            opts = PCRE2_NO_UTF_CHECK;
            if (rc == PCRE2_ERROR_NOMATCH)
                break;
            if (rc < 0)
                throw std::runtime_error("pcre2_match: " + pcre2_error(rc));
            const PCRE2_SIZE *ov = pcre2_get_ovector_pointer(st.md);
            if (ov[1] == ov[0]) { // empty match: step over one UTF-8 char
                off = ov[0] + 1;
                while (off < len && (static_cast<unsigned char>(text[off]) & 0xC0) == 0x80)
                    ++off;
                continue;
            }
            out.push_back(text.substr(ov[0], ov[1] - ov[0]));
            off = ov[1];
        }
    }

} // namespace nanochat
