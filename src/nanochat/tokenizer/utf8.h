// UTF-8 helpers. Code point counts match Python len(str) on valid UTF-8.
#pragma once

#include <string>
#include <string_view>

namespace nanochat {

    size_t utf8_length(std::string_view s);

    // The first n code points of s (all of s if shorter).
    std::string_view utf8_prefix(std::string_view s, size_t n);

    // Decodes bytes like Python bytes.decode("utf-8", errors="replace"): each maximal invalid subpart becomes U+FFFD.
    std::string utf8_replace_invalid(std::string_view bytes);

} // namespace nanochat
