#include "nanochat/tokenizer/utf8.h"

namespace nanochat {

static bool is_continuation(unsigned char c) {
  return (c & 0xC0) == 0x80;
}

size_t utf8_length(std::string_view s) {
  size_t n = 0;
  for (unsigned char c : s)
    n += !is_continuation(c);
  return n;
}

std::string_view utf8_prefix(std::string_view s, size_t n) {
  size_t i = 0;
  for (; i < s.size(); ++i)
    if (!is_continuation(s[i]) && n-- == 0)
      break;
  return s.substr(0, i);
}

std::string utf8_replace_invalid(std::string_view bytes) {
  static constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
  std::string out;
  out.reserve(bytes.size());
  const size_t n = bytes.size();
  size_t i = 0;
  while (i < n) {
    const auto b = static_cast<unsigned char>(bytes[i]);
    if (b < 0x80) {
      out.push_back(static_cast<char>(b));
      ++i;
      continue;
    }
    // sequence length and valid range of the second byte (Unicode Table 3-7)
    size_t len;
    unsigned char lo = 0x80, hi = 0xBF;
    if (b >= 0xC2 && b <= 0xDF) {
      len = 2;
    }
    else if (b >= 0xE0 && b <= 0xEF) {
      len = 3;
      if (b == 0xE0)
        lo = 0xA0;
      else if (b == 0xED)
        hi = 0x9F;
    }
    else if (b >= 0xF0 && b <= 0xF4) {
      len = 4;
      if (b == 0xF0)
        lo = 0x90;
      else if (b == 0xF4)
        hi = 0x8F;
    }
    else {
      out += kReplacement;
      ++i;
      continue;
    }
    size_t j = 1;
    for (; j < len && i + j < n; ++j) {
      const auto c = static_cast<unsigned char>(bytes[i + j]);
      if (j == 1 ? (c < lo || c > hi) : !is_continuation(c))
        break;
    }
    if (j == len)
      out.append(bytes.substr(i, len));
    else
      out += kReplacement; // maximal subpart bytes[i, i+j) is replaced as one
    i += j;
  }
  return out;
}

} // namespace nanochat
