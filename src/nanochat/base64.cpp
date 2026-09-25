#include "nanochat/base64.h"

#include <cstdint>
#include <stdexcept>

namespace nanochat {

    static constexpr std::string_view kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string base64_encode(std::string_view in) {
        std::string out;
        out.reserve((in.size() + 2) / 3 * 4);
        for (size_t i = 0; i < in.size(); i += 3) {
            uint32_t v = static_cast<uint8_t>(in[i]) << 16;
            if (i + 1 < in.size())
                v |= static_cast<uint8_t>(in[i + 1]) << 8;
            if (i + 2 < in.size())
                v |= static_cast<uint8_t>(in[i + 2]);
            out += kB64[v >> 18 & 63];
            out += kB64[v >> 12 & 63];
            out += i + 1 < in.size() ? kB64[v >> 6 & 63] : '=';
            out += i + 2 < in.size() ? kB64[v & 63] : '=';
        }
        return out;
    }

    std::string base64_decode(std::string_view in) {
        std::string out;
        uint32_t v = 0;
        int bits = 0;
        for (char c: in) {
            if (c == '=')
                break;
            auto pos = kB64.find(c);
            if (pos == std::string_view::npos)
                throw std::invalid_argument("invalid base64");
            v = v << 6 | static_cast<uint32_t>(pos);
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                out += static_cast<char>(v >> bits & 0xFF);
            }
        }
        return out;
    }

} // namespace nanochat
