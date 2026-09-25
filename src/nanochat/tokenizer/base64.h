#pragma once

#include <string>
#include <string_view>

namespace nanochat {

    std::string base64_encode(std::string_view in);
    std::string base64_decode(std::string_view in);

} // namespace nanochat
