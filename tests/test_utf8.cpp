#include <gtest/gtest.h>

#include "nanochat/utf8.h"

using namespace nanochat;

TEST(Utf8, LengthAndPrefix) {
    EXPECT_EQ(utf8_length("naïve 你好 🙂"), 10);
    EXPECT_EQ(utf8_prefix("naïve 你好 🙂", 3), "naï");
    EXPECT_EQ(utf8_prefix("你好", 1), "你");
    EXPECT_EQ(utf8_prefix("abc", 10), "abc");
    EXPECT_EQ(utf8_prefix("abc", 0), "");
}

// expected values from Python: bytes.decode("utf-8", errors="replace")
TEST(Utf8, ReplaceInvalidMatchesPython) {
    const std::string R = "\xEF\xBF\xBD";
    EXPECT_EQ(utf8_replace_invalid("\xff"), R);
    EXPECT_EQ(utf8_replace_invalid("\xe2\x82"), R);
    EXPECT_EQ(utf8_replace_invalid("\xed\xa0\x80"), R + R + R);
    EXPECT_EQ(utf8_replace_invalid("\xf0\x9f\x99" "A"), R + "A");
    EXPECT_EQ(utf8_replace_invalid("\xc0\xaf"), R + R);
    EXPECT_EQ(utf8_replace_invalid("\xf4\x90\x80\x80"), R + R + R + R);
    EXPECT_EQ(utf8_replace_invalid("a\xe2\x82\xac" "b\x80"), "a€b" + R);
}
