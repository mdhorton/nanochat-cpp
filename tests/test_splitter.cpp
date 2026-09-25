#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "nanochat/tokenizer/splitter.h"

using nanochat::Pcre2Splitter;

static std::vector<std::string> split(std::string_view text) {
    static const Pcre2Splitter splitter;
    std::vector<std::string_view> chunks;
    splitter.split(text, chunks);
    return {chunks.begin(), chunks.end()};
}

using V = std::vector<std::string>;

TEST(Splitter, Words) {
    EXPECT_EQ(split("Hello world! This is a test."), (V{"Hello", " world", "!", " This", " is", " a", " test", "."}));
}

TEST(Splitter, Contractions) {
    EXPECT_EQ(split("I'm you're it's"), (V{"I", "'m", " you", "'re", " it", "'s"}));
}

TEST(Splitter, NumbersInPairs) {
    EXPECT_EQ(split("Numbers: 123, 4567"), (V{"Numbers", ":", " ", "12", "3", ",", " ", "45", "67"}));
}

TEST(Splitter, Whitespace) {
    EXPECT_EQ(split("  hello"), (V{" ", " hello"}));
    EXPECT_EQ(split("a\n\nb"), (V{"a", "\n\n", "b"}));
    EXPECT_EQ(split("x   "), (V{"x", "   "}));
}

TEST(Splitter, Unicode) {
    EXPECT_EQ(split("你好世界 🌍"), (V{"你好世界", " 🌍"}));
    EXPECT_EQ(split("naïve café"), (V{"naïve", " café"}));
}

TEST(Splitter, ChunksCoverInput) {
    for (std::string_view text: {"def f(x):\n    return x + 1\n", "a\r\n\r\n  b\t\tc!!! ?? 12345 ÄÖÜ 🙂🙂", ""}) {
        std::string joined;
        for (const auto &c: split(text))
            joined += c;
        EXPECT_EQ(joined, text);
    }
}

TEST(Splitter, InvalidUtf8Throws) {
    EXPECT_THROW(split("ok \xff bad"), std::runtime_error);
}
