#include "nanochat/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

#include "nanochat/base64.h"
#include "nanochat/bpe.h"
#include "nanochat/utf8.h"

namespace nanochat {

    using json = nlohmann::json;

    static constexpr token_t kNoRank = std::numeric_limits<token_t>::max();

    Tokenizer::Tokenizer(std::vector<std::string> mergeable_ranks,
                         const std::vector<std::pair<std::string, token_t>> &special_tokens, std::string pattern,
                         std::string_view bos_token) :
        pattern_(std::move(pattern)), splitter_(std::make_unique<Pcre2Splitter>(pattern_)),
        decoder_(std::move(mergeable_ranks)) {
        const size_t num_ranks = decoder_.size();
        for (const auto &[name, id]: special_tokens) {
            if (id < num_ranks)
                throw std::invalid_argument("special token id collides with a mergeable rank: " + name);
            if (id >= decoder_.size())
                decoder_.resize(id + 1);
            decoder_[id] = name;
            special_encoder_[name] = id;
        }
        // decoder_ is final: safe to take views. Duplicate bytes keep the last rank, like a Python dict.
        encoder_.reserve(num_ranks);
        for (token_t id = 0; id < num_ranks; ++id)
            encoder_[decoder_[id]] = id;
        bos_ = encode_special(bos_token);
    }

    Tokenizer Tokenizer::from_ranks(std::vector<std::string> mergeable_ranks) {
        std::vector<std::pair<std::string, token_t>> specials;
        const auto offset = static_cast<token_t>(mergeable_ranks.size());
        for (size_t i = 0; i < SPECIAL_TOKENS.size(); ++i)
            specials.emplace_back(SPECIAL_TOKENS[i], offset + i);
        return {std::move(mergeable_ranks), specials};
    }

    Tokenizer Tokenizer::train(const TextSource &source, uint32_t vocab_size, int num_threads) {
        if (vocab_size < 256 + SPECIAL_TOKENS.size())
            throw std::invalid_argument("vocab_size must leave room for 256 bytes and the special tokens");
        Pcre2Splitter splitter;
        ChunkCounter counter(splitter, num_threads);
        std::vector<std::string> docs;
        while (source(docs))
            counter.add(docs);
        return from_ranks(train_bpe(counter.counts(), vocab_size - SPECIAL_TOKENS.size()));
    }

    Tokenizer Tokenizer::load(const std::filesystem::path &json_path, std::string_view bos_token) {
        std::ifstream in(json_path);
        if (!in)
            throw std::runtime_error("cannot open " + json_path.string());
        const json j = json::parse(in);
        std::vector<std::string> ranks;
        for (const auto &b64: j.at("mergeable_ranks"))
            ranks.push_back(base64_decode(b64.get<std::string>()));
        std::vector<std::pair<std::string, token_t>> specials;
        for (const auto &[name, id]: j.at("special_tokens").items())
            specials.emplace_back(name, id.get<token_t>());
        return {std::move(ranks), specials, j.at("pattern").get<std::string>(), bos_token};
    }

    void Tokenizer::save(const std::filesystem::path &json_path) const {
        json j;
        j["pattern"] = pattern_;
        auto &ranks = j["mergeable_ranks"] = json::array();
        auto &specials = j["special_tokens"] = json::object();
        for (token_t id = 0; id < decoder_.size(); ++id) {
            if (auto it = special_encoder_.find(decoder_[id]); it != special_encoder_.end() && it->second == id)
                specials[decoder_[id]] = id;
            else
                ranks.push_back(base64_encode(decoder_[id]));
        }
        if (json_path.has_parent_path())
            std::filesystem::create_directories(json_path.parent_path());
        std::ofstream(json_path) << j.dump();
    }

    std::vector<std::string> Tokenizer::special_tokens() const {
        std::vector<std::string> names;
        for (const auto &[name, _]: special_encoder_)
            names.push_back(name);
        return names;
    }

    token_t Tokenizer::encode_special(std::string_view name) const {
        auto it = special_encoder_.find(std::string(name));
        if (it == special_encoder_.end())
            throw std::invalid_argument("unknown special token: " + std::string(name));
        return it->second;
    }

    // Port of tiktoken's byte_pair_encode / _byte_pair_merge.
    void Tokenizer::byte_pair_encode(std::string_view piece, std::vector<token_t> &out) const {
        auto rank_of = [&](size_t start, size_t end) {
            auto it = encoder_.find(piece.substr(start, end - start));
            return it == encoder_.end() ? kNoRank : it->second;
        };
        if (piece.size() == 1) {
            out.push_back(rank_of(0, 1));
            return;
        }
        // parts[i] = (start of part i, rank of merging part i with part i+1)
        struct Part {
            size_t start;
            token_t rank;
        };
        thread_local std::vector<Part> parts;
        parts.clear();
        std::pair<token_t, size_t> min_rank{kNoRank, SIZE_MAX};
        for (size_t i = 0; i + 1 < piece.size(); ++i) {
            token_t r = rank_of(i, i + 2);
            if (r < min_rank.first)
                min_rank = {r, i};
            parts.push_back({i, r});
        }
        parts.push_back({piece.size() - 1, kNoRank});
        parts.push_back({piece.size(), kNoRank});

        // rank of merging part i with part i+1, before part i+1 is removed (hence i+3)
        auto get_rank = [&](size_t i) {
            return i + 3 < parts.size() ? rank_of(parts[i].start, parts[i + 3].start) : kNoRank;
        };
        while (min_rank.first != kNoRank) {
            const size_t i = min_rank.second;
            if (i > 0)
                parts[i - 1].rank = get_rank(i - 1);
            parts[i].rank = get_rank(i);
            parts.erase(parts.begin() + static_cast<ptrdiff_t>(i) + 1);

            min_rank = {kNoRank, SIZE_MAX};
            for (size_t j = 0; j + 1 < parts.size(); ++j) {
                if (parts[j].rank < min_rank.first)
                    min_rank = {parts[j].rank, j};
            }
        }
        for (size_t j = 0; j + 1 < parts.size(); ++j)
            out.push_back(rank_of(parts[j].start, parts[j + 1].start));
    }

    void Tokenizer::encode_ordinary(std::string_view text, std::vector<token_t> &out) const {
        thread_local std::vector<std::string_view> chunks;
        chunks.clear();
        splitter_->split(text, chunks);
        for (auto piece: chunks) {
            if (auto it = encoder_.find(piece); it != encoder_.end())
                out.push_back(it->second);
            else
                byte_pair_encode(piece, out);
        }
    }

    std::vector<token_t> Tokenizer::encode(std::string_view text, std::optional<token_t> prepend,
                                           std::optional<token_t> append) const {
        std::vector<token_t> ids;
        if (prepend)
            ids.push_back(*prepend);
        encode_ordinary(text, ids);
        if (append)
            ids.push_back(*append);
        return ids;
    }

    std::vector<std::vector<token_t>> Tokenizer::encode_batch(std::span<const std::string> texts,
                                                              std::optional<token_t> prepend,
                                                              std::optional<token_t> append, int num_threads) const {
        std::vector<std::vector<token_t>> out(texts.size());
        const auto nt = static_cast<size_t>(std::clamp<int>(num_threads, 1, std::max<int>(1, texts.size())));
        std::vector<std::jthread> threads;
        for (size_t t = 0; t < nt; ++t) {
            threads.emplace_back([&, t] {
                for (size_t i = t; i < texts.size(); i += nt)
                    out[i] = encode(texts[i], prepend, append);
            });
        }
        threads.clear(); // join
        return out;
    }

    std::string Tokenizer::decode_bytes(std::span<const token_t> ids) const {
        std::string out;
        for (auto id: ids)
            out += decode_single_token_bytes(id);
        return out;
    }

    std::string Tokenizer::decode(std::span<const token_t> ids) const { return utf8_replace_invalid(decode_bytes(ids)); }

    const std::string &Tokenizer::decode_single_token_bytes(token_t id) const {
        if (id >= decoder_.size())
            throw std::out_of_range("invalid token id " + std::to_string(id));
        return decoder_[id];
    }

    RenderedConversation Tokenizer::render_conversation(const json &conversation, size_t max_tokens) const {
        RenderedConversation r;
        auto add = [&](token_t id, uint8_t m) {
            r.ids.push_back(id);
            r.mask.push_back(m);
        };
        auto add_text = [&](const std::string &text, uint8_t m) {
            encode_ordinary(text, r.ids);
            r.mask.resize(r.ids.size(), m);
        };

        // a leading system message is merged into the first user message
        json messages = conversation.at("messages");
        if (!messages.empty() && messages[0].at("role") == "system") {
            if (messages.size() < 2 || messages[1].at("role") != "user")
                throw std::invalid_argument("System message must be followed by a user message");
            messages[1]["content"] =
                    messages[0].at("content").get<std::string>() + "\n\n" + messages[1].at("content").get<std::string>();
            messages.erase(messages.begin());
        }
        if (messages.empty())
            throw std::invalid_argument("Conversation has less than 1 message");

        const token_t user_start = encode_special("<|user_start|>"), user_end = encode_special("<|user_end|>");
        const token_t assistant_start = encode_special("<|assistant_start|>");
        const token_t assistant_end = encode_special("<|assistant_end|>");
        const token_t python_start = encode_special("<|python_start|>"), python_end = encode_special("<|python_end|>");
        const token_t output_start = encode_special("<|output_start|>"), output_end = encode_special("<|output_end|>");

        add(bos_, 0);
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto &message = messages[i];
            const std::string role = message.at("role");
            const std::string must_be_from = i % 2 == 0 ? "user" : "assistant";
            if (role != must_be_from)
                throw std::invalid_argument("Message " + std::to_string(i) + " is from " + role + " but should be from " +
                                            must_be_from);
            const auto &content = message.at("content");
            if (role == "user") {
                if (!content.is_string())
                    throw std::invalid_argument("User messages are simply expected to be strings");
                add(user_start, 0);
                add_text(content.get<std::string>(), 0);
                add(user_end, 0);
                continue;
            }
            add(assistant_start, 0);
            if (content.is_string()) {
                add_text(content.get<std::string>(), 1);
            } else if (content.is_array()) {
                for (const auto &part: content) {
                    const std::string type = part.at("type");
                    const std::string text = part.at("text");
                    if (type == "text") {
                        add_text(text, 1);
                    } else if (type == "python") {
                        add(python_start, 1);
                        add_text(text, 1);
                        add(python_end, 1);
                    } else if (type == "python_output") {
                        // comes from the interpreter at test time: not supervised
                        add(output_start, 0);
                        add_text(text, 0);
                        add(output_end, 0);
                    } else {
                        throw std::invalid_argument("Unknown part type: " + type);
                    }
                }
            } else {
                throw std::invalid_argument("Unknown content type");
            }
            add(assistant_end, 1);
        }

        if (r.ids.size() > max_tokens) {
            r.ids.resize(max_tokens);
            r.mask.resize(max_tokens);
        }
        return r;
    }

    std::vector<token_t> Tokenizer::render_for_completion(const json &conversation) const {
        json conv = conversation;
        auto &messages = conv.at("messages");
        if (messages.empty() || messages.back().at("role") != "assistant")
            throw std::invalid_argument("Last message must be from the Assistant");
        messages.erase(messages.end() - 1);
        auto ids = render_conversation(conv).ids;
        ids.push_back(encode_special("<|assistant_start|>"));
        return ids;
    }

    std::vector<int32_t> Tokenizer::token_bytes() const {
        std::vector<int32_t> out(decoder_.size());
        for (token_t id = 0; id < decoder_.size(); ++id) {
            auto it = special_encoder_.find(decoder_[id]);
            const bool special = it != special_encoder_.end() && it->second == id;
            out[id] = special ? 0 : static_cast<int32_t>(decoder_[id].size());
        }
        return out;
    }

} // namespace nanochat
