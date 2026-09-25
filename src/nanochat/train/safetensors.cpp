#include "nanochat/train/safetensors.h"

#include <bit>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nanochat::safetensors {

    using json = nlohmann::json;
    namespace fs = std::filesystem;

    static_assert(std::endian::native == std::endian::little, "safetensors data is little-endian");

    static const std::map<torch::ScalarType, std::string> &dtype_names() {
        static const std::map<torch::ScalarType, std::string> names = {
                {torch::kFloat64, "F64"}, {torch::kFloat32, "F32"}, {torch::kFloat16, "F16"},
                {torch::kBFloat16, "BF16"}, {torch::kInt64, "I64"}, {torch::kInt32, "I32"},
                {torch::kInt16, "I16"}, {torch::kInt8, "I8"}, {torch::kUInt8, "U8"},
                {torch::kBool, "BOOL"}, {torch::kFloat8_e4m3fn, "F8_E4M3"}, {torch::kFloat8_e5m2, "F8_E5M2"}};
        return names;
    }

    static torch::ScalarType dtype_from_name(const std::string &name) {
        for (const auto &[type, n]: dtype_names())
            if (n == name)
                return type;
        throw std::runtime_error("unsupported safetensors dtype: " + name);
    }

    void save(const fs::path &path, const TensorMap &tensors, const Metadata &metadata) {
        json header = json::object();
        if (!metadata.empty())
            header["__metadata__"] = metadata;
        std::vector<torch::Tensor> data;
        uint64_t offset = 0;
        for (const auto &[name, tensor]: tensors) {
            auto t = tensor.detach().to(torch::kCPU).contiguous();
            const auto it = dtype_names().find(t.scalar_type());
            if (it == dtype_names().end())
                throw std::runtime_error("unsupported dtype for " + name);
            const uint64_t nbytes = t.nbytes();
            header[name] = {{"dtype", it->second}, {"shape", t.sizes().vec()}, {"data_offsets", {offset, offset + nbytes}}};
            offset += nbytes;
            data.push_back(std::move(t));
        }
        auto text = header.dump();
        text.append((8 - text.size() % 8) % 8, ' '); // align data to 8 bytes
        const uint64_t size = text.size();

        std::ofstream out(path, std::ios::binary);
        if (!out)
            throw std::runtime_error("cannot write " + path.string());
        out.write(reinterpret_cast<const char *>(&size), sizeof(size));
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        for (const auto &t: data)
            out.write(static_cast<const char *>(t.data_ptr()), static_cast<std::streamsize>(t.nbytes()));
        if (!out)
            throw std::runtime_error("write failed: " + path.string());
    }

    // Returns the parsed header; leaves in positioned at the start of the data section.
    static json read_header(std::ifstream &in, const fs::path &path) {
        if (!in)
            throw std::runtime_error("cannot read " + path.string());
        uint64_t size = 0;
        in.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (!in || size > fs::file_size(path))
            throw std::runtime_error("bad safetensors header in " + path.string());
        std::string text(size, '\0');
        in.read(text.data(), static_cast<std::streamsize>(size));
        return json::parse(text);
    }

    TensorMap load(const fs::path &path, torch::Device device) {
        std::ifstream in(path, std::ios::binary);
        const auto header = read_header(in, path);
        const auto data_start = in.tellg();
        TensorMap tensors;
        for (const auto &[name, info]: header.items()) {
            if (name == "__metadata__")
                continue;
            const auto shape = info["shape"].get<std::vector<int64_t>>();
            const auto begin = info["data_offsets"][0].get<uint64_t>(), end = info["data_offsets"][1].get<uint64_t>();
            auto t = torch::empty(shape, torch::TensorOptions().dtype(dtype_from_name(info["dtype"])));
            if (t.nbytes() != end - begin)
                throw std::runtime_error("size mismatch for " + name + " in " + path.string());
            in.seekg(data_start + static_cast<std::streamoff>(begin));
            in.read(static_cast<char *>(t.data_ptr()), static_cast<std::streamsize>(end - begin));
            if (!in)
                throw std::runtime_error("truncated data for " + name + " in " + path.string());
            tensors[name] = device.is_cpu() ? t : t.to(device);
        }
        return tensors;
    }

    Metadata load_metadata(const fs::path &path) {
        std::ifstream in(path, std::ios::binary);
        const auto header = read_header(in, path);
        return header.contains("__metadata__") ? header["__metadata__"].get<Metadata>() : Metadata{};
    }

} // namespace nanochat::safetensors
