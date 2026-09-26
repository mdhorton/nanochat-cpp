// safetensors files: an 8-byte little-endian header size, a JSON header, then raw tensor data.
// https://github.com/huggingface/safetensors
#pragma once

#include <filesystem>
#include <map>
#include <string>

#include <torch/torch.h>

namespace nanochat::safetensors {

using TensorMap = std::map<std::string, torch::Tensor>;
using Metadata = std::map<std::string, std::string>;

// Writes tensors (copied to CPU, contiguous) and optional string metadata.
void save(const std::filesystem::path& path, const TensorMap& tensors, const Metadata& metadata = {});

// Reads all tensors onto device.
TensorMap load(const std::filesystem::path& path, torch::Device device = torch::kCPU);

// Reads only the "__metadata__" entry.
Metadata load_metadata(const std::filesystem::path& path);

} // namespace nanochat::safetensors
