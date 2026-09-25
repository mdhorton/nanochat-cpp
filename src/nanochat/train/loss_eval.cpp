#include "nanochat/train/loss_eval.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace nanochat {

    torch::Tensor load_token_bytes(const std::filesystem::path &path, torch::Device device) {
        const auto size = std::filesystem::file_size(path);
        if (size % sizeof(int32_t) != 0)
            throw std::runtime_error("bad token_bytes file " + path.string());
        auto t = torch::empty({static_cast<int64_t>(size / sizeof(int32_t))}, torch::kInt32);
        std::ifstream in(path, std::ios::binary);
        in.read(static_cast<char *>(t.data_ptr()), static_cast<std::streamsize>(size));
        if (!in)
            throw std::runtime_error("cannot read " + path.string());
        return t.to(device);
    }

    double evaluate_bpb(GPTImpl &model, DataLoader &batches, int64_t steps, const torch::Tensor &token_bytes) {
        torch::NoGradGuard no_grad;
        const auto device = token_bytes.device();
        auto total_nats = torch::zeros({}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto total_bytes = torch::zeros({}, torch::TensorOptions().device(device).dtype(torch::kInt64));
        for (int64_t i = 0; i < steps; ++i) {
            auto [x, y] = batches.next();
            auto loss = model.forward(x, y, torch::kNone).view(-1);
            auto targets = y.view(-1);
            torch::Tensor num_bytes;
            if ((targets.to(torch::kInt32) < 0).any().item<bool>()) {
                // ignored targets (-1) contribute 0 bytes; don't index token_bytes with them
                auto valid = targets >= 0;
                auto safe = torch::where(valid, targets, torch::zeros_like(targets));
                num_bytes = torch::where(valid, token_bytes.index({safe}), torch::zeros_like(targets, token_bytes.dtype()));
            }
            else {
                num_bytes = token_bytes.index({targets});
            }
            total_nats += (loss * (num_bytes > 0)).sum();
            total_bytes += num_bytes.sum();
        }
        const double nats = total_nats.item<float>();
        const auto bytes = total_bytes.item<int64_t>();
        if (bytes == 0)
            return std::numeric_limits<double>::infinity();
        return nats / (std::log(2.0) * static_cast<double>(bytes));
    }

} // namespace nanochat
