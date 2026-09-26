// Multi-GPU: an NCCL process group (c10d) over a TCPStore, and a launcher that runs one process per GPU.
#pragma once

#include <string>
#include <vector>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/torch.h>

namespace nanochat {

class Dist {
public:
  using Work = c10::intrusive_ptr<c10d::Work>; // null for no-ops; wait() makes the current stream wait
  enum class Op { Sum, Avg, Max };

  Dist() = default; // one process: collectives are no-ops
  // Rank 0 hosts the store at master_addr:master_port. Binds this process to GPU `rank`.
  Dist(int rank, int world_size, const std::string& master_addr, int master_port);
  ~Dist();
  Dist(const Dist&) = delete;
  Dist& operator=(const Dist&) = delete;

  int rank() const {
    return rank_;
  }

  int world_size() const {
    return world_size_;
  }

  // Async, as Python's async_op=True.
  Work all_reduce(torch::Tensor& t, Op op);
  Work reduce_scatter(torch::Tensor& out, torch::Tensor& in, Op op); // out = in's rank-th slice of dim 0
  Work all_gather(torch::Tensor& out, torch::Tensor& in);            // in may be out's rank-th slice (in place)
  void barrier();

  static void wait(const Work& work) {
    if (work)
      work->wait();
  }

private:
  int rank_ = 0, world_size_ = 1;
  c10::intrusive_ptr<c10d::Store> store_;
  c10::intrusive_ptr<c10d::Backend> pg_;
};

// Runs this executable (/proc/self/exe) once per rank with `args` plus "--rank r"; returns 0 if every rank
// succeeds, else the first failure's exit code (the other ranks are then terminated).
int launch_ranks(const std::vector<std::string>& args, int nproc);

} // namespace nanochat
