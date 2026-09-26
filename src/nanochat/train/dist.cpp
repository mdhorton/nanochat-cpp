#include "nanochat/train/dist.h"

#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

extern char** environ;

namespace nanochat {

namespace {

c10d::ReduceOp reduce_op(Dist::Op op) {
  switch (op) {
    case Dist::Op::Sum:
      return c10d::ReduceOp::SUM;
    case Dist::Op::Avg:
      return c10d::ReduceOp::AVG;
    default:
      return c10d::ReduceOp::MAX;
  }
}

} // namespace

Dist::Dist(int rank, int world_size, const std::string& master_addr, int master_port)
    : rank_(rank)
    , world_size_(world_size) {
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("bad rank " + std::to_string(rank));
  const torch::Device device(torch::kCUDA, static_cast<c10::DeviceIndex>(rank));
  c10::cuda::set_device(device.index());
  if (world_size == 1)
    return;
  c10d::TCPStoreOptions store_opts;
  store_opts.port = static_cast<uint16_t>(master_port);
  store_opts.isServer = rank == 0;
  store_opts.numWorkers = world_size;
  store_ = c10::make_intrusive<c10d::TCPStore>(master_addr, store_opts);
  auto pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(store_, rank, world_size);
  pg->setBoundDeviceId(device); // as init_process_group(device_id=...): connect eagerly
  pg->eagerConnectSingleDevice(device);
  pg_ = pg;
  barrier();
}

Dist::~Dist() {
  if (pg_)
    pg_->shutdown();
}

Dist::Work Dist::all_reduce(torch::Tensor& t, Op op) {
  if (!pg_)
    return {};
  std::vector<torch::Tensor> tensors{t};
  c10d::AllreduceOptions opts;
  opts.reduceOp = reduce_op(op);
  return pg_->allreduce(tensors, opts);
}

Dist::Work Dist::reduce_scatter(torch::Tensor& out, torch::Tensor& in, Op op) {
  if (!pg_) {
    out.copy_(in);
    return {};
  }
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = reduce_op(op);
  return pg_->reduce_scatter_single(out, in, opts);
}

Dist::Work Dist::all_gather(torch::Tensor& out, torch::Tensor& in) {
  if (!pg_) {
    if (!out.is_same(in))
      out.copy_(in);
    return {};
  }
  return pg_->all_gather_single(out, in);
}

void Dist::barrier() {
  if (pg_)
    pg_->barrier()->wait();
}

int launch_ranks(const std::vector<std::string>& args, int nproc) {
  std::vector<pid_t> pids;
  for (int rank = 0; rank < nproc; ++rank) {
    std::vector<std::string> argv{"/proc/self/exe"};
    argv.insert(argv.end(), args.begin(), args.end());
    argv.insert(argv.end(), {"--rank", std::to_string(rank)});
    std::vector<char*> cargv;
    for (auto& a : argv)
      cargv.push_back(a.data());
    cargv.push_back(nullptr);
    pid_t pid;
    if (const int err = posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, cargv.data(), environ))
      throw std::runtime_error(std::string("cannot launch rank: ") + std::strerror(err));
    pids.push_back(pid);
  }
  int result = 0;
  for (size_t remaining = pids.size(); remaining > 0; --remaining) {
    int status = 0;
    const pid_t pid = waitpid(-1, &status, 0);
    if (pid < 0) {
      if (errno == EINTR) {
        ++remaining;
        continue;
      }
      break;
    }
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    if (code != 0 && result == 0) {
      result = code;
      std::cerr << "a rank failed (exit " << code << "), stopping the others" << std::endl;
      for (const pid_t other : pids)
        if (other != pid)
          kill(other, SIGTERM);
    }
  }
  return result;
}

} // namespace nanochat
