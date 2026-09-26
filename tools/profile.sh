#!/usr/bin/env bash

# nsys then ncu on base_train, as a report pair cache/profiles/<model>_N.{nsys,ncu}-rep.
#   nsys: 2 GPUs, steps 5-7 (timeline). ncu: 1 GPU, kernels in step 2 (NVTX range "profile").
# usage: tools/profile.sh d12|d24 [nsys|ncu|both] [ncu flags...] [-- base_train flags...]
# e.g.   tools/profile.sh d12 ncu --metrics gpu__time_duration.sum --kernel-name regex:nvjet

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 d12|d24 [nsys|ncu|both] [ncu flags...] [-- base_train flags...]" >&2
  exit 2
}

case ${1:-} in
  d12) model=(--depth=12 --device-batch-size=8) ;;
  d24) model=(--depth=24 --device-batch-size=2) ;;
  *) usage ;;
esac

tag=$1
shift

mode=both
case ${1:-} in
  nsys | ncu | both) mode=$1 && shift ;;
esac

ncu_args=() train_args=()
while (($#)); do
  if [[ $1 == -- ]]; then
    shift
    train_args=("$@")
    break
  fi
  ncu_args+=("$1")
  shift
done

if [[ $mode == nsys && ${#ncu_args[@]} -gt 0 ]]; then
  echo "ncu flags given with mode nsys: ${ncu_args[*]}" >&2
  usage
fi

# next available `i` for both report types
mkdir -p cache/profiles
for ((i = 1; ; ++i)); do
  report=cache/profiles/${tag}_$i
  [[ -e $report.nsys-rep || -e $report.ncu-rep ]] || break
done

base_train=(cmake-build-pixi/base_train --fp8 "${model[@]}" "${train_args[@]}" --eval-every=0 --save=false)

# default ncu metrics (used unless --metrics is given)
ncu_metrics=(
  sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum
  sm__ops_path_tensor_src_fp8_dst_fp32.sum
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
  smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
  smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
  sm__sass_data_bytes_mem_shared.sum
  l1tex__t_bytes.sum
  l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum
  l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum
  lts__t_bytes.sum
  dram__bytes.sum
  gpu__time_duration.sum
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__sass_inst_executed.sum
  sm__sass_inst_executed_op_local_ld.sum
  sm__sass_inst_executed_op_local_st.sum
  l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
  l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
  l1tex__t_bytes_pipe_lsu_mem_local_op_ld_lookup_miss.sum
  l1tex__t_bytes_pipe_lsu_mem_local_op_st_lookup_miss.sum
  sm__cycles_active.avg
  sm__cycles_active.min
  sm__cycles_active.max
  sm__cycles_active.sum
  sm__cycles_elapsed.sum
  smsp__cycles_active.avg
  smsp__cycles_active.min
  smsp__cycles_active.max
  smsp__cycles_active.sum
  smsp__cycles_elapsed.sum
  smsp__thread_inst_executed_per_inst_executed.ratio
  smsp__thread_inst_executed_pred_on_per_inst_executed.ratio
  smsp__issue_active.avg.per_cycle_active
  smsp__average_warps_active_per_inst_executed.ratio
  smsp__average_warps_issue_stalled_long_scoreboard_pipe_l1tex_per_issue_active.ratio
  smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
  smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
  smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_mio_throttle_pipe_mio_per_issue_active.ratio
  smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
  smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
  smsp__average_warps_issue_stalled_misc_per_issue_active.ratio
  smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
  smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio
)

if [[ $mode != ncu ]]; then
  nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --nvtx-capture=profile --capture-range-end=stop \
      --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 -o "$report" \
    "${base_train[@]}" --nproc=2 --num-iterations=5 --profile-start=2 --profile-steps=3
fi

if [[ $mode != nsys ]]; then
  # defaults unless given (ncu rejects repeated options)
  has() { printf '%s\n' "${ncu_args[@]}" | grep -qE "^($1)(=|$)"; }
  ncu_defaults=()
  has '-o|--export' || ncu_defaults+=(-o "$report")
  has '--kernel-name-base' || ncu_defaults+=(--kernel-name-base demangled)
  has '--filter-mode' || ncu_defaults+=(--filter-mode per-launch-config)
  has '-c|--launch-count' || ncu_defaults+=(-c 1)
  has '--metrics' || ncu_defaults+=(--metrics "$(IFS=,; echo "${ncu_metrics[*]}")")

  ncu --nvtx --nvtx-include profile "${ncu_defaults[@]}" "${ncu_args[@]}" \
    "${base_train[@]}" --nproc=1 --num-iterations=3 --profile-start=2 --profile-steps=1
fi

echo "reports: $(ls "$report".*-rep 2>/dev/null | tr '\n' ' ')"
