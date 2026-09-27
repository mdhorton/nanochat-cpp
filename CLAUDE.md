# packages

do not install, remove, upgrade or downgrade any packages (system-wide or within the project).

do not make any system-wide changes (eg, OS level changes). let me know if something needs to be changed on the system
and I will do it.

let me know if there are any external projects or packages that would speed development. no need to re-invent the wheel.
but let me make the decision and install them.

# metrics

- notable run history with comments: RUNS.md
- historical run metric jsonl files: cache/metrics
- cache/metrics/medium-d12 line up with data in RUNS.md

# testing code changes

after making code changes, if you need to run a test that uses the GPU, run the shorted test reasonably possible. for
example, usually a simple d4 or d12 test will suffice.

Leave the longer running tests to me.

# target GPU

RTX Pro 4000 Blackwell (sm120)

measured roofline data for this GPU: roofline/cuda-rtx-pro-4000-blackwell-0-peak.json
