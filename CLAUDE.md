# packages

Do not install, remove, upgrade or downgrade any packages (system-wide or within the project).

Do not make system-wide changes (eg, OS level changes). Let me know if something needs to be changed on the system and I
will do it.

Let me know if there are any external projects or packages that would speed development. No need to re-invent the wheel.
However, let me make the decision and install them.

# testing code changes

After making code changes, if you need to run a test that uses the GPU, run the shortest test reasonably possible. For
example, usually a short d12 test will suffice.

Leave the longer running tests to me.

Always check to see if the GPUs are in use before running a test that will hit the GPU.

# sm120 target GPUs

- RTX Pro 4000
- RTX Pro 6000
- RTX 5090
