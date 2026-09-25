# c++ port of nanochat

This continues the saga of exploring sm120 GPUs. Specifically 2x RTX Pro 4000, which is my local setup.

My first pass modified a fork of nanochat. This was python only and no c++. This worked but required lots of changes.
For example, sm120 doesn't support FA3. FA2 works pretty good, but its not designed for sm120 either.

# goals

After thinking about it, I decided on the following goals:

1. how long would it take for claude/codex to port nanochat to c++
2. how fast can we train using sm120



# why c++?

To be honest, curiosity. This is first and foremost a learning project.

With LLM agents at our disposal, it certainly makes the task significantly easier. I never would have attempted this
without support from an agent.
