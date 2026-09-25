# c++ port of nanochat

This continues the saga of exploring sm120 GPUs. Specifically 2x RTX Pro 4000, which is my local setup.

My first pass created a fork of nanochat and made python only changes, no c++. This worked but it required fairly
intensive changes. For example, sm120 doesn't support FA3. FA2 works pretty good, but it's also not designed for sm120.

# goals

After thinking about it, I decided on the following goals (or questions really):

1. how fast can claude port nanochat to c++?
2. how fast can we train using sm120?

By itself, c++ won't speed things up.

# why c++ then?

To be honest, curiosity. This is first and foremost a learning project.

With LLM agents at our disposal, it certainly makes the task significantly easier. I never would have attempted this
without support from an agent.
