# c++ port of nanochat

This continues the saga of exploring sm120 GPUs. Specifically 2x RTX Pro 4000, which is my local setup. With the added
twist that we're going with c++.

My first pass created a fork of nanochat and made python only changes, no c++. This worked but it required a good many
changes. For example, sm120 doesn't support FA3. FA2 works pretty good, but it's also not designed for sm120. It often
selected ampere kernels.

# goals

After thinking about it, I decided on the following goals (or questions really):

1. how fast can claude port nanochat to c++?
2. how fast can we train the model using sm120?

# why c++ then?

To be honest, curiosity. This is first and foremost a learning project.

c++ has nothing to do with making nanochat work with sm120 GPUs. By itself, c++ won't speed things up.

Does the language even matter? With powerful LLMs, these porting exercises become almost trivial. Translation from one
language to another is one of LLMs strongest features.
