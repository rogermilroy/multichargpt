# multichargpt

A character level transformer to test chunked autoregression.

The foundation of the code is drawn from another character level GPT test repo of mine: [chargpt](https://github.com/rogermilroy/chargpt/)

The basic concept is that if we can generate more than 1 token per execution of a Transformer we might be able to reduce the latency for generating longer responses by the same proportion (as long as we don't introduce significant additional computational demand).

This repo is just to test a few different ways that we could do this chunked auto-regression and to see if I can get some early idea as to whether it has favourable scaling and convergence properties as compared to single token autoregression (the normal approach).
