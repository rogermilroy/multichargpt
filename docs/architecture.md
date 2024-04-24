# Architecture

General concept is that the model already predicts multiple outputs, the same size as the input context. We can think of this conceptually as the model is mapping tokens onto the next token. We select the last token and go again, ignoring the repredicted tokens.

There are two aspects to this. One is we could use repredicted tokens somehow - this seems less helpful but could allow some version of deliberative thinking.

The more important thought is that we can train in a manner that shifts the lookahead of the predictions from 1 to more than 1.

There are two options for this:
    1. Fixed size lookahead. We will need a minimum number of tokens to start the model. This shouldn't be a real problem as we can use \<empty> or \<start> tokens to take up this empty space
    2. Variable size lookahead. We add a parameter to the model input to dictate how many tokens to predict ahead. This is a harder problem for a model to solve and we should expect an increase in model complexity required to solve this new problem. This is because the weights will have to account for multiple different stepsize predictions, each of which is a different problem.
    This should be learnable by an MLP (add to the end of the transformer blocks but before the output layer) but it may need to be larger/more complex for this. Probably no need to change the Transformer blocks.

I will try both.
