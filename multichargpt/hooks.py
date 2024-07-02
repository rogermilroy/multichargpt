import logging
import os
from typing import Callable

import torch

logger = logging.getLogger(__name__)


class Validate:
    def __init__(self, eval_fn: Callable, eval_iters: int, validate_interval: int):
        self.eval_fn = eval_fn
        self.eval_iters = eval_iters
        self.validate_interval = validate_interval

    def __call__(self, step, model, dataset, losses, **kwargs):
        if step % self.validate_interval == 0:
            checkpoint_losses = self.eval_fn(
                model=model, dataset=dataset, eval_iters=self.eval_iters
            )
            losses.append(
                f"Step {step} Train loss: {checkpoint_losses['train']:.4f} "
                f"| Val loss: {checkpoint_losses['val']:.4f}"
            )


class TextSample:
    def __init__(
        self, sample_interval: int, tokens: int, device, tokenizer, chunk_size
    ):
        self.interval = sample_interval
        self.device = device
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.chunk_size = chunk_size

    def __call__(self, step, model, samples, **kwargs):
        if step % self.interval == 0:
            inputs = torch.zeros(
                (1, self.chunk_size), dtype=torch.long, device=self.device
            )
            model.eval()
            samples.append(
                f"\n##### Step: {step} sample #####\n"
                f"{self.tokenizer.decode(model.generate(inputs, generate_limit=self.tokens)[0])}"
                f"\n#####"
            )


class Checkpoint:
    def __init__(self, checkpoint_interval):
        self.checkpoint_interval = checkpoint_interval

    def __call__(self, step, model, optimizer, loss, **kwargs):
        if step % self.checkpoint_interval == 0:
            # create the checkpoint name - might want it to
            checkpoint_fname = os.path.join(
                os.getcwd(), os.path.join("checkpoints", f"checkpoint_{step}.pt")
            )
            logger.debug("Saving checkpoint")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_fname,
            )
