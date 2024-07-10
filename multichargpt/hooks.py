from abc import ABC
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from multichargpt.model import TorchLanguageModel

logger = logging.getLogger(__name__)


class Hook(ABC):
    def __call__(self, epoch, minibatch, model, train_losses, val_losses, **kwargs): ...


# TODO refactor to not use eval_iters - just len of dataloader
@torch.no_grad()
def evaluate_val(model: TorchLanguageModel, dataloader: DataLoader):
    model.eval()

    losses = torch.zeros(len(dataloader))
    # TODO fix the train, val
    for i, minibatch in enumerate(dataloader):
        x, y = minibatch
        logits = model(x)
        loss = model.loss(logits=logits, targets=y)
        losses[i] = loss.item()
    out = losses.mean()

    model.train()
    return out


class Validate(Hook):
    def __init__(
        self,
        dataloader: DataLoader,
        validate_interval: int,
        context_size: int,
        batch_size: int,
        **kwargs,
    ):
        self.validate_interval = validate_interval
        self.context_size = context_size
        self.batch_size = batch_size
        self.dataloader = dataloader

    def __call__(self, epoch: int, minibatch: int, model, val_losses: list, **kwargs):
        if minibatch % self.validate_interval == 0:
            checkpoint_losses = evaluate_val(model=model, dataloader=self.dataloader)
            val_losses.append(
                f"Epoch: {epoch} Minibatch: {minibatch} Tokens: {minibatch * self.context_size * self.batch_size} "
                f"| Val loss: {checkpoint_losses:.4f}"
            )


class TrainLoss(Hook):
    # accumulate training losses - average
    def __init__(self, interval: int, context_size: int, batch_size: int, **kwargs):
        # interval in minibatches
        self.train_loss_interval = interval
        self.losses = torch.zeros(self.train_loss_interval)
        self.context_size = context_size
        self.batch_size = batch_size

    def __call__(self, epoch, minibatch, loss, train_losses: list):
        self.losses[minibatch % self.train_loss_interval] = loss.item()
        if minibatch % self.train_loss_interval == 0:
            train_losses.append(
                f"Epoch: {epoch} Minibatch: {minibatch} Tokens: {minibatch * self.context_size * self.batch_size} "
                f"| Train loss: {self.losses.mean():.4f}"
            )
            # reset losses for the next interval
            self.losses = torch.zeros(self.train_loss_interval)


class TextSample(Hook):
    def __init__(
        self,
        sample_interval: int,
        tokens: int,
        device,
        tokenizer,
        context_size,
        batch_size,
        chunk_size,
    ):
        self.interval = sample_interval
        self.device = device
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.context_size = context_size
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def __call__(self, epoch, minibatch, model, samples, **kwargs):
        if minibatch % self.interval == 0:
            inputs = torch.zeros(
                (1, self.chunk_size), dtype=torch.long, device=self.device
            )
            model.eval()
            samples.append(
                # TODO fix the tokens calculation - needs to account for which epoch its in - maybe...
                f"\n##### Epoch: {epoch} Minibatch: {minibatch} Tokens: { minibatch * self.context_size * self.batch_size} sample #####\n"
                f"{self.tokenizer.decode(model.generate(inputs, tokens=self.tokens)[0])}"
                f"\n#####"
            )
            model.train()


class Checkpoint(Hook):
    def __init__(self, checkpoint_interval):
        self.checkpoint_interval = checkpoint_interval

    def __call__(self, epoch, minibatch, model, optimizer, loss, **kwargs):
        if minibatch % self.checkpoint_interval == 0:
            # create the checkpoint name - might want it to
            checkpoint_fname = os.path.join(
                os.getcwd(),
                os.path.join("checkpoints", f"checkpoint_{epoch}_{minibatch}.pt"),
            )
            logger.debug("Saving checkpoint")
            torch.save(
                {
                    "epoch": epoch,
                    "minibatch": minibatch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_fname,
            )
