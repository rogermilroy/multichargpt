import math
from typing import Optional, Callable, List
from torch.utils.data import DataLoader

from tqdm import tqdm

from multichargpt.hooks import Hook
from multichargpt.hooks import evaluate_val


# TODO set up params better to use config for experimentation
# TODO change to use dataloader and epochs over iterations
def train_language_model(
    epochs: float,
    model,
    optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    post_hooks: Optional[List[Hook]] = None,
):
    loss = None
    val_losses = list()
    train_losses = list()
    samples = list()
    if post_hooks is None:
        post_hooks = list()

    # do some processing to allow sub epoch runs (ie. half a dataset = 0.5)
    minibatch_limit = None
    if 0 < epochs < 1:
        minibatch_limit = len(train_dataloader) * epochs
        # TODO add printout for minibatch limit if it's set here.
    else:
        minibatch_limit = math.inf

    for epoch in tqdm(range(math.ceil(epochs))):
        for i, minibatch in train_dataloader:
            if i > minibatch_limit:
                break
            x, y = minibatch

            model.zero_grad(set_to_none=True)
            # ^ set to none is default True in 2.0 (should save mem but may cost in allocation?)
            logits = model(x)
            loss = model.loss(logits, y)
            loss.backward()
            optimizer.step()

            for hook in post_hooks:
                # TODO improve this to be less restrictive
                hook(
                    epoch=epoch,
                    minibatch=i,
                    model=model,
                    optimizer=optimizer,
                    logits=logits,
                    loss=loss,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    samples=samples,
                )
    # test loss is the mean loss over the whole test dataset
    test_loss = evaluate_val(model=model, dataloader=test_dataloader)

    return model, test_loss, train_losses, val_losses, samples
