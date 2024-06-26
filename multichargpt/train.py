from typing import Optional, Callable, List

from tqdm import tqdm


# TODO set up params better to use config for experimentation
def train_language_model(
    model,
    dataset,
    optimizer,
    iterations,
    post_hooks: Optional[List[Callable]] = None,
):
    loss = None
    losses = list()
    if post_hooks is None:
        post_hooks = list()
    for step in tqdm(range(iterations)):
        x, y = dataset.get_batch("train")

        model.zero_grad(set_to_none=True)
        # ^ set to none is default True in 2.0 (should save mem but may cost in allocation?)
        logits = model(x)
        loss = model.loss(logits, y)
        loss.backward()
        optimizer.step()

        for hook in post_hooks:
            # TODO improve this to be less restrictive
            hook(
                step=step,
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                logits=logits,
                loss=loss,
                losses=losses,
            )
    assert loss  # assert loss is not None for type hints #nosec
    return model, loss.item(), losses
