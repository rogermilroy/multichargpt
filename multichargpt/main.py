import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from multichargpt.dataset import ShakespeareDataset, SizedSubset, partition_dataset
from multichargpt.hooks import TextSample, Validate, Checkpoint
from multichargpt.model import (
    TransformerFixedLookahead,
    TransformerMultiBlockLanguageModel,
)
from multichargpt.tokenizer import IndexTokenizer
from multichargpt.train import train_language_model

project_base_dir = os.path.dirname(os.path.abspath(__file__))


logger = logging.getLogger(__name__)


def available_device() -> str:
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.backends.cuda.is_built():
        return "cuda"
    else:
        return "cpu"


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_training(cfg: DictConfig):

    models = {
        "chunked": TransformerFixedLookahead,
        "standard": TransformerMultiBlockLanguageModel,
    }

    out_dir = HydraConfig.get().runtime.output_dir
    logger.debug(
        f"CWD: {os.getcwd()}, project root: {project_base_dir}, output_dir: "
        f"{out_dir}"
    )

    data_filename = os.path.join(
        project_base_dir, f"{cfg['data']['data_dir']}/{cfg['data']['data_filename']}"
    )
    tok = IndexTokenizer()

    device = available_device() if cfg["device"] == "available" else cfg["device"]
    logger.info(f"Device: {device}")

    # TODO change to create dataset and dataloaders

    base_dataset = ShakespeareDataset(
        filename=data_filename,
        tokenizer=tok,
        device=device,
        **cfg["shared"],
    )

    train, test = partition_dataset(
        base_dataset, test_proportion=cfg["data"]["test_proportion"], **cfg["shared"]
    )

    train, val = partition_dataset(train, test_proportion=cfg["data"]["val_proportion"], **cfg["shared"])  # type: ignore

    train_dataloader = DataLoader(train, **cfg["dataloading"])
    val_dataloader = DataLoader(val, **cfg["dataloading"])
    test_dataloader = DataLoader(test, **cfg["dataloading"])

    sample_tokens = 200

    model = models[cfg["model_type"]](
        vocab_size=tok.vocab_size,
        **cfg["shared"],
        **cfg["model"],
    )
    model.to(device)

    # try loading weights
    if cfg["run"].get("resume") is not None:
        logger.info(f"resuming from : {cfg['run'].get('resume')}")
        checkpoint = torch.load(
            os.path.join(project_base_dir, cfg["run"]["resume"]["path"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.AdamW(params=model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:
        logger.info("Starting from scratch")
        optimizer = torch.optim.AdamW(params=model.parameters(), **cfg["optimizer"])

    #### Before sample #####
    inputs = torch.zeros(
        (1, cfg["shared"]["chunk_size"]), dtype=torch.long, device=device
    )
    model.eval()
    logger.info(
        f"\n##### Before #####\n"
        f"{tok.decode(model.generate(inputs, tokens=sample_tokens)[0])}"
        f"\n##### Before #####"
    )
    #### Before sample #####

    # TODO - extract hooks setup into a helper function?
    post_hooks = list()
    if cfg["hooks"]["validate"]:
        post_hooks.append(
            Validate(
                dataloader=val_dataloader,
                batch_size=cfg["dataloading"]["batch_size"],
                **cfg["hooks"]["validate"],
                **cfg["shared"],
            )
        )
    if cfg["hooks"]["sample"]:
        post_hooks.append(
            TextSample(
                **cfg["hooks"]["sample"],
                device=device,
                tokenizer=tok,
                batch_size=cfg["dataloading"]["batch_size"],
                **cfg["shared"],
            )
        )
    if cfg["hooks"]["checkpoint"]:
        os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
        post_hooks.append(Checkpoint(**cfg["hooks"]["checkpoint"]))

    model.train()
    trained_model, final_loss, train_losses, val_losses, samples = train_language_model(
        epochs=cfg["run"]["epochs"],
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        post_hooks=post_hooks,
    )
    # save final model
    if cfg["save_final"]:
        torch.save(
            {
                "epoch": cfg["run"]["epochs"],
                "minibatch": len(
                    train_dataloader
                ),  # TODO think more about this - sub integer epochs...
                "model_state_dict": trained_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": final_loss,
            },
            # TODO maybe change filename to match normal naming scheme?
            os.path.join(os.getcwd(), os.path.join("checkpoints", "final.pt")),
        )

    # TODO remove all this - when flushing losses directly in hooks
    for train_loss in train_losses:
        logger.info(train_loss)
    # TODO change to test loss
    logger.info(f"Final Loss: {final_loss}")
    for val_loss in val_losses:
        logger.info(val_loss)
    for sample in samples:
        logger.info(sample)

    #### After sample #####
    inputs = torch.zeros(
        (1, cfg["shared"]["chunk_size"]), dtype=torch.long, device=device
    )
    trained_model.eval()
    logger.info(
        f"\n##### After #####\n"
        f"{tok.decode(trained_model.generate(inputs, tokens=sample_tokens)[0])}"
        f"\n##### After #####"
    )
    #### After sample #####


if __name__ == "__main__":
    torch.manual_seed(42)
    run_training()
