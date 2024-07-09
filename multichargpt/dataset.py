import os
from typing import Sized, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from multichargpt.tokenizer import Tokenizer, IndexTokenizer

project_base_dir = os.path.dirname(os.path.abspath(__file__))


class SizedDataset(Dataset, Sized): ...


class SizedSubset(Subset, Sized):

    def __init__(self, dataset: Dataset, indices: os.Sequence[int]) -> None:  # type: ignore
        super().__init__(dataset, indices)
        self._size = len(indices)

    def __len__(self):
        return self._size


def partition_dataset(
    dataset: SizedDataset | SizedSubset,
    test_proportion: float,
    context_size: int,
    chunk_size: int,
) -> Tuple[Subset, Subset]:

    train = SizedSubset(
        dataset,
        range(round(len(dataset) * (1 - test_proportion)) - context_size - chunk_size),
    )
    test = SizedSubset(
        dataset, range(round(len(dataset) * (1 - test_proportion)) + 1, len(dataset))
    )

    return train, test


class ShakespeareDataset(SizedDataset):
    def __init__(
        self,
        filename,
        tokenizer: Tokenizer,
        context_size: int,
        device: str | torch.device = "cpu",
        chunk_size: int = 1,
    ):
        # NOTE this loads everything into memory - if too big load in __getitem__
        self.tokenizer = tokenizer
        with open(filename, "r", encoding="utf8") as f:
            data = f.read()
        self.tokenizer.fit(data)
        encoded_data = torch.tensor(
            self.tokenizer.encode(data), dtype=torch.long, device=device
        )
        # here stack sections of context size
        self.x = torch.stack(
            [
                encoded_data[idx : idx + context_size]
                for idx in range(len(encoded_data) - context_size)
            ]
        )
        slices = list()
        for i in range(1, chunk_size + 1):
            slices.append(
                torch.stack(
                    [
                        encoded_data[idx + i : idx + context_size + i]
                        for idx in range(len(encoded_data) - context_size - chunk_size)
                    ]
                )
            )
        self.y = torch.stack(slices, axis=-1).squeeze()  # type: ignore

    def __len__(self) -> int:
        # TODO check this - I think I need to account for context size and chunk size
        return len(self.x)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]


# Basic data handling functionality - for the initial implementation
# to be rewritten using Dataset and Dataloader
# this provides a behaviour specification for the above.


class BasicShakespeareDataset:
    def __init__(
        self,
        filename,
        tokenizer: Tokenizer,
        context_size: int,
        batch_size: int,
        val_proportion: float,
        chunk_size: int = 1,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        """

        :param filename:
        :param tokenizer:
        :param context_size: dimension of the context (i.e. length of an input in time
        dimension)
        :param batch_size: size of a batch
        :param val_proportion:
        :param device:
        """
        self.context_len = context_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        with open(filename, "r", encoding="utf8") as f:
            data = f.read()
        self.tokenizer.fit(data)
        encoded_data = torch.tensor(
            self.tokenizer.encode(data), dtype=torch.long, device=device
        )
        # very simple train val split here - TODO improve
        self.train_data = encoded_data[
            : round(len(encoded_data) * (1 - val_proportion))
        ]
        self.val_data = encoded_data[
            round(len(encoded_data) * val_proportion) + 1 :
        ]  # TODO check if 1 should be chunk_size
        self.split_dataset_map = {"train": self.train_data, "val": self.val_data}
        self.chunk_size = chunk_size

    def get_batch(self, split: str) -> Tuple[Tensor, Tensor]:
        """
        Return a batch of the dataset.
        :param split: train, val, test
        :return: Tuple[Tensor, Tensor] x, y dims [batch, context_len]
        """
        if split not in self.split_dataset_map.keys():
            raise ValueError("Options are 'train' or 'val'. Try again!")
        # select some random indices for batch starting points

        indices = torch.randint(
            high=len(self.split_dataset_map[split])
            - (self.context_len + self.chunk_size),
            size=(self.batch_size,),
        )
        x = torch.stack(
            [
                self.split_dataset_map[split][idx : idx + self.context_len]
                for idx in indices
            ]
        )
        slices = list()
        for i in range(1, self.chunk_size + 1):
            slices.append(
                torch.stack(
                    [
                        self.split_dataset_map[split][
                            idx + i : idx + self.context_len + i
                        ]
                        for idx in indices
                    ]
                )
            )
        y = torch.stack(slices, axis=-1).squeeze()  # type: ignore
        return x, y


if __name__ == "__main__":
    data_filename = os.path.join(project_base_dir, "data/input.txt")
    tokenizer = IndexTokenizer()
    context_len = 8
    chunk_size = 2
    batch_size = 4
    val_proportion = 0.1

    base_dataset = ShakespeareDataset(
        filename=data_filename,
        tokenizer=tokenizer,
        context_size=context_len,
        chunk_size=chunk_size,
    )

    # subsets for train/test
    train, test = partition_dataset(
        base_dataset,
        test_proportion=val_proportion,
        context_size=context_len,
        chunk_size=chunk_size,
    )

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    dataset = BasicShakespeareDataset(
        filename=data_filename,
        tokenizer=tokenizer,
        context_size=context_len,
        batch_size=batch_size,
        val_proportion=val_proportion,
        chunk_size=chunk_size,
    )
    x, y = dataset.get_batch(split="train")

    x_dl, y_dl = next(iter(train_dataloader))

    print(f"x shape: {x.shape}\n" f"x      : {x}")
    print(f"y shape: {y.shape}\n" f"y      : {y}")

    # visualize the context target matching.
    for b in range(batch_size):
        for t in range(context_len):
            context = x[b][: t + 1]
            target = y[b][t : t + chunk_size]
            print(f"Context: {context}, target: {target}")
