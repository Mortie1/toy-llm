import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate fields in the dataset items.
    Converts individual items into a batch.

    All windows have the same length (block_size), so no padding
    is needed.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    return {
        key: torch.stack([elem[key] for elem in dataset_items])
        for key in dataset_items[0]
    }
