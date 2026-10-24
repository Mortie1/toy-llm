import logging
from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

logger = logging.getLogger(__name__)


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_leaf_datasets(dataset):
    """
    Unwrap ConcatDataset into the list of underlying datasets.

    Args:
        dataset (Dataset): dataset, possibly a ConcatDataset.
    Returns:
        leaves (list[Dataset]): datasets that are not ConcatDataset.
    """
    if isinstance(dataset, ConcatDataset):
        return [leaf for d in dataset.datasets for leaf in get_leaf_datasets(d)]
    return [dataset]


def fit_tokenizer(tokenizer, dataset, vocab_size, min_count=None, save_counts=True):
    """
    Fit the tokenizer on raw texts of the dataset and save it.

    Args:
        tokenizer (BaseTokenizer): tokenizer to fit.
        dataset (Dataset): dataset (usually the train partition) with
            iter_texts method, possibly a ConcatDataset of such datasets.
        vocab_size (int): target vocabulary size.
        min_count (int | None): if not None, drop chunks rarer than this
            before fitting.
        save_counts (bool): if True, save chunk frequencies with the
            tokenizer to continue fitting later.
    """
    logger.info(f"Tokenizer {tokenizer.path} not found, fitting it...")
    for leaf in get_leaf_datasets(dataset):
        texts = tqdm(leaf.iter_texts(), desc=f"Counting chunks ({leaf.part})", unit="doc")
        for text in texts:
            tokenizer.update(text)

    if min_count is not None:
        tokenizer.prune(min_count)

    tokenizer.fit(vocab_size)
    tokenizer.save(with_counts=save_counts)
    logger.info(
        f"Tokenizer saved to {tokenizer.path}, vocab size: {tokenizer.vocab_size}"
    )


def prepare_dataset(dataset, tokenizer, force=False):
    """
    Tokenize the dataset (or every part of ConcatDataset).

    ConcatDataset computes its length at init, when the parts are not
    tokenized yet, so it is recreated after the preparation.

    Args:
        dataset (Dataset): BaseTextDataset or ConcatDataset of them.
        tokenizer (BaseTokenizer): fitted tokenizer.
        force (bool): if True, ignore cached tokens.
    Returns:
        dataset (Dataset): prepared dataset.
    """
    if isinstance(dataset, ConcatDataset):
        return ConcatDataset(
            [prepare_dataset(d, tokenizer, force) for d in dataset.datasets]
        )
    dataset.prepare(tokenizer, force=force)
    return dataset


def get_dataloaders(config, device, tokenizer, allow_tokenizer_fit=False):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates batch transforms.

    If the tokenizer is not fitted and allow_tokenizer_fit is True, it
    is fitted on the train partition before tokenization.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
        tokenizer (BaseTokenizer): tokenizer for the datasets.
        allow_tokenizer_fit (bool): if True, fit the tokenizer when it
            is not fitted. Must be False for inference, so a wrong
            tokenizer path is not silently replaced by a new tokenizer.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init, raw data is downloaded here
    datasets = instantiate(config.datasets)

    just_fitted = False
    if not tokenizer.is_fitted:
        assert allow_tokenizer_fit, f"Tokenizer {tokenizer.path} not found"
        assert "train" in datasets, "Tokenizer is fitted on the train partition"
        fit_tokenizer(tokenizer, datasets["train"], **config.tokenizer_fit)
        just_fitted = True

    # cached tokens were made by the old tokenizer if it was refitted
    datasets = {
        part: prepare_dataset(dataset, tokenizer, force=just_fitted)
        for part, dataset in datasets.items()
    }

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
