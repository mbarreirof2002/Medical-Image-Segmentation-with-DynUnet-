from monai.data import CacheDataset, DataLoader
from transforms import get_train_transforms, get_val_transforms

def get_dataloaders(train_files, val_files, batch_size=2, num_workers=4, use_strong_augmentation=True):
    """
    Creates the training and validation DataLoaders with optional aggressive augmentation.

    Args:
        train_files (list of dict): Training data ("image" and "label" keys).
        val_files (list of dict): Validation data.
        batch_size (int): Training batch size.
        num_workers (int): Number of data loader workers.
        use_strong_augmentation (bool): Whether to use aggressive augmentations.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(use_strong_augmentation=use_strong_augmentation),
        cache_rate=1.0,
        num_workers=num_workers
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_rate=1.0,
        num_workers=num_workers
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
