import torch
from monai.metrics import DiceMetric

# MONAI Dice metric instance for batch-wise evaluation
dice_metric = DiceMetric(include_background=False, reduction="mean")

def compute_dice(pred, target):
    """
    Computes the Dice coefficient between predicted and target segmentation masks.
    Args:
        pred (Tensor): Predicted binary segmentation mask (thresholded).
        target (Tensor): Ground truth binary segmentation mask.
    Returns:
        float: Dice coefficient value.
    """
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return 2.0 * intersection / (union + 1e-8)  # Add epsilon to avoid division by zero
