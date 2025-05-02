import torch
import torch.nn as nn
from monai.losses import DiceLoss, TverskyLoss

# Define individual loss components
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss(sigmoid=True)
tversky_loss = TverskyLoss(sigmoid=False, alpha=0.7, beta=0.3, reduction="mean")

def bce_dice_loss(pred, target):
    """
    Combined BCE + Dice Loss.
    Args:
        pred (Tensor): Model outputs (logits).
        target (Tensor): Ground truth masks.
    Returns:
        Tensor: Combined loss value.
    """
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    return 0.3 * bce + 0.7 * dice 

def dice_only_loss(pred, target):
    """
    Dice Loss only.
    Args:
        pred (Tensor): Model outputs (logits).
        target (Tensor): Ground truth masks.
    Returns:
        Tensor: Dice loss value.
    """
    return dice_loss(pred, target)

def tversky_only_loss(pred, target):
    """
    Tversky Loss only.
    Args:
        pred (Tensor): Model outputs (logits).
        target (Tensor): Ground truth masks.
    Returns:
        Tensor: Tversky loss value.
    """
    pred_sigmoid = torch.sigmoid(pred).clamp(min=1e-5, max=1 - 1e-5)
    target = target.clamp(0.0, 1.0)
    return tversky_loss(pred_sigmoid, target)

# Optional: dictionary to select loss by name
loss_functions = {
    "bce_dice": bce_dice_loss,
    "dice": dice_only_loss,
    "tversky": tversky_only_loss
}
