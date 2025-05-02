import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configuration
max_epochs = 300
learning_rate = 1e-3
validate_every = 5
fast_until_epoch = 100

# Validation settings
fast_roi_size = (96, 96, 96)
full_roi_size = (128, 128, 128)
fast_overlap = 0.1
full_overlap = 0.25
fast_sw_batch_size = 2
full_sw_batch_size = 1

# Path for saving the best model
model_path = "outputs/best_model.pth"