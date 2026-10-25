"""Settings taken from the paper (Sections IV-F and V-A)."""

SEED = 42

# ---------------------------------------------------------------- data
CROP_SIZE = 320          # images are centre-cropped to 320 x 320 (fastMRI convention)
ACS_LINES = 24           # fully sampled centre of k-space (auto-calibration signal)
DEFAULT_ACCELERATION = 4
ACCELERATIONS = (2, 4, 6, 8)                 # Table III
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)         # train / val / test by volume

# ---------------------------------------------------------------- training (Sec. IV-F)
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LR_PLATEAU_FACTOR = 0.5   # "learning rate was halved ..."
LR_PLATEAU_PATIENCE = 10  # "... every 10 epochs without validation improvement"

# Combined loss L = alpha * L2 + beta * L_SSIM + gamma * L_perceptual (Eq. 10)
LOSS_WEIGHTS = {"alpha": 0.6, "beta": 0.3, "gamma": 0.1}

# Augmentation: random flips, 90-degree rotations, additive Gaussian noise
AUG_FLIP_PROB = 0.5
AUG_ROT90_PROB = 0.5
AUG_NOISE_STD = 0.01      # relative to the k-space magnitude scale

# ---------------------------------------------------------------- proposed model (Sec. IV-D)
UNET_DC = {
    "iterations": 10,       # K unrolled iterations
    "levels": 4,            # encoder / decoder levels
    "base_channels": 24,     # 4.4M parameters (paper: 4.2M)
    "initial_lambda": 1.0,  # learned DC weight, kept positive with softplus
}

MODELS = ["Cascade_Net", "KIKI_Net", "UNet", "PD_Net", "UNet_DC"]
CLASSICAL = ["Zero_Filled", "CS_TV", "SENSE", "GRAPPA"]
