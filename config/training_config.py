"""
Configuration parameters for training
"""

# Basic training parameters
TRAINING_CONFIG = {
    'batch_size': 8,                     # Batch size for training
    'epochs': 100,                       # Number of training epochs
    'val_split': 0.2,                    # Validation split ratio
    'num_workers': 4                     # Data loader worker threads
}

# Learning rates
LEARNING_RATES = {
    'feature_analyzer': 0.0001,          # Learning rate for feature analyzer
    'encoder': 0.0001,                   # Learning rate for encoder
    'decoder': 0.0001,                   # Learning rate for decoder
    'discriminator': 0.0001              # Learning rate for discriminator
}

# Loss weights - Initial phase (message recovery focus)
INITIAL_LOSS_WEIGHTS = {
    'lambda_message': 20.0,              # Weight for message reconstruction
    'lambda_image': 1.0,                 # Weight for image distortion
    'lambda_perceptual': 0.3,            # Weight for perceptual loss
    'lambda_adv': 0.1                    # Weight for adversarial loss
}

# Loss weights - Middle phase (balanced)
MIDDLE_LOSS_WEIGHTS = {
    'lambda_message': 10.0,              # Weight for message reconstruction
    'lambda_image': 2.0,                 # Weight for image distortion
    'lambda_perceptual': 0.5,            # Weight for perceptual loss
    'lambda_adv': 0.1                    # Weight for adversarial loss
}

# Loss weights - Final phase (image quality focus)
FINAL_LOSS_WEIGHTS = {
    'lambda_message': 5.0,               # Weight for message reconstruction
    'lambda_image': 3.0,                 # Weight for image distortion
    'lambda_perceptual': 1.0,            # Weight for perceptual loss
    'lambda_adv': 0.1                    # Weight for adversarial loss
}

# Adaptive training settings
ADAPTIVE_TRAINING = {
    'use_adaptive_weights': True,        # Whether to use adaptive loss weights
    'initial_phase_epochs': 30,          # Epochs for initial phase
    'middle_phase_epochs': 30,           # Epochs for middle phase
    'use_scheduler': True,               # Whether to use learning rate scheduler
    'scheduler_patience': 5,             # Epochs to wait before reducing LR
    'scheduler_factor': 0.5              # Factor to reduce LR by
}

# GAN training settings
GAN_CONFIG = {
    'label_smoothing': 0.1               # Label smoothing for discriminator
}

# Logging and checkpoints
LOGGING_CONFIG = {
    'log_dir': './logs',                 # Directory for tensorboard logs
    'model_save_path': './models/weights',  # Directory to save model weights
    'log_interval': 10,                  # Log every N batches
    'save_interval': 5                   # Save models every N epochs
}

# Data processing
DATA_CONFIG = {
    'resolutions': [(256, 256), (512, 512), (1024, 1024)],  # Supported resolutions
    'default_resolution': (512, 512),    # Default resolution
    'message_length': 4096,              # Length of binary message in bits
    'use_error_correction': True,        # Whether to use error correction
    'ecc_bytes': 16                      # Error correction bytes per chunk
}