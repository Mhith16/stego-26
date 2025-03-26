"""
Configuration parameters for model architectures
"""

# Feature Analysis Network - DenseNet
FEATURE_DENSENET_CONFIG = {
    'in_channels': 1,                         # Grayscale X-ray images
    'model_type': 'densenet121',              # Base backbone model
    'pretrained': True,                       # Use pretrained weights
    'freeze_backbone': True                   # Freeze backbone weights
}

# Feature Analysis Network - U-Net
FEATURE_UNET_CONFIG = {
    'in_channels': 1,                         # Grayscale X-ray images
    'base_filters': 64,                       # Base filter count
    'pretrained_encoder': True                # Use pretrained encoder
}

# Encoder Network
ENCODER_CONFIG = {
    'image_channels': 1,                      # Grayscale X-ray images
    'embedding_strength': 0.02,               # Strength of embedding (0.01-0.03 recommended)
    'block_size': 8                           # DCT block size
}

# Decoder Network - Standard
DECODER_CONFIG = {
    'image_channels': 1,                      # Grayscale X-ray images
    'message_length': 4096,                   # Binary message length
    'growth_rate': 32,                        # Growth rate for dense blocks
    'num_dense_layers': 6,                    # Number of layers in dense blocks
    'with_dct': True                          # Use DCT transformation
}

# Decoder Network - Multi-scale
MULTI_SCALE_DECODER_CONFIG = {
    'image_channels': 1,                      # Grayscale X-ray images
    'message_length': 4096                    # Binary message length
}

# Discriminator Network
DISCRIMINATOR_CONFIG = {
    'image_channels': 1,                      # Grayscale X-ray images
    'base_filters': 64,                       # Base filter count
    'use_spectral_norm': True                 # Use spectral normalization
}

# Patch Discriminator Network
PATCH_DISCRIMINATOR_CONFIG = {
    'image_channels': 1,                      # Grayscale X-ray images
    'base_filters': 64,                       # Base filter count
    'n_layers': 3                             # Number of downsampling layers
}

# Noise Layer
NOISE_LAYER_CONFIG = {
    'noise_types': ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper'],
    'noise_params': {
        'dropout': {'prob': 0.5},              # Dropout probability
        'jpeg': {'quality_factor': 50},        # JPEG quality (0-100)
        'gaussian': {'std': 0.05},             # Gaussian noise std dev
        'blur': {'kernel_size': 5, 'sigma': 1.0},  # Gaussian blur parameters
        'salt_pepper': {'density': 0.1}        # Salt and pepper noise density
    }
}

# Perceptual Loss
PERCEPTUAL_LOSS_CONFIG = {
    'loss_type': 'l1',                        # Loss function type ('l1', 'l2', 'smooth_l1')
    'weight_factors': [1.0, 0.75, 0.5, 0.25]  # Layer weights (deeper layers get less weight)
}

# Error Correction
ERROR_CORRECTION_CONFIG = {
    'ecc_bytes': 16,                          # Error correction bytes per chunk
    'message_chunk_size': 255                 # Maximum chunk size (Reed-Solomon limitation)
}