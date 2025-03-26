import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DCTLayer(nn.Module):
    """
    Custom layer for applying DCT transformation to image blocks
    for frequency domain embedding.
    """
    def __init__(self, block_size=8):
        super(DCTLayer, self).__init__()
        self.block_size = block_size
        
        # Pre-compute DCT transformation matrix
        self._build_dct_matrix(block_size)
        
    def _build_dct_matrix(self, n):
        """Build the DCT transformation matrix"""
        # Create transformation matrices for DCT and inverse DCT
        dct_mat = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i == 0:
                    dct_mat[i, j] = math.sqrt(1.0 / n) * math.cos(math.pi * (2 * j + 1) * i / (2 * n))
                else:
                    dct_mat[i, j] = math.sqrt(2.0 / n) * math.cos(math.pi * (2 * j + 1) * i / (2 * n))
        
        # Register as buffer so it moves to the right device with the model
        self.register_buffer('dct_mat', dct_mat)
        self.register_buffer('idct_mat', dct_mat.transpose(0, 1))
    
    def forward(self, x, inverse=False):
        """
        Apply block-wise DCT or inverse DCT transformation
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            inverse (bool): If True, apply inverse DCT, otherwise apply DCT
            
        Returns:
            torch.Tensor: DCT coefficients or spatial domain image
        """
        batch_size, channels, height, width = x.size()
        
        # Pad image if dimensions are not multiples of block_size
        pad_h = (self.block_size - height % self.block_size) % self.block_size
        pad_w = (self.block_size - width % self.block_size) % self.block_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        # Updated dimensions after padding
        _, _, height, width = x.size()
        
        # Reshape to block format
        blocks = x.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        blocks = blocks.contiguous().view(batch_size, channels, -1, self.block_size, self.block_size)
        num_blocks = blocks.size(2)
        
        # Reshape for matrix multiplication
        blocks_reshaped = blocks.view(-1, self.block_size, self.block_size)
        
        if not inverse:
            # Apply DCT: matrix multiplication on both sides
            # Y = A * X * A^T
            temp = torch.matmul(self.dct_mat, blocks_reshaped)
            dct_blocks = torch.matmul(temp, self.idct_mat)  # Using idct_mat as transpose
            
            # Reshape back
            dct_blocks = dct_blocks.view(batch_size, channels, num_blocks, self.block_size, self.block_size)
            
            # Convert to proper format for further processing
            h_blocks = height // self.block_size
            w_blocks = width // self.block_size
            
            dct_coef = dct_blocks.view(batch_size, channels, h_blocks, w_blocks, self.block_size, self.block_size)
            dct_coef = dct_coef.permute(0, 1, 2, 4, 3, 5).contiguous()
            dct_coef = dct_coef.view(batch_size, channels, height, width)
            
            return dct_coef
            
        else:
            # Apply inverse DCT: matrix multiplication on both sides
            # X = A^T * Y * A
            temp = torch.matmul(self.idct_mat, blocks_reshaped)
            idct_blocks = torch.matmul(temp, self.dct_mat)  # Using dct_mat as transpose
            
            # Reshape back
            idct_blocks = idct_blocks.view(batch_size, channels, num_blocks, self.block_size, self.block_size)
            
            # Convert blocks back to image
            h_blocks = height // self.block_size
            w_blocks = width // self.block_size
            
            output = idct_blocks.view(batch_size, channels, h_blocks, w_blocks, self.block_size, self.block_size)
            output = output.permute(0, 1, 2, 4, 3, 5).contiguous()
            output = output.view(batch_size, channels, height, width)
            
            # Remove padding if necessary
            if pad_h > 0 or pad_w > 0:
                output = output[:, :, :height-pad_h, :width-pad_w]
                
            return output


class SteganographyEncoder(nn.Module):
    """
    Enhanced DCT-based steganography encoder with adjustable embedding strength
    and medium frequency targeting to maximize PSNR.
    """
    
    def __init__(self, image_channels=1, embedding_strength=0.02, block_size=8):
        """
        Initialize the steganography encoder
        
        Args:
            image_channels (int): Number of image channels (1 for grayscale)
            embedding_strength (float): Strength of embedding (0.01-0.03 recommended)
            block_size (int): DCT block size (8 is standard for DCT)
        """
        super(SteganographyEncoder, self).__init__()
        
        self.image_channels = image_channels
        self.embedding_strength = embedding_strength
        self.block_size = block_size
        
        # DCT transformation layer
        self.dct = DCTLayer(block_size=block_size)
        
        # Feature processing for better message blending
        self.initial_conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Message embedding network
        self.message_processor = nn.Sequential(
            nn.Linear(4096, 2048),  # Assuming message is 4096 bits
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True)
        )
        
        # Frequency analysis network
        self.frequency_analyzer = nn.Sequential(
            nn.Conv2d(32 + 1, 64, kernel_size=3, padding=1),  # +1 for message channel
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, image_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Tanh for controlled modifications
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Generate fixed mask for middle frequency coefficients
        self._generate_frequency_mask()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def _generate_frequency_mask(self):
        """
        Generate a frequency mask that targets medium frequencies for embedding,
        which are less perceptible to human vision than low or high frequencies.
        """
        mask = torch.zeros(self.block_size, self.block_size)
        
        # Create a zigzag pattern indexing for DCT coefficients
        zigzag_indices = self._zigzag_indices(self.block_size)
        
        # Medium frequencies are better for hiding (skip first few coefficients which are low frequencies)
        # and avoid highest frequencies which are vulnerable to compression
        start_idx = 3  # Skip the first few low frequencies (DC and lowest AC)
        end_idx = self.block_size * self.block_size - self.block_size//2  # Avoid highest frequencies
        
        # Set medium frequencies to 1.0
        for idx in range(start_idx, end_idx):
            i, j = zigzag_indices[idx]
            mask[i, j] = 1.0
        
        # Create a smooth falloff for transition areas
        for idx in range(1, start_idx):
            i, j = zigzag_indices[idx]
            mask[i, j] = 0.5  # Reduced strength for low frequencies
            
        for idx in range(end_idx, self.block_size * self.block_size - 1):
            i, j = zigzag_indices[idx]
            mask[i, j] = 0.3  # Reduced strength for high frequencies
            
        # Register mask as a buffer
        self.register_buffer('frequency_mask', mask)
        
    def _zigzag_indices(self, n):
        """Generate zigzag traversal indices for an n×n matrix"""
        # Initialize empty array for zigzag pattern
        indices = []
        
        # Generate zigzag pattern
        for sum_idx in range(2*n-1):
            if sum_idx % 2 == 0:  # Even sum - go down-left
                for i in range(min(sum_idx+1, n)):
                    j = sum_idx - i
                    if j < n:
                        indices.append((i, j))
            else:  # Odd sum - go up-right
                for j in range(min(sum_idx+1, n)):
                    i = sum_idx - j
                    if i < n:
                        indices.append((i, j))
        
        return indices
    
    def forward(self, image, message, feature_weights=None):
        """
        Embed message into image using DCT domain steganography
        
        Args:
            image (torch.Tensor): Original image tensor [B, C, H, W]
            message (torch.Tensor): Binary message tensor [B, 1, L] or [B, L]
            feature_weights (torch.Tensor, optional): Feature importance map [B, 1, H, W]
            
        Returns:
            torch.Tensor: Stego image with embedded message [B, C, H, W]
        """
        batch_size, _, height, width = image.size()
        
        # Ensure message has the right dimensions [B, L]
        if message.dim() == 3:  # [B, 1, L]
            message = message.squeeze(1)
        
        # Store original dimensions for later use
        self.original_shape = (height, width)
        
        # Apply DCT transformation
        dct_image = self.dct(image)
        
        # Extract image features
        img_features = self.initial_conv(image)
        
        # Process message
        processed_msg = self.message_processor(message)
        
        # Create spatial message representation
        # Convert to 32x32 spatial grid
        spatial_size = 32
        spatial_msg = processed_msg.view(batch_size, 1, spatial_size, spatial_size)
        
        # Resize to match feature dimensions
        spatial_msg = F.interpolate(
            spatial_msg, 
            size=(img_features.size(2), img_features.size(3)),
            mode='bilinear', 
            align_corners=False
        )
        
        # Apply feature weights if provided
        if feature_weights is not None:
            # Ensure feature_weights has the right dimensions
            if feature_weights.size()[2:] != spatial_msg.size()[2:]:
                feature_weights = F.interpolate(
                    feature_weights, 
                    size=spatial_msg.size()[2:],
                    mode='bilinear', 
                    align_corners=False
                )
            # Weight the message embedding
            spatial_msg = spatial_msg * feature_weights
        
        # Combine image features and message
        combined_features = torch.cat([img_features, spatial_msg], dim=1)
        
        # Generate embedding modifications in frequency domain
        modifications = self.frequency_analyzer(combined_features)
        
        # Resize modifications to match DCT dimensions if needed
        if modifications.size()[2:] != dct_image.size()[2:]:
            modifications = F.interpolate(
                modifications, 
                size=dct_image.size()[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply frequency mask to target medium frequencies
        # Reshape for block-wise masking
        b, c, h, w = dct_image.size()
        h_blocks = h // self.block_size
        w_blocks = w // self.block_size
        
        # Apply frequency mask to each block
        modifications_blocks = modifications.view(b, c, h_blocks, self.block_size, w_blocks, self.block_size)
        
        # Broadcast frequency mask to all blocks
        masked_modifications = modifications_blocks * self.frequency_mask.view(1, 1, 1, self.block_size, 1, self.block_size)
        
        # Reshape back to original dimensions
        masked_modifications = masked_modifications.view(b, c, h, w)
        
        # Apply modifications to DCT coefficients with controlled strength
        modified_dct = dct_image + masked_modifications * self.embedding_strength
        
        # Apply inverse DCT to get back to spatial domain
        stego_image = self.dct(modified_dct, inverse=True)
        
        # Ensure output values are in valid range [0, 1]
        stego_image = torch.clamp(stego_image, 0, 1)
        
        return stego_image