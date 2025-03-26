import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoiseLayer(nn.Module):
    """
    Noise simulation layer to increase robustness against various transformations.
    Simulates real-world noise and processing that might occur during transmission.
    """
    
    def __init__(self, noise_types=None, noise_params=None):
        """
        Initialize the noise layer
        
        Args:
            noise_types (list, optional): List of noise types to include
            noise_params (dict, optional): Parameters for each noise type
        """
        super(NoiseLayer, self).__init__()
        
        # Default noise types
        self.noise_types = noise_types or ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']
        
        # Default noise parameters
        self.noise_params = {
            'dropout': {'prob': 0.5},
            'jpeg': {'quality_factor': 50},
            'gaussian': {'std': 0.05},
            'blur': {'kernel_size': 5, 'sigma': 1.0},
            'salt_pepper': {'density': 0.1}
        }
        
        # Update with provided parameters
        if noise_params:
            for noise_type, params in noise_params.items():
                if noise_type in self.noise_params:
                    self.noise_params[noise_type].update(params)
        
    def forward(self, x, noise_type=None, noise_params=None):
        """
        Apply noise to the input tensor
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            noise_type (str, optional): Specific noise type to apply
                                       If None, a random type is selected
            noise_params (dict, optional): Override default noise parameters
            
        Returns:
            torch.Tensor: Noisy tensor with same shape as input
        """
        # If no specific noise type is specified, randomly choose one
        if noise_type is None:
            noise_type = np.random.choice(self.noise_types)
        
        # Get noise parameters
        params = self.noise_params.get(noise_type, {}).copy()
        if noise_params:
            params.update(noise_params)
        
        # Apply the selected noise
        if noise_type == 'dropout':
            return self.dropout(x, **params)
        elif noise_type == 'jpeg':
            return self.jpeg_compression(x, **params)
        elif noise_type == 'gaussian':
            return self.gaussian_noise(x, **params)
        elif noise_type == 'blur':
            return self.gaussian_blur(x, **params)
        elif noise_type == 'salt_pepper':
            return self.salt_pepper_noise(x, **params)
        elif noise_type == 'identity':
            return x  # No noise (identity)
        else:
            print(f"Warning: Unknown noise type '{noise_type}', returning input unchanged")
            return x
    
    def dropout(self, x, prob=0.5):
        """
        Random pixel dropout (zeroing out random pixels)
        
        Args:
            x (torch.Tensor): Input tensor
            prob (float): Dropout probability
            
        Returns:
            torch.Tensor: Tensor with random pixels dropped out
        """
        mask = torch.rand_like(x, device=x.device) > prob
        return x * mask
    
    def jpeg_compression(self, x, quality_factor=50):
        """
        Simulate JPEG compression artifacts
        
        Args:
            x (torch.Tensor): Input tensor
            quality_factor (int): JPEG quality (0-100)
            
        Returns:
            torch.Tensor: Tensor with simulated JPEG artifacts
        """
        # This is a simulation, as true JPEG compression is not differentiable
        
        # DCT block size (8x8 for JPEG)
        block_size = 8
        
        # Ensure batch processing
        batch_size, channels, height, width = x.size()
        
        # Convert to YCbCr-like (simplified for grayscale)
        y = x  # For grayscale, Y channel is the same as input
        
        # Pad if necessary to make dimensions multiple of block_size
        pad_h = (block_size - (height % block_size)) % block_size
        pad_w = (block_size - (width % block_size)) % block_size
        
        if pad_h > 0 or pad_w > 0:
            y = F.pad(y, (0, pad_w, 0, pad_h))
        
        # Get padded dimensions    
        _, _, padded_h, padded_w = y.size()
        
        # Create blocks of 8x8
        y_blocks = y.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        y_blocks = y_blocks.contiguous().view(-1, block_size, block_size)
        
        # Create DCT basis matrices
        # Note: This is a simplified DCT approximation
        dct_matrix = torch.zeros(block_size, block_size, device=x.device)
        for i in range(block_size):
            for j in range(block_size):
                if i == 0:
                    dct_matrix[i, j] = 1.0 / np.sqrt(block_size)
                else:
                    dct_matrix[i, j] = np.sqrt(2.0 / block_size) * np.cos(np.pi * (2 * j + 1) * i / (2 * block_size))
        
        # Compute approximate DCT by matrix multiplication
        # True DCT would be: D * block * D^T
        dct_blocks = torch.matmul(torch.matmul(dct_matrix, y_blocks), dct_matrix.t())
        
        # Simulate quantization based on quality factor
        # Create a simple quantization matrix that increases with frequency
        quant_matrix = torch.ones(block_size, block_size, device=x.device)
        for i in range(block_size):
            for j in range(block_size):
                quant_matrix[i, j] = 1 + (i + j) * (101 - quality_factor) / 25.0
        
        # Quantize DCT coefficients
        quantized_blocks = torch.round(dct_blocks / quant_matrix)
        dequantized_blocks = quantized_blocks * quant_matrix
        
        # Inverse DCT approximation
        y_compressed_blocks = torch.matmul(torch.matmul(dct_matrix.t(), dequantized_blocks), dct_matrix)
        
        # Reshape blocks back to image
        num_blocks = y_compressed_blocks.size(0)
        h_blocks = padded_h // block_size
        w_blocks = padded_w // block_size
        
        y_compressed = y_compressed_blocks.view(batch_size, channels, h_blocks, w_blocks, block_size, block_size)
        y_compressed = y_compressed.permute(0, 1, 2, 4, 3, 5).contiguous()
        y_compressed = y_compressed.view(batch_size, channels, padded_h, padded_w)
        
        # Remove padding if necessary
        if pad_h > 0 or pad_w > 0:
            y_compressed = y_compressed[:, :, :height, :width]
        
        # Ensure output is clamped to valid range
        return torch.clamp(y_compressed, 0, 1)
    
    def gaussian_noise(self, x, std=0.05):
        """
        Add Gaussian noise
        
        Args:
            x (torch.Tensor): Input tensor
            std (float): Standard deviation of the noise
            
        Returns:
            torch.Tensor: Noisy tensor
        """
        noise = torch.randn_like(x, device=x.device) * std
        noisy = x + noise
        return torch.clamp(noisy, 0, 1)
    
    def gaussian_blur(self, x, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian blur
        
        Args:
            x (torch.Tensor): Input tensor
            kernel_size (int): Size of Gaussian kernel
            sigma (float): Standard deviation of Gaussian kernel
            
        Returns:
            torch.Tensor: Blurred tensor
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create Gaussian kernel
        channels = x.shape[1]
        
        # Create 1D Gaussian kernel
        half_size = kernel_size // 2
        kernel_1d = torch.exp(-torch.arange(-half_size, half_size+1, dtype=torch.float, device=x.device) ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
        
        # Apply padding
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Apply convolution with the Gaussian kernel
        blurred = F.conv2d(x_padded, kernel_2d, groups=channels)
        
        return torch.clamp(blurred, 0, 1)
    
    def salt_pepper_noise(self, x, density=0.1):
        """
        Add salt and pepper noise
        
        Args:
            x (torch.Tensor): Input tensor
            density (float): Noise density
            
        Returns:
            torch.Tensor: Noisy tensor
        """
        noise = torch.rand_like(x, device=x.device)
        
        # Salt (white) noise
        salt = (noise < density/2).float()
        
        # Pepper (black) noise
        pepper = (noise > 1 - density/2).float()
        
        # Apply salt and pepper noise
        noisy = x.clone()
        noisy[salt > 0] = 1.0
        noisy[pepper > 0] = 0.0
        
        return noisy
    
    def resize(self, x, scale_factor=0.5):
        """
        Resize image (downsample and upsample)
        
        Args:
            x (torch.Tensor): Input tensor
            scale_factor (float): Scale factor for resize
            
        Returns:
            torch.Tensor: Resized tensor
        """
        # Downsample
        downsampled = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', 
                                   align_corners=False)
        
        # Upsample back to original size
        upsampled = F.interpolate(downsampled, size=(x.size(2), x.size(3)), 
                                 mode='bilinear', align_corners=False)
        
        return upsampled
    
    def combined_noise(self, x, noise_count=2):
        """
        Apply multiple noise types in sequence
        
        Args:
            x (torch.Tensor): Input tensor
            noise_count (int): Number of noise types to apply
            
        Returns:
            torch.Tensor: Noisy tensor
        """
        # Randomly select noise_count noise types
        selected_noise = np.random.choice(self.noise_types, size=noise_count, replace=False)
        
        # Apply each noise type in sequence
        noisy = x
        for noise_type in selected_noise:
            noisy = self.forward(noisy, noise_type)
            
        return noisy