import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DenseLayer(nn.Module):
    """A single dense layer with skip connection"""
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """Block of densely connected layers"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        
        self.out_channels = current_channels
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SelfAttention(nn.Module):
    """Memory-efficient self-attention module for focusing on important features during decoding"""
    def __init__(self, in_channels, downsample_factor=4):
        super(SelfAttention, self).__init__()
        self.downsample_factor = downsample_factor
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight for attention
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Downsample feature map for efficient attention computation
        if self.downsample_factor > 1:
            h, w = height // self.downsample_factor, width // self.downsample_factor
            x_down = F.adaptive_avg_pool2d(x, (h, w))
            
            # Create key and value projections on downsampled feature map
            proj_key = self.key(x_down).view(batch_size, -1, h * w)
            proj_value = self.value(x_down).view(batch_size, -1, h * w)
            
            # Create query projection on original resolution
            proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
            
            # Use chunked matrix multiplication to save memory
            # Process in chunks along spatial dimension to avoid OOM
            chunk_size = min(1024, height * width)
            num_chunks = (height * width + chunk_size - 1) // chunk_size
            output_chunks = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, height * width)
                
                # Get query chunk
                query_chunk = proj_query[:, start_idx:end_idx, :]
                
                # Calculate attention map for chunk
                energy = torch.bmm(query_chunk, proj_key)
                attention = F.softmax(energy, dim=-1)
                
                # Apply attention to values
                chunk_output = torch.bmm(attention, proj_value.permute(0, 2, 1))
                output_chunks.append(chunk_output)
            
            # Concatenate chunks
            out = torch.cat(output_chunks, dim=1)
        else:
            # Standard attention if no downsampling
            proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
            proj_key = self.key(x).view(batch_size, -1, height * width)
            proj_value = self.value(x).view(batch_size, -1, height * width)
            
            # Calculate attention map
            energy = torch.bmm(proj_query, proj_key)
            attention = F.softmax(energy, dim=-1)
            
            # Apply attention to values
            out = torch.bmm(attention, proj_value.permute(0, 2, 1))
        
        # Reshape back and add residual connection with learnable weight
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        
        return out

class DCTLayer(nn.Module):
    """
    DCT transformation layer (same as in encoder)
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
        """Apply block-wise DCT or inverse DCT transformation"""
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
            # Apply DCT: matrix multiplication on both sides (Y = A * X * A^T)
            temp = torch.matmul(self.dct_mat, blocks_reshaped)
            dct_blocks = torch.matmul(temp, self.idct_mat)
            
            # Reshape back to proper format
            dct_blocks = dct_blocks.view(batch_size, channels, num_blocks, self.block_size, self.block_size)
            h_blocks = height // self.block_size
            w_blocks = width // self.block_size
            dct_coef = dct_blocks.view(batch_size, channels, h_blocks, w_blocks, self.block_size, self.block_size)
            dct_coef = dct_coef.permute(0, 1, 2, 4, 3, 5).contiguous()
            dct_coef = dct_coef.view(batch_size, channels, height, width)
            
            return dct_coef
            
        else:
            # Apply inverse DCT: matrix multiplication on both sides (X = A^T * Y * A)
            temp = torch.matmul(self.idct_mat, blocks_reshaped)
            idct_blocks = torch.matmul(temp, self.dct_mat)
            
            # Reshape back to image format
            idct_blocks = idct_blocks.view(batch_size, channels, num_blocks, self.block_size, self.block_size)
            h_blocks = height // self.block_size
            w_blocks = width // self.block_size
            output = idct_blocks.view(batch_size, channels, h_blocks, w_blocks, self.block_size, self.block_size)
            output = output.permute(0, 1, 2, 4, 3, 5).contiguous()
            output = output.view(batch_size, channels, height, width)
            
            # Remove padding if necessary
            if pad_h > 0 or pad_w > 0:
                output = output[:, :, :height-pad_h, :width-pad_w]
                
            return output

class SteganographyDecoder(nn.Module):
    """
    Enhanced steganography decoder with self-attention and confidence estimation
    for improved extraction accuracy.
    """
    
    def __init__(self, image_channels=1, message_length=4096, growth_rate=32, num_dense_layers=6, with_dct=True):
        """
        Initialize the steganography decoder
        
        Args:
            image_channels (int): Number of image channels (1 for grayscale)
            message_length (int): Length of binary message to extract
            growth_rate (int): Growth rate for dense blocks
            num_dense_layers (int): Number of layers in dense blocks
            with_dct (bool): Whether to use DCT transformation
        """
        super(SteganographyDecoder, self).__init__()
        
        self.message_length = message_length
        self.with_dct = with_dct
        
        # DCT transformation layer if used
        if with_dct:
            self.dct = DCTLayer(block_size=8)
        
        # Initial convolution to extract features
        self.initial_conv = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction with dense blocks
        self.dense_block1 = DenseBlock(64, growth_rate, num_dense_layers)
        current_channels = 64 + growth_rate * num_dense_layers
        
        # Transition layer with downsampling
        self.transition1 = nn.Sequential(
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, 128, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Memory-efficient self-attention for focusing on embedded data regions
        # Use higher downsample factor for larger images
        self.attention = SelfAttention(128, downsample_factor=8)
        
        # Second dense block
        self.dense_block2 = DenseBlock(128, growth_rate, num_dense_layers)
        current_channels = 128 + growth_rate * num_dense_layers
        
        # Transition layer with downsampling
        self.transition2 = nn.Sequential(
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, 256, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten_size = 256 * 8 * 8
        
        # Fully connected layers to decode the message
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, message_length * 2)  # *2 for values and confidence
        )
        
        # Initialize weights
        self._initialize_weights()
        
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
    
    def forward(self, stego_image):
        """
        Extract message from stego image
        
        Args:
            stego_image (torch.Tensor): Stego image tensor [B, C, H, W]
            
        Returns:
            tuple: (message, confidence) tensors [B, L]
        """
        batch_size = stego_image.size(0)
        
        # Apply DCT transform if configured
        if self.with_dct:
            x = self.dct(stego_image)
        else:
            x = stego_image
        
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # First dense block
        x = self.dense_block1(x)
        x = self.transition1(x)
        
        # Apply self-attention to focus on relevant regions
        x = self.attention(x)
        
        # Second dense block
        x = self.dense_block2(x)
        x = self.transition2(x)
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Decode message with confidence
        x = self.fc(x)
        
        # Split output into message bits and confidence scores
        x = x.view(batch_size, 2, self.message_length)
        
        message = torch.sigmoid(x[:, 0, :])  # Binary message prediction
        confidence = torch.sigmoid(x[:, 1, :])  # Confidence scores in [0,1]
        
        return message, confidence


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder that processes the stego image at multiple resolutions
    for more robust extraction.
    """
    
    def __init__(self, image_channels=1, message_length=4096):
        """
        Initialize the multi-scale decoder
        
        Args:
            image_channels (int): Number of image channels (1 for grayscale)
            message_length (int): Length of binary message to extract
        """
        super(MultiScaleDecoder, self).__init__()
        
        # Create decoder branches for different scales
        # For memory efficiency with large images, use only mid and small scale decoders
        # Use fewer dense layers for memory efficiency
        
        self.mid_scale_decoder = SteganographyDecoder(
            image_channels=image_channels,
            message_length=message_length,
            growth_rate=24,
            num_dense_layers=4,
            with_dct=True
        )
        
        self.small_scale_decoder = SteganographyDecoder(
            image_channels=image_channels,
            message_length=message_length,
            growth_rate=16,
            num_dense_layers=3,
            with_dct=False  # Smaller scale might lose DCT precision
        )
        
        # For 512x512 or larger images, skip the full-scale decoder to save memory
        self.use_full_scale = False
        
        
        # Fusion layer to combine predictions from different scales
        self.fusion = nn.Sequential(
            nn.Linear(message_length * 3, message_length * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(message_length * 2, message_length * 2)
        )
    
    def forward(self, stego_image):
        """
        Extract message from stego image at multiple scales
        
        Args:
            stego_image (torch.Tensor): Stego image tensor [B, C, H, W]
            
        Returns:
            tuple: (message, confidence) tensors [B, L]
        """
        batch_size = stego_image.size(0)
        h, w = stego_image.size()[2:]
        
        # Check image size and dynamically determine if we should use full-scale processing
        # For large images (512x512 or bigger), skip full-scale to save memory
        self.use_full_scale = h < 512 and w < 512
        
        # Process at mid scale (60-75%)
        mid_scale_factor = 0.6 if h >= 512 else 0.75
        mid_size = (int(h * mid_scale_factor), int(w * mid_scale_factor))
        mid_scale_img = F.interpolate(stego_image, size=mid_size, mode='bilinear', align_corners=False)
        mid_msg, mid_conf = self.mid_scale_decoder(mid_scale_img)
        
        # Process at small scale (40-50%)
        small_scale_factor = 0.4 if h >= 512 else 0.5
        small_size = (int(h * small_scale_factor), int(w * small_scale_factor))
        small_scale_img = F.interpolate(stego_image, size=small_size, mode='bilinear', align_corners=False)
        small_msg, small_conf = self.small_scale_decoder(small_scale_img)
        
        if self.use_full_scale:
            # Process at full scale only for smaller images
            full_msg, full_conf = self.full_scale_decoder(stego_image)
            # Concatenate all predictions for fusion
            combined = torch.cat([full_msg, mid_msg, small_msg], dim=1)
        else:
            # For larger images, only use mid and small scale predictions
            # Duplicate mid_msg to maintain the expected input dimension for fusion
            combined = torch.cat([mid_msg, mid_msg, small_msg], dim=1)
        
        # Fuse predictions
        fused = self.fusion(combined)
        fused = fused.view(batch_size, 2, -1)
        
        # Final message and confidence
        message = torch.sigmoid(fused[:, 0, :])
        confidence = torch.sigmoid(fused[:, 1, :])
        
        return message, confidence