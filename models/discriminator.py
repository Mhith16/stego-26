import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    Discriminator network to distinguish between original and stego images.
    Uses spectral normalization for training stability.
    """
    
    def __init__(self, image_channels=1, base_filters=64, use_spectral_norm=True):
        """
        Initialize the discriminator
        
        Args:
            image_channels (int): Number of image channels (1 for grayscale)
            base_filters (int): Number of base filters
            use_spectral_norm (bool): Whether to use spectral normalization
        """
        super(Discriminator, self).__init__()
        
        # Apply spectral normalization if requested
        norm_layer = lambda x: nn.utils.spectral_norm(x) if use_spectral_norm else x
        
        # First layer - no normalization
        self.conv1 = nn.Conv2d(image_channels, base_filters, kernel_size=4, stride=2, padding=1)
        
        # Main convolutional blocks
        self.conv2 = norm_layer(nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        
        self.conv3 = norm_layer(nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(base_filters * 4)
        
        self.conv4 = norm_layer(nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=2, padding=1))
        self.bn4 = nn.BatchNorm2d(base_filters * 8)
        
        # Attention layer for better focus
        self.attention = SelfAttention(base_filters * 8)
        
        # Final layers
        self.conv5 = norm_layer(nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=4, stride=2, padding=1))
        self.bn5 = nn.BatchNorm2d(base_filters * 16)
        
        # Adaptive pool to handle multiple resolutions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected classification layer
        self.fc = nn.Sequential(
            nn.Linear(base_filters * 16, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass to predict if image is real or stego
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Probability of being a real image [B, 1]
        """
        # Initial conv without normalization
        x = F.leaky_relu(self.conv1(x), 0.2)
        
        # Main conv blocks
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Apply attention
        x = self.attention(x)
        
        # Final conv
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        # Global pooling and classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class SelfAttention(nn.Module):
    """Self-attention module for the discriminator to better identify subtle changes"""
    
    def __init__(self, in_channels):
        """
        Initialize the self-attention module
        
        Args:
            in_channels (int): Number of input channels
        """
        super(SelfAttention, self).__init__()
        
        # Layers for query, key, value
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """Apply self-attention mechanism"""
        batch_size, channels, height, width = x.size()
        
        # Create projections
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        # Reshape and apply scaling
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        
        return out


class PatchDiscriminator(nn.Module):
    """
    Patch-based discriminator that operates on smaller patches of the image
    for better detail discrimination.
    """
    
    def __init__(self, image_channels=1, base_filters=64, n_layers=3):
        """
        Initialize the patch discriminator
        
        Args:
            image_channels (int): Number of image channels (1 for grayscale)
            base_filters (int): Number of base filters
            n_layers (int): Number of downsampling layers
        """
        super(PatchDiscriminator, self).__init__()
        
        # First layer - no normalization
        sequence = [
            nn.Conv2d(image_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Build subsequent layers with increasing filters
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            sequence += [
                nn.Conv2d(base_filters * nf_mult_prev, base_filters * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Add one more layer with stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [
            nn.Conv2d(base_filters * nf_mult_prev, base_filters * nf_mult, 
                     kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Final output layer - real/fake map
        sequence += [
            nn.Conv2d(base_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ]
        
        # Create sequential model
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        """
        Forward pass to generate patch-wise predictions
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Patch-wise predictions [B, 1, H', W']
        """
        return self.model(x)