import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GrayscaleVGG(nn.Module):
    """
    VGG-based perceptual loss network adapted for grayscale images,
    used to compare structural features between original and stego images.
    """
    
    def __init__(self, layers=None):
        """
        Initialize the perceptual loss network
        
        Args:
            layers (list, optional): List of layer indices to extract features from.
                                    If None, uses [3, 8, 15, 22] (VGG16)
        """
        super(GrayscaleVGG, self).__init__()
        
        # Default layers if none provided
        self.layers = layers or [3, 8, 15, 22]  # Selected layers from VGG16
        
        # Load pretrained VGG16 with batch normalization
        vgg = models.vgg16_bn(weights='DEFAULT')
        
        # Create feature extractor up to the deepest requested layer
        max_layer = max(self.layers)
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer+1])
        
        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # First layer adaptation for grayscale
        # Replace the first conv layer to accept grayscale images (1 channel)
        # while keeping the original weights for the first channel
        original_conv = self.features[0]
        grayscale_conv = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )
        
        # Copy the first channel weights
        with torch.no_grad():
            grayscale_conv.weight.data = original_conv.weight.data[:, 0:1, :, :]
            if original_conv.bias is not None:
                grayscale_conv.bias.data = original_conv.bias.data
        
        # Replace the first layer
        self.features[0] = grayscale_conv
        
        # Set to evaluation mode
        self.features.eval()
        
    def forward(self, x):
        """
        Extract features from specified layers
        
        Args:
            x (torch.Tensor): Input image tensor [B, 1, H, W]
            
        Returns:
            list: Feature maps from specified layers
        """
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss based on feature differences from VGG network
    """
    
    def __init__(self, loss_type='l1', weight_factors=None):
        """
        Initialize the perceptual loss
        
        Args:
            loss_type (str): Loss function type ('l1', 'l2', 'smooth_l1')
            weight_factors (list, optional): Weights for each layer's contribution
        """
        super(PerceptualLoss, self).__init__()
        
        # Initialize feature extractor
        self.feature_extractor = GrayscaleVGG()
        self.loss_type = loss_type
        
        # Default weight factors for different layers (deeper layers get less weight)
        self.weight_factors = weight_factors or [1.0, 0.75, 0.5, 0.25]
        
        # Set evaluation mode
        self.eval()
        
    def forward(self, original, stego):
        """
        Calculate perceptual loss between original and stego images
        
        Args:
            original (torch.Tensor): Original image tensor [B, 1, H, W]
            stego (torch.Tensor): Stego image tensor [B, 1, H, W]
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        # Extract features
        original_features = self.feature_extractor(original)
        stego_features = self.feature_extractor(stego)
        
        # Calculate loss at each feature level
        loss = 0.0
        for i, (orig_feat, stego_feat) in enumerate(zip(original_features, stego_features)):
            # Choose loss function
            if self.loss_type == 'l1':
                layer_loss = F.l1_loss(orig_feat, stego_feat)
            elif self.loss_type == 'l2':
                layer_loss = F.mse_loss(orig_feat, stego_feat)
            elif self.loss_type == 'smooth_l1':
                layer_loss = F.smooth_l1_loss(orig_feat, stego_feat)
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
            # Apply weight factor for this layer
            loss += layer_loss * self.weight_factors[i]
        
        return loss


class StyleContentLoss(nn.Module):
    """
    Combined style and content perceptual loss based on VGG features,
    encouraging both global structure preservation and fine texture details.
    """
    
    def __init__(self, content_weight=1.0, style_weight=0.05):
        """
        Initialize the style and content loss
        
        Args:
            content_weight (float): Weight for content loss
            style_weight (float): Weight for style loss
        """
        super(StyleContentLoss, self).__init__()
        
        # Initialize feature extractor
        self.feature_extractor = GrayscaleVGG()
        
        # Weights for content and style components
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Set evaluation mode
        self.eval()
    
    def _gram_matrix(self, x):
        """Calculate Gram matrix for style loss"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, original, stego):
        """
        Calculate combined style and content loss
        
        Args:
            original (torch.Tensor): Original image tensor [B, 1, H, W]
            stego (torch.Tensor): Stego image tensor [B, 1, H, W]
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Extract features
        original_features = self.feature_extractor(original)
        stego_features = self.feature_extractor(stego)
        
        # Content loss (deeper layers for high-level content)
        content_loss = F.mse_loss(original_features[-1], stego_features[-1])
        
        # Style loss (from all layers for comprehensive style matching)
        style_loss = 0.0
        for i, (orig_feat, stego_feat) in enumerate(zip(original_features, stego_features)):
            # Calculate Gram matrices
            orig_gram = self._gram_matrix(orig_feat)
            stego_gram = self._gram_matrix(stego_feat)
            
            # Add to style loss
            style_loss += F.mse_loss(orig_gram, stego_gram)
        
        # Combine losses
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return total_loss