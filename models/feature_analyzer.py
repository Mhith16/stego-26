import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureAnalysisDenseNet(nn.Module):
    """
    Feature analysis network based on pretrained DenseNet to identify optimal hiding regions
    in medical images. Uses transfer learning from models pretrained on medical or general datasets.
    """
    
    def __init__(self, model_type='densenet121', pretrained=True, in_channels=1, freeze_backbone=True):
        """
        Initialize the feature analysis network
        
        Args:
            model_type (str): Type of backbone model ('densenet121', 'resnet50', 'efficientnet_b0')
            pretrained (bool): Whether to use pretrained weights
            in_channels (int): Number of input channels (1 for grayscale)
            freeze_backbone (bool): Whether to freeze backbone weights
        """
        super(FeatureAnalysisDenseNet, self).__init__()
        
        self.model_type = model_type
        self.in_channels = in_channels
        
        # Load the appropriate backbone model
        if model_type == 'densenet121':
            self.backbone = models.densenet121(weights='DEFAULT' if pretrained else None)
            feature_dim = 1024  # DenseNet121 feature dimension
            
            # Modify first layer for grayscale if needed
            if in_channels != 3:
                self.backbone.features.conv0 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                
        elif model_type == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
            feature_dim = 2048  # ResNet50 feature dimension
            
            # Modify first layer for grayscale if needed
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                
        elif model_type == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            feature_dim = 1280  # EfficientNet-B0 feature dimension
            
            # Modify first layer for grayscale if needed
            if in_channels != 3:
                self.backbone.features[0][0] = nn.Conv2d(
                    in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
                )
                
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add feature projection layers to create importance map
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0,1] range for importance map
        )
        
        # Initialize projection weights
        self._initialize_weights(self.feature_projection)
        
    def _initialize_weights(self, module):
        """Initialize weights of the given module"""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _extract_features(self, x):
        """Extract features from the backbone model"""
        if self.model_type == 'densenet121':
            # Get features before the classifier
            features = self.backbone.features(x)
            return features
            
        elif self.model_type == 'resnet50':
            # Extract features from ResNet before the final pooling
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return x
            
        elif self.model_type == 'efficientnet_b0':
            # Extract features from EfficientNet
            return self.backbone.features(x)
    
    def forward(self, x):
        """
        Forward pass to generate importance map
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Importance map [B, 1, H, W] with values in [0,1]
        """
        # Handle grayscale to RGB conversion if model expects 3 channels but input is 1
        if self.in_channels == 1 and self.backbone.features[0].in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract high-level features
        features = self._extract_features(x)
        
        # Project features to importance map
        importance_map = self.feature_projection(features)
        
        # Upsample to match input resolution if needed
        if importance_map.size()[2:] != x.size()[2:]:
            importance_map = F.interpolate(
                importance_map, 
                size=x.size()[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        return importance_map


class FeatureAnalysisUNet(nn.Module):
    """
    U-Net based feature analysis network specifically designed for medical images,
    with optional pretrained encoder components.
    """
    
    def __init__(self, in_channels=1, base_filters=64, pretrained_encoder=True):
        """
        Initialize the U-Net based feature analyzer
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale)
            base_filters (int): Number of base filters (growth factor)
            pretrained_encoder (bool): Whether to use pretrained weights for encoder
        """
        super(FeatureAnalysisUNet, self).__init__()
        
        # Use ResNet18 as encoder for transfer learning
        if pretrained_encoder:
            resnet = models.resnet18(weights='DEFAULT')
            if in_channels != 3:
                # Modify first layer for grayscale
                self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
            else:
                self.conv1 = resnet.conv1
                
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            
            self.encoder1 = resnet.layer1  # 64 channels
            self.encoder2 = resnet.layer2  # 128 channels
            self.encoder3 = resnet.layer3  # 256 channels
            self.encoder4 = resnet.layer4  # 512 channels
        else:
            # Build encoder from scratch if no pretrained weights
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Simple convolutional blocks for encoder
            self.encoder1 = self._make_layer(64, 64, 2, stride=1)
            self.encoder2 = self._make_layer(64, 128, 2, stride=2)
            self.encoder3 = self._make_layer(128, 256, 2, stride=2)
            self.encoder4 = self._make_layer(256, 512, 2, stride=2)
        
        # Decoder (upsampling) path
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_layer(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_layer(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_layer(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self._make_decoder_layer(96, 32)  # 96 = 64 + 32 (skip connection)
        
        # Final convolution to generate importance map
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0,1] range
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Helper method to create convolutional layers"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def _make_decoder_layer(self, in_channels, out_channels):
        """Helper method to create decoder convolutional blocks"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass to generate importance map
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Importance map [B, 1, H, W] with values in [0,1]
        """
        # Initial convolution
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1_pool = self.maxpool(x1)
        
        # Encoder path
        x2 = self.encoder1(x1_pool)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        
        # Decoder path with skip connections
        x = self.upconv4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        
        # Final convolution
        importance_map = self.final_conv(x)
        
        # Ensure output is the same size as input
        if importance_map.size()[2:] != x.size()[2:]:
            importance_map = F.interpolate(
                importance_map, 
                size=x.size()[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        return importance_map