import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg_perceptual import PerceptualLoss, StyleContentLoss

class SteganoLoss(nn.Module):
    """
    Combined loss function for steganography training with adjustable weights
    and dynamic priorities.
    """
    
    def __init__(self, lambda_message=20.0, lambda_image=1.0, lambda_adv=0.1, 
                 lambda_perceptual=0.5, use_perceptual=True, adaptive_weights=False):
        """
        Initialize the steganography loss function
        
        Args:
            lambda_message (float): Weight for message extraction loss
            lambda_image (float): Weight for image distortion loss
            lambda_adv (float): Weight for adversarial loss
            lambda_perceptual (float): Weight for perceptual loss
            use_perceptual (bool): Whether to use perceptual loss
            adaptive_weights (bool): Whether to adjust weights dynamically during training
        """
        super(SteganoLoss, self).__init__()
        
        self.lambda_message = lambda_message
        self.lambda_image = lambda_image
        self.lambda_adv = lambda_adv
        self.lambda_perceptual = lambda_perceptual
        
        self.use_perceptual = use_perceptual
        self.adaptive_weights = adaptive_weights
        
        # Message loss (binary cross-entropy)
        self.bce_loss = nn.BCELoss()
        
        # Image distortion loss (mean squared error)
        self.mse_loss = nn.MSELoss()
        
        # Perceptual loss for better image quality
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(loss_type='l1')
        
        # Track current epoch for adaptive weights
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for adaptive weight adjustment"""
        self.current_epoch = epoch
        
        if self.adaptive_weights:
            # Adjust weights based on training progress
            if epoch < 30:  # Early stages: focus on message embedding
                self.lambda_message = 20.0
                self.lambda_image = 1.0
                self.lambda_perceptual = 0.3
            elif epoch < 60:  # Middle stages: balanced approach
                self.lambda_message = 10.0
                self.lambda_image = 2.0
                self.lambda_perceptual = 0.5
            else:  # Later stages: focus on image quality
                self.lambda_message = 5.0
                self.lambda_image = 3.0
                self.lambda_perceptual = 1.0
    
    def forward(self, original_images, stego_images, 
               original_messages, decoded_messages, 
               confidence_scores=None, disc_real_pred=None, disc_fake_pred=None):
        """
        Calculate total loss for steganography training
        
        Args:
            original_images (torch.Tensor): Original input images
            stego_images (torch.Tensor): Generated stego images
            original_messages (torch.Tensor): Original binary messages
            decoded_messages (torch.Tensor): Decoded messages from stego images
            confidence_scores (torch.Tensor, optional): Confidence scores for decoded bits
            disc_real_pred (torch.Tensor, optional): Discriminator predictions for real images
            disc_fake_pred (torch.Tensor, optional): Discriminator predictions for stego images
            
        Returns:
            dict: Dictionary of individual losses and total loss
        """
        losses = {}
        
        # Message extraction loss with confidence weighting if available
        if confidence_scores is not None:
            # Weight message loss by confidence (lower confidence, lower loss weight)
            message_weights = confidence_scores.detach()  # Don't propagate gradients through weights
            message_loss = F.binary_cross_entropy(decoded_messages, original_messages, 
                                                 reduction='none') * message_weights
            message_loss = message_loss.mean()
        else:
            message_loss = self.bce_loss(decoded_messages, original_messages)
        
        losses['message'] = message_loss
        
        # Image distortion loss
        image_loss = self.mse_loss(stego_images, original_images)
        losses['image'] = image_loss
        
        # Perceptual loss for better visual quality
        if self.use_perceptual:
            p_loss = self.perceptual_loss(original_images, stego_images)
            losses['perceptual'] = p_loss
        else:
            p_loss = torch.tensor(0.0, device=image_loss.device)
            losses['perceptual'] = p_loss
        
        # Adversarial loss if discriminator outputs are provided
        if disc_fake_pred is not None and disc_real_pred is not None:
            adv_loss = self.bce_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            losses['adversarial'] = adv_loss
        else:
            adv_loss = torch.tensor(0.0, device=image_loss.device)
            losses['adversarial'] = adv_loss
        
        # Calculate total loss with weighting
        total_loss = (
            self.lambda_message * message_loss + 
            self.lambda_image * image_loss + 
            self.lambda_perceptual * p_loss + 
            self.lambda_adv * adv_loss
        )
        
        losses['total'] = total_loss
        
        return losses


class DiscriminatorLoss(nn.Module):
    """
    Loss function for the discriminator network.
    """
    
    def __init__(self, label_smoothing=0.1):
        """
        Initialize the discriminator loss
        
        Args:
            label_smoothing (float): Amount of label smoothing for GAN stability
        """
        super(DiscriminatorLoss, self).__init__()
        
        self.bce_loss = nn.BCELoss()
        self.label_smoothing = label_smoothing
    
    def forward(self, real_pred, fake_pred):
        """
        Calculate discriminator loss
        
        Args:
            real_pred (torch.Tensor): Discriminator predictions for real images
            fake_pred (torch.Tensor): Discriminator predictions for stego images
            
        Returns:
            dict: Dictionary of loss components and total loss
        """
        # Create target labels with smoothing for better stability
        real_target = torch.ones_like(real_pred) * (1.0 - self.label_smoothing)
        fake_target = torch.zeros_like(fake_pred)
        
        # Calculate losses
        real_loss = self.bce_loss(real_pred, real_target)
        fake_loss = self.bce_loss(fake_pred, fake_target)
        
        # Total discriminator loss
        total_loss = real_loss + fake_loss
        
        return {
            'real': real_loss,
            'fake': fake_loss,
            'total': total_loss
        }