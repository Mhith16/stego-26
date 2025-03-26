import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
import time
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_analyzer import FeatureAnalysisDenseNet, FeatureAnalysisUNet
from models.encoder import SteganographyEncoder
from models.decoder import SteganographyDecoder, MultiScaleDecoder
from models.discriminator import Discriminator, PatchDiscriminator
from models.noise_layer import NoiseLayer
from models.vgg_perceptual import PerceptualLoss
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_bit_accuracy
from utils.error_correction import ErrorCorrection
from utils.losses import SteganoLoss, DiscriminatorLoss

def train(args):
    """
    Main training function with support for resuming from checkpoints
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate a unique run ID if not resuming
    if not args.resume:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(args.model_save_path, run_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = os.path.dirname(args.resume)
        run_id = os.path.basename(checkpoint_dir)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_id))
    
    # Save configuration
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if not args.resume or not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = get_data_loaders(
        args.xray_dir, 
        args.label_dir,
        batch_size=args.batch_size,
        resolution=(args.image_width, args.image_height),
        max_message_length=args.message_length,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    
    # Initialize models
    print("Initializing models...")
    
    # Feature analyzer model
    if args.feature_analyzer == 'densenet':
        feature_analyzer = FeatureAnalysisDenseNet(
            model_type=args.feature_backbone,
            pretrained=args.pretrained,
            in_channels=1,
            freeze_backbone=args.freeze_backbone
        ).to(device)
    elif args.feature_analyzer == 'unet':
        feature_analyzer = FeatureAnalysisUNet(
            in_channels=1,
            base_filters=64,
            pretrained_encoder=args.pretrained
        ).to(device)
    else:
        raise ValueError(f"Unsupported feature analyzer type: {args.feature_analyzer}")
    
    # Steganography encoder
    encoder = SteganographyEncoder(
        image_channels=1,
        embedding_strength=args.embedding_strength
    ).to(device)
    
    # Steganography decoder
    if args.multi_scale_decoder:
        decoder = MultiScaleDecoder(
            image_channels=1,
            message_length=args.message_length
        ).to(device)
    else:
        decoder = SteganographyDecoder(
            image_channels=1,
            message_length=args.message_length,
            growth_rate=32,
            num_dense_layers=6,
            with_dct=args.use_dct_decoder
        ).to(device)
    
    # Discriminator
    if args.patch_discriminator:
        discriminator = PatchDiscriminator(
            image_channels=1,
            base_filters=64,
            n_layers=3
        ).to(device)
    else:
        discriminator = Discriminator(
            image_channels=1,
            base_filters=64,
            use_spectral_norm=args.spectral_norm
        ).to(device)
    
    # Noise layer
    noise_layer = NoiseLayer().to(device)
    
    # Error correction if enabled
    if args.use_error_correction:
        error_corrector = ErrorCorrection(ecc_bytes=args.ecc_bytes)
    else:
        error_corrector = None
    
    # Initialize optimizers
    fa_optimizer = optim.Adam(feature_analyzer.parameters(), lr=args.lr_feature)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr_encoder)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr_decoder)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_disc)
    
    # Initialize learning rate schedulers
    if args.use_scheduler:
        fa_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fa_optimizer, 'min', patience=5, factor=0.5)
        encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=5, factor=0.5)
        decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=5, factor=0.5)
        disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(disc_optimizer, 'min', patience=5, factor=0.5)
    else:
        fa_scheduler = None
        encoder_scheduler = None
        decoder_scheduler = None
        disc_scheduler = None
    
    # Initialize loss functions
    stegano_loss = SteganoLoss(
        lambda_message=args.lambda_message,
        lambda_image=args.lambda_image,
        lambda_adv=args.lambda_adv,
        lambda_perceptual=args.lambda_perceptual,
        use_perceptual=args.use_perceptual,
        adaptive_weights=args.adaptive_weights
    ).to(device)
    
    disc_loss = DiscriminatorLoss(label_smoothing=args.label_smoothing).to(device)
    
    # Starting epoch and best validation metrics
    start_epoch = 0
    best_val_psnr = 0
    best_val_bit_acc = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model weights
        feature_analyzer.load_state_dict(checkpoint['feature_analyzer_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states
        fa_optimizer.load_state_dict(checkpoint['fa_optimizer_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        # Load scheduler states if they exist
        if 'fa_scheduler_state_dict' in checkpoint and fa_scheduler:
            fa_scheduler.load_state_dict(checkpoint['fa_scheduler_state_dict'])
            encoder_scheduler.load_state_dict(checkpoint['encoder_scheduler_state_dict'])
            decoder_scheduler.load_state_dict(checkpoint['decoder_scheduler_state_dict'])
            disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        
        # Load training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_psnr = checkpoint.get('best_val_psnr', 0)
        best_val_bit_acc = checkpoint.get('best_val_bit_acc', 0)
        
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Update loss function with current epoch
        if args.adaptive_weights:
            stegano_loss.set_epoch(epoch)
        
        # Training phase
        feature_analyzer.train()
        encoder.train()
        decoder.train()
        discriminator.train()
        
        train_losses = {
            'disc_loss': 0,
            'encoder_loss': 0,
            'decoder_loss': 0,
            'perceptual_loss': 0,
            'total_loss': 0,
            'psnr': 0,
            'bit_accuracy': 0
        }
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training loop with progress bar
        with tqdm(train_loader, desc=f"Train Epoch {epoch+1}", unit="batch") as t:
            for batch_idx, data in enumerate(t):
                images = data['image'].to(device)
                messages = data['patient_data'].to(device)
                
                # Apply error correction if enabled
                if error_corrector:
                    messages = error_corrector.encode_message(messages)
                
                # Ensure messages have the right dimensions [B, L]
                if messages.dim() == 3:  # [B, 1, L]
                    messages = messages.squeeze(1)
                
                # Step 1: Train discriminator
                disc_optimizer.zero_grad()
                
                # Generate stego images
                feature_weights = feature_analyzer(images)
                stego_images = encoder(images, messages, feature_weights)
                
                # Get discriminator predictions
                real_preds = discriminator(images)
                fake_preds = discriminator(stego_images.detach())
                
                # Calculate discriminator loss
                d_loss = disc_loss(real_preds, fake_preds)
                d_loss['total'].backward()
                disc_optimizer.step()
                
                # Step 2: Train encoder, decoder, and feature analyzer
                fa_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                
                # Generate feature weights again (for encoder training)
                feature_weights = feature_analyzer(images)
                
                # Generate stego images
                stego_images = encoder(images, messages, feature_weights)
                
                # Apply noise
                noisy_stego_images = noise_layer(stego_images)
                
                # Decode messages from noisy stego images
                if args.multi_scale_decoder:
                    decoded_messages, confidence_scores = decoder(noisy_stego_images)
                else:
                    # Handle both regular and multi-output decoders
                    decoder_output = decoder(noisy_stego_images)
                    if isinstance(decoder_output, tuple):
                        decoded_messages, confidence_scores = decoder_output
                    else:
                        decoded_messages = decoder_output
                        confidence_scores = None
                
                # Get discriminator predictions for generator training
                disc_preds = discriminator(stego_images)
                
                # Calculate combined loss
                losses = stegano_loss(
                    original_images=images, 
                    stego_images=stego_images,
                    original_messages=messages, 
                    decoded_messages=decoded_messages,
                    confidence_scores=confidence_scores,
                    disc_real_pred=real_preds, 
                    disc_fake_pred=disc_preds
                )
                
                # Backpropagate and update weights
                losses['total'].backward()
                fa_optimizer.step()
                encoder_optimizer.step()
                decoder_optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    psnr = compute_psnr(images, stego_images)
                    bit_acc = compute_bit_accuracy(messages, decoded_messages)
                
                # Update running losses
                train_losses['disc_loss'] += d_loss['total'].item()
                train_losses['encoder_loss'] += losses['image'].item()
                train_losses['decoder_loss'] += losses['message'].item()
                train_losses['perceptual_loss'] += losses.get('perceptual', torch.tensor(0.0)).item()
                train_losses['total_loss'] += losses['total'].item()
                train_losses['psnr'] += psnr
                train_losses['bit_accuracy'] += bit_acc
                
                # Update progress bar
                t.set_postfix(psnr=f"{psnr:.2f}", bit_acc=f"{bit_acc:.4f}")
                
                # Log every N batches
                if batch_idx % args.log_interval == 0:
                    # Log to tensorboard
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Train/Discriminator_Loss', d_loss['total'].item(), step)
                    writer.add_scalar('Train/Encoder_Loss', losses['image'].item(), step)
                    writer.add_scalar('Train/Decoder_Loss', losses['message'].item(), step)
                    writer.add_scalar('Train/Total_Loss', losses['total'].item(), step)
                    writer.add_scalar('Train/PSNR', psnr, step)
                    writer.add_scalar('Train/BitAccuracy', bit_acc, step)
                    
                    # Add images to tensorboard
                    if batch_idx % (args.log_interval * 5) == 0:
                        writer.add_images('Train/Original', images[:4], step)
                        writer.add_images('Train/Stego', stego_images[:4], step)
                        writer.add_images('Train/Noisy_Stego', noisy_stego_images[:4], step)
                        # Log feature weights
                        writer.add_images('Train/Feature_Weights', feature_weights[:4], step)
                        
                        # Create difference visualization
                        diff = torch.abs(images - stego_images)
                        normalized_diff = diff / diff.max()
                        writer.add_images('Train/Difference', normalized_diff[:4], step)
        
        # Calculate average training losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Validation phase
        feature_analyzer.eval()
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        
        val_losses = {
            'total_loss': 0,
            'psnr': 0,
            'ssim': 0,
            'bit_accuracy': 0
        }
        
        val_noise_accuracy = {noise_type: 0 for noise_type in ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']}
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Validate Epoch {epoch+1}", unit="batch") as t:
                for batch_idx, data in enumerate(t):
                    images = data['image'].to(device)
                    messages = data['patient_data'].to(device)
                    
                    # Apply error correction if enabled
                    if error_corrector:
                        messages = error_corrector.encode_message(messages)
                    
                    # Ensure messages have the right dimensions
                    if messages.dim() == 3:  # [B, 1, L]
                        messages = messages.squeeze(1)
                    
                    # Generate feature weights
                    feature_weights = feature_analyzer(images)
                    
                    # Generate stego images
                    stego_images = encoder(images, messages, feature_weights)
                    
                    # Calculate metrics
                    psnr = compute_psnr(images, stego_images)
                    ssim = compute_ssim(images, stego_images)
                    
                    # Decode from clean stego images
                    if args.multi_scale_decoder:
                        decoded_messages, _ = decoder(stego_images)
                    else:
                        decoder_output = decoder(stego_images)
                        if isinstance(decoder_output, tuple):
                            decoded_messages, _ = decoder_output
                        else:
                            decoded_messages = decoder_output
                    
                    # Calculate bit accuracy
                    clean_bit_acc = compute_bit_accuracy(messages, decoded_messages)
                    
                    # Apply different noise types and evaluate
                    for noise_type in val_noise_accuracy.keys():
                        # Apply noise
                        noisy_stego = noise_layer(stego_images, noise_type)
                        
                        # Decode message
                        if args.multi_scale_decoder:
                            decoded, _ = decoder(noisy_stego)
                        else:
                            decoder_output = decoder(noisy_stego)
                            if isinstance(decoder_output, tuple):
                                decoded, _ = decoder_output
                            else:
                                decoded = decoder_output
                        
                        # Calculate bit accuracy with error correction if enabled
                        if error_corrector:
                            decoded = error_corrector.decode_message(decoded)
                        
                        noise_bit_acc = compute_bit_accuracy(messages, decoded)
                        val_noise_accuracy[noise_type] += noise_bit_acc
                    
                    # Update validation metrics
                    val_losses['psnr'] += psnr
                    val_losses['ssim'] += ssim
                    val_losses['bit_accuracy'] += clean_bit_acc
                    
                    # Update progress bar
                    t.set_postfix(psnr=f"{psnr:.2f}", bit_acc=f"{clean_bit_acc:.4f}")
        
        # Calculate average validation metrics
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        for key in val_noise_accuracy:
            val_noise_accuracy[key] /= len(val_loader)
        
        # Calculate average noise bit accuracy
        avg_noise_bit_acc = sum(val_noise_accuracy.values()) / len(val_noise_accuracy)
        
        # Log validation metrics
        writer.add_scalar('Validation/PSNR', val_losses['psnr'], epoch)
        writer.add_scalar('Validation/SSIM', val_losses['ssim'], epoch)
        writer.add_scalar('Validation/BitAccuracy', val_losses['bit_accuracy'], epoch)
        writer.add_scalar('Validation/AvgNoiseAccuracy', avg_noise_bit_acc, epoch)
        
        for noise_type, accuracy in val_noise_accuracy.items():
            writer.add_scalar(f'Validation/BitAcc_{noise_type}', accuracy, epoch)
        
        # Update learning rate schedulers if used
        if args.use_scheduler:
            fa_scheduler.step(val_losses['psnr'])
            encoder_scheduler.step(val_losses['psnr'])
            decoder_scheduler.step(1 - val_losses['bit_accuracy'])
            disc_scheduler.step(train_losses['disc_loss'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training   - PSNR: {train_losses['psnr']:.2f} dB, Bit Accuracy: {train_losses['bit_accuracy']:.4f}")
        print(f"  Validation - PSNR: {val_losses['psnr']:.2f} dB, SSIM: {val_losses['ssim']:.4f}, "
              f"Bit Accuracy: {val_losses['bit_accuracy']:.4f}")
        print(f"  Noise Robustness - Average: {avg_noise_bit_acc:.4f}")
        for noise_type, accuracy in val_noise_accuracy.items():
            print(f"    {noise_type}: {accuracy:.4f}")
        
        # Check if this is the best model so far (based on PSNR and bit accuracy)
        is_best_psnr = val_losses['psnr'] > best_val_psnr
        is_best_bit_acc = val_losses['bit_accuracy'] > best_val_bit_acc
        
        if is_best_psnr:
            best_val_psnr = val_losses['psnr']
            # Save best PSNR model
            torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, 'best_psnr_encoder.pth'))
            torch.save(feature_analyzer.state_dict(), os.path.join(checkpoint_dir, 'best_psnr_feature_analyzer.pth'))
        
        if is_best_bit_acc:
            best_val_bit_acc = val_losses['bit_accuracy']
            # Save best bit accuracy model
            torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, 'best_bitacc_decoder.pth'))
        
        # Save checkpoint if needed
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'feature_analyzer_state_dict': feature_analyzer.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'fa_optimizer_state_dict': fa_optimizer.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'best_val_psnr': best_val_psnr,
                'best_val_bit_acc': best_val_bit_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_noise_accuracy': val_noise_accuracy
            }
            
            # Add scheduler states if they exist
            if args.use_scheduler:
                checkpoint.update({
                    'fa_scheduler_state_dict': fa_scheduler.state_dict(),
                    'encoder_scheduler_state_dict': encoder_scheduler.state_dict(),
                    'decoder_scheduler_state_dict': decoder_scheduler.state_dict(),
                    'disc_scheduler_state_dict': disc_scheduler.state_dict()
                })
            
            # Save the checkpoint
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Save individual models for easier loading
            torch.save(feature_analyzer.state_dict(), os.path.join(checkpoint_dir, 'feature_analyzer.pth'))
            torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, 'encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, 'decoder.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))
            
            print(f"Checkpoint saved to {checkpoint_dir}")
    
    # Training complete
    print("\nTraining complete!")
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Best validation bit accuracy: {best_val_bit_acc:.4f}")
    
    # Close tensorboard writer
    writer.close()
    
    # Return paths to best models
    return {
        'feature_analyzer': os.path.join(checkpoint_dir, 'best_psnr_feature_analyzer.pth'),
        'encoder': os.path.join(checkpoint_dir, 'best_psnr_encoder.pth'),
        'decoder': os.path.join(checkpoint_dir, 'best_bitacc_decoder.pth'),
        'final_dir': checkpoint_dir
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train enhanced steganography models")
    
    # Data paths
    parser.add_argument('--xray_dir', type=str, required=True, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing patient data text files')
    
    # Image and message parameters
    parser.add_argument('--image_width', type=int, default=512, help='Target image width')
    parser.add_argument('--image_height', type=int, default=512, help='Target image height')
    parser.add_argument('--message_length', type=int, default=4096, help='Length of binary message in bits')
    
    # Model configuration
    parser.add_argument('--feature_analyzer', type=str, default='densenet', choices=['densenet', 'unet'], 
                      help='Type of feature analyzer model')
    parser.add_argument('--feature_backbone', type=str, default='densenet121', 
                      choices=['densenet121', 'resnet50', 'efficientnet_b0'], 
                      help='Backbone model for feature analyzer')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights for feature analyzer')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights')
    parser.add_argument('--embedding_strength', type=float, default=0.02, help='Strength of embedding')
    parser.add_argument('--multi_scale_decoder', action='store_true', help='Use multi-scale decoder')
    parser.add_argument('--use_dct_decoder', action='store_true', help='Use DCT in decoder')
    parser.add_argument('--patch_discriminator', action='store_true', help='Use patch discriminator')
    parser.add_argument('--spectral_norm', action='store_true', help='Use spectral normalization in discriminator')
    
    # Error correction
    parser.add_argument('--use_error_correction', action='store_true', help='Use Reed-Solomon error correction')
    parser.add_argument('--ecc_bytes', type=int, default=16, help='Error correction bytes per chunk')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr_feature', type=float, default=0.0001, help='Learning rate for feature analyzer')
    parser.add_argument('--lr_encoder', type=float, default=0.0001, help='Learning rate for encoder')
    parser.add_argument('--lr_decoder', type=float, default=0.0001, help='Learning rate for decoder')
    parser.add_argument('--lr_disc', type=float, default=0.0001, help='Learning rate for discriminator')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    # Loss weights
    parser.add_argument('--lambda_message', type=float, default=20.0, help='Weight for message loss')
    parser.add_argument('--lambda_image', type=float, default=1.0, help='Weight for image loss')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Weight for adversarial loss')
    parser.add_argument('--lambda_perceptual', type=float, default=0.5, help='Weight for perceptual loss')
    parser.add_argument('--use_perceptual', action='store_true', help='Use perceptual loss')
    parser.add_argument('--adaptive_weights', action='store_true', help='Adjust loss weights adaptively')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing for GAN training')
    
    # Learning rate scheduling
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')
    parser.add_argument('--model_save_path', type=str, default='./models/weights', help='Directory to save models')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging')
    parser.add_argument('--save_interval', type=int, default=5, help='How many epochs to wait before saving')
    
    # Resumable training
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    train(args)