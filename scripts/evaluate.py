import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import json
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_analyzer import FeatureAnalysisDenseNet, FeatureAnalysisUNet
from models.encoder import SteganographyEncoder
from models.decoder import SteganographyDecoder, MultiScaleDecoder
from models.noise_layer import NoiseLayer
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_bit_accuracy, compute_embedding_capacity
from utils.error_correction import ErrorCorrection

def evaluate(args):
    """
    Comprehensive evaluation of steganography models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    _, val_loader = get_data_loaders(
        args.xray_dir,
        args.label_dir,
        batch_size=args.batch_size,
        resolution=(args.image_width, args.image_height),
        max_message_length=args.message_length,
        val_split=1.0  # Use all data for evaluation
    )
    
    print(f"Loaded {len(val_loader.dataset)} test samples")
    
    # Initialize models
    print("Initializing models...")
    
    # Feature analyzer
    if args.feature_analyzer == 'densenet':
        feature_analyzer = FeatureAnalysisDenseNet(
            model_type=args.feature_backbone,
            pretrained=False,  # No need for pretrained during evaluation
            in_channels=1
        ).to(device)
    elif args.feature_analyzer == 'unet':
        feature_analyzer = FeatureAnalysisUNet(
            in_channels=1,
            base_filters=64,
            pretrained_encoder=False
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
    
    # Noise layer
    noise_layer = NoiseLayer().to(device)
    
    # Error correction if enabled
    if args.use_error_correction:
        error_corrector = ErrorCorrection(ecc_bytes=args.ecc_bytes)
    else:
        error_corrector = None
    
    # Load model weights
    try:
        feature_analyzer.load_state_dict(torch.load(os.path.join(args.model_path, 'feature_analyzer.pth')))
        encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'decoder.pth')))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Set models to evaluation mode
    feature_analyzer.eval()
    encoder.eval()
    decoder.eval()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"eval_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
    
    # Save evaluation settings
    with open(os.path.join(results_dir, 'eval_settings.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Define noise types to evaluate
    noise_types = ['none', 'dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']
    
    # Initialize metrics storage
    metrics = {
        noise_type: {
            'psnr': [],
            'ssim': [],
            'bit_accuracy': [],
            'bit_accuracy_with_ecc': [],
            'embedding_capacity': [],
            'char_error_rate': []
        } for noise_type in noise_types
    }
    
    # Store examples for visualization
    example_images = []
    
    # Initialize counters for perfect recovery
    perfect_recovery = {noise_type: 0 for noise_type in noise_types}
    total_samples = 0
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = data['image'].to(device)
            messages = data['patient_data'].to(device)
            original_text = data['patient_text']
            
            # Ensure messages have the right dimensions
            if messages.dim() == 3:  # [B, 1, L]
                messages = messages.squeeze(1)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Apply error correction encoding if enabled
            if error_corrector:
                encoded_messages = error_corrector.encode_message(messages)
            else:
                encoded_messages = messages
                
            # Generate feature weights
            feature_weights = feature_analyzer(images)
            
            # Generate stego images
            stego_images = encoder(images, encoded_messages, feature_weights)
            
            # Compute metrics for clean stego images
            metrics['none']['psnr'].append(compute_psnr(images, stego_images))
            metrics['none']['ssim'].append(compute_ssim(images, stego_images))
            
            # Extract message from clean stego images
            if args.multi_scale_decoder:
                clean_decoded, clean_confidence = decoder(stego_images)
            else:
                decoder_output = decoder(stego_images)
                if isinstance(decoder_output, tuple):
                    clean_decoded, clean_confidence = decoder_output
                else:
                    clean_decoded = decoder_output
                    clean_confidence = None
            
            # Apply error correction decoding if enabled
            if error_corrector:
                clean_corrected = error_corrector.decode_message(clean_decoded)
                clean_bit_acc_with_ecc = compute_bit_accuracy(messages, clean_corrected)
                metrics['none']['bit_accuracy_with_ecc'].append(clean_bit_acc_with_ecc)
            else:
                clean_corrected = clean_decoded
                metrics['none']['bit_accuracy_with_ecc'].append(0)  # Placeholder
            
            # Count perfect recoveries for clean images
            for i in range(batch_size):
                if torch.all(torch.abs(clean_corrected[i] - messages[i]) < 0.1):
                    perfect_recovery['none'] += 1
            
            # Compute raw bit accuracy
            clean_bit_acc = compute_bit_accuracy(messages, clean_decoded)
            metrics['none']['bit_accuracy'].append(clean_bit_acc)
            
            # Compute embedding capacity
            embedding_capacity = compute_embedding_capacity(images.shape, args.message_length)
            metrics['none']['embedding_capacity'].append(embedding_capacity)
            
            # Calculate character error rate
            # (Placeholder - would be implemented with actual text recovery)
            metrics['none']['char_error_rate'].append(0)
            
            # Apply different noise types and evaluate
            for noise_type in noise_types:
                if noise_type == 'none':
                    continue
                
                # Apply noise
                noisy_stego = noise_layer(stego_images, noise_type)
                
                # Decode message from noisy stego images
                if args.multi_scale_decoder:
                    noisy_decoded, noisy_confidence = decoder(noisy_stego)
                else:
                    decoder_output = decoder(noisy_stego)
                    if isinstance(decoder_output, tuple):
                        noisy_decoded, noisy_confidence = decoder_output
                    else:
                        noisy_decoded = decoder_output
                        noisy_confidence = None
                
                # Apply error correction if enabled
                if error_corrector:
                    noisy_corrected = error_corrector.decode_message(noisy_decoded)
                    noisy_bit_acc_with_ecc = compute_bit_accuracy(messages, noisy_corrected)
                    metrics[noise_type]['bit_accuracy_with_ecc'].append(noisy_bit_acc_with_ecc)
                else:
                    noisy_corrected = noisy_decoded
                    metrics[noise_type]['bit_accuracy_with_ecc'].append(0)  # Placeholder
                
                # Count perfect recoveries for noisy images
                for i in range(batch_size):
                    if torch.all(torch.abs(noisy_corrected[i] - messages[i]) < 0.1):
                        perfect_recovery[noise_type] += 1
                
                # Compute bit accuracy
                bit_acc = compute_bit_accuracy(messages, noisy_decoded)
                metrics[noise_type]['bit_accuracy'].append(bit_acc)
                
                # PSNR and SSIM compare original to stego, so they're the same for all noise types
                metrics[noise_type]['psnr'] = metrics['none']['psnr']
                metrics[noise_type]['ssim'] = metrics['none']['ssim']
                
                # Compute effective embedding capacity based on bit accuracy
                effective_capacity = embedding_capacity * bit_acc
                metrics[noise_type]['embedding_capacity'].append(effective_capacity)
                
                # Calculate character error rate
                # (Placeholder - would be implemented with actual text recovery)
                metrics[noise_type]['char_error_rate'].append(0)
            
            # Save example images from the first few batches
            if batch_idx < args.num_examples:
                for i in range(min(args.batch_size, images.size(0))):
                    example = {
                        'original': images[i].cpu().numpy(),
                        'stego': stego_images[i].cpu().numpy(),
                        'feature_weights': feature_weights[i].cpu().numpy(),
                        'noisy_images': {},
                        'message': messages[i].cpu().numpy(),
                        'decoded_message': clean_decoded[i].cpu().numpy(),
                        'original_text': original_text[i]
                    }
                    
                    for noise_type in noise_types:
                        if noise_type == 'none':
                            continue
                        noisy_stego = noise_layer(stego_images[i:i+1], noise_type)
                        example['noisy_images'][noise_type] = noisy_stego[0].cpu().numpy()
                    
                    example_images.append(example)
    
    # Compute average metrics
    avg_metrics = {
        noise_type: {
            metric: np.mean(values) for metric, values in noise_metrics.items()
            if len(values) > 0
        } for noise_type, noise_metrics in metrics.items()
    }
    
    # Calculate perfect recovery rates
    recovery_rates = {
        noise_type: count / total_samples for noise_type, count in perfect_recovery.items()
    }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"{'Noise Type':15} {'PSNR (dB)':10} {'SSIM':10} {'Bit Accuracy':15} {'With ECC':15} {'Perfect Recovery':15}")
    print("-" * 80)
    
    for noise_type, noise_metrics in avg_metrics.items():
        print(f"{noise_type:15} {noise_metrics['psnr']:10.2f} {noise_metrics['ssim']:10.4f} "
              f"{noise_metrics['bit_accuracy']:15.4f} "
              f"{noise_metrics.get('bit_accuracy_with_ecc', 0):15.4f} "
              f"{recovery_rates[noise_type]*100:15.2f}%")
    
    # Save results
    with open(os.path.join(results_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"{'Noise Type':15} {'PSNR (dB)':10} {'SSIM':10} {'Bit Accuracy':15} {'With ECC':15} {'Perfect Recovery':15}\n")
        f.write("-" * 80 + "\n")
        
        for noise_type, noise_metrics in avg_metrics.items():
            f.write(f"{noise_type:15} {noise_metrics['psnr']:10.2f} {noise_metrics['ssim']:10.4f} "
                   f"{noise_metrics['bit_accuracy']:15.4f} "
                   f"{noise_metrics.get('bit_accuracy_with_ecc', 0):15.4f} "
                   f"{recovery_rates[noise_type]*100:15.2f}%\n")
    
    # Save metrics as JSON
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'avg_metrics': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in avg_metrics.items()},
            'recovery_rates': {k: float(v) for k, v in recovery_rates.items()}
        }, f, indent=4)
    
    # Generate and save visualizations
    generate_visualizations(example_images, results_dir)
    
    print(f"\nResults saved to {results_dir}")
    
    return {
        'avg_metrics': avg_metrics,
        'recovery_rates': recovery_rates,
        'results_dir': results_dir
    }

def generate_visualizations(example_images, results_dir):
    """
    Generate and save visualizations of the steganography results
    
    Args:
        example_images (list): List of example image dictionaries
        results_dir (str): Directory to save visualizations
    """
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, example in enumerate(example_images):
        # 1. Original vs Stego comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(example['original'][0], cmap='gray')
        plt.title("Original X-ray")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(example['stego'][0], cmap='gray')
        plt.title("Stego Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Enhance difference for visualization
        diff = np.abs(example['original'][0] - example['stego'][0])
        # Scale the difference for better visibility
        scaled_diff = diff / max(diff.max(), 0.001) * 10
        plt.imshow(scaled_diff, cmap='hot')
        plt.title("Difference (×10)")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'example_{i+1}_comparison.png'), dpi=200)
        plt.close()
        
        # 2. Feature weights visualization
        plt.figure(figsize=(6, 6))
        plt.imshow(example['feature_weights'][0], cmap='viridis')
        plt.title("Feature Density Map")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'example_{i+1}_feature_weights.png'), dpi=200)
        plt.close()
        
        # 3. Noise comparisons
        noise_types = list(example['noisy_images'].keys())
        n_cols = 3
        n_rows = (len(noise_types) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for j, noise_type in enumerate(noise_types):
            plt.subplot(n_rows, n_cols, j + 1)
            plt.imshow(example['noisy_images'][noise_type][0], cmap='gray')
            plt.title(f"Noise: {noise_type}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'example_{i+1}_noise_comparison.png'), dpi=200)
        plt.close()
        
        # 4. Message bit visualization - original vs decoded
        plt.figure(figsize=(12, 4))
        
        # Only show first 100 bits for clarity
        display_len = min(100, len(example['message']))
        
        # Original message
        plt.subplot(2, 1, 1)
        plt.imshow(example['message'][:display_len].reshape(1, -1), cmap='binary', aspect='auto')
        plt.title("Original Message (first 100 bits)")
        plt.yticks([])
        
        # Decoded message
        plt.subplot(2, 1, 2)
        plt.imshow(np.round(example['decoded_message'][:display_len]).reshape(1, -1), cmap='binary', aspect='auto')
        plt.title("Decoded Message (first 100 bits)")
        plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'example_{i+1}_message_bits.png'), dpi=200)
        plt.close()
        
        # 5. Text visualization
        with open(os.path.join(vis_dir, f'example_{i+1}_text.txt'), 'w') as f:
            f.write(f"Original text:\n{example['original_text']}\n\n")
            f.write(f"Bit accuracy: {np.mean(np.abs(example['message'] - example['decoded_message']) < 0.1):.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate steganography models")
    
    # Data paths
    parser.add_argument('--xray_dir', type=str, required=True, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing patient data text files')
    
    # Model paths
    parser.add_argument('--model_path', type=str, required=True, help='Directory containing trained models')
    
    # Model configuration
    parser.add_argument('--feature_analyzer', type=str, default='densenet', choices=['densenet', 'unet'], 
                      help='Type of feature analyzer model')
    parser.add_argument('--feature_backbone', type=str, default='densenet121', 
                      choices=['densenet121', 'resnet50', 'efficientnet_b0'], 
                      help='Backbone model for feature analyzer')
    parser.add_argument('--embedding_strength', type=float, default=0.02, help='Strength of embedding')
    parser.add_argument('--multi_scale_decoder', action='store_true', help='Use multi-scale decoder')
    parser.add_argument('--use_dct_decoder', action='store_true', help='Use DCT in decoder')
    
    # Error correction
    parser.add_argument('--use_error_correction', action='store_true', help='Use Reed-Solomon error correction')
    parser.add_argument('--ecc_bytes', type=int, default=16, help='Error correction bytes per chunk')
    
    # Evaluation parameters
    parser.add_argument('--image_width', type=int, default=512, help='Target image width')
    parser.add_argument('--image_height', type=int, default=512, help='Target image height')
    parser.add_argument('--message_length', type=int, default=4096, help='Length of binary message')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of example images to visualize')
    
    # Results
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    evaluate(args)