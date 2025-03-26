import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

from models.feature_analyzer import FeatureAnalysisDenseNet, FeatureAnalysisUNet
from models.encoder import SteganographyEncoder
from models.decoder import SteganographyDecoder, MultiScaleDecoder
from models.noise_layer import NoiseLayer
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_bit_accuracy
from utils.error_correction import ErrorCorrection
from scripts.embed_data import embed_patient_data
from scripts.extract_data import extract_patient_data

def demo(args):
    """
    Run a demonstration of the steganography system
    
    Args:
        args: Command line arguments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"demo_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print("\n=== Step 1: Loading Models ===")
    
    # Initialize models based on command line arguments
    if args.feature_analyzer == 'densenet':
        feature_analyzer = FeatureAnalysisDenseNet(
            model_type=args.feature_backbone,
            pretrained=False,
            in_channels=1
        ).to(device)
    else:
        feature_analyzer = FeatureAnalysisUNet(
            in_channels=1,
            base_filters=64,
            pretrained_encoder=False
        ).to(device)
    
    encoder = SteganographyEncoder(
        image_channels=1,
        embedding_strength=args.embedding_strength
    ).to(device)
    
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
    
    noise_layer = NoiseLayer().to(device)
    
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
    
    # Initialize error correction if enabled
    if args.use_error_correction:
        error_corrector = ErrorCorrection(ecc_bytes=args.ecc_bytes)
        print(f"Using Reed-Solomon error correction with {args.ecc_bytes} ECC bytes per chunk")
    else:
        error_corrector = None
        print("Error correction disabled")
    
    if args.mode == 'sample':
        print("\n=== Step 2: Loading Sample Data ===")
        
        # Load sample dataset
        _, val_loader = get_data_loaders(
            args.xray_dir,
            args.label_dir,
            batch_size=1,
            resolution=(args.image_width, args.image_height),
            max_message_length=args.message_length,
            val_split=1.0
        )
        
        if len(val_loader) == 0:
            print("Error: No valid samples found in the dataset")
            return
        
        # Get a sample
        data = next(iter(val_loader))
        original_image = data['image'].to(device)
        original_message = data['patient_data'].to(device)
        patient_text = data['patient_text'][0]
        
        print(f"Loaded sample X-ray image: {data['file_name'][0]}")
        print(f"Patient data sample: {patient_text[:100]}...")
        
        print("\n=== Step 3: Embedding Data ===")
        
        with torch.no_grad():
            # Generate feature weights
            feature_weights = feature_analyzer(original_image)
            
            # Apply error correction if enabled
            if error_corrector:
                encoded_message = error_corrector.encode_message(original_message)
            else:
                encoded_message = original_message
            
            # Generate stego image
            stego_image = encoder(original_image, encoded_message, feature_weights)
            
            # Calculate metrics
            psnr = compute_psnr(original_image, stego_image)
            ssim = compute_ssim(original_image, stego_image)
            
            print(f"Generated stego image - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
            print("\n=== Step 4: Extracting Data ===")
            
            # Decode message directly
            if args.multi_scale_decoder:
                decoded_message, confidence_scores = decoder(stego_image)
            else:
                decoder_output = decoder(stego_image)
                if isinstance(decoder_output, tuple):
                    decoded_message, confidence_scores = decoder_output
                else:
                    decoded_message = decoder_output
                    confidence_scores = None
            
            # Apply error correction if enabled
            if error_corrector:
                corrected_message = error_corrector.decode_message(decoded_message)
            else:
                corrected_message = decoded_message
            
            # Calculate bit accuracy
            bit_accuracy = compute_bit_accuracy(original_message, corrected_message)
            print(f"Clean extraction - Bit accuracy: {bit_accuracy:.4f}")
            
            # Apply different noise types and evaluate
            noise_results = {}
            for noise_type in ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']:
                # Apply noise
                noisy_stego = noise_layer(stego_image, noise_type)
                
                # Decode message
                if args.multi_scale_decoder:
                    noisy_decoded, _ = decoder(noisy_stego)
                else:
                    decoder_output = decoder(noisy_stego)
                    if isinstance(decoder_output, tuple):
                        noisy_decoded, _ = decoder_output
                    else:
                        noisy_decoded = decoder_output
                
                # Apply error correction if enabled
                if error_corrector:
                    noisy_corrected = error_corrector.decode_message(noisy_decoded)
                else:
                    noisy_corrected = noisy_decoded
                
                # Calculate bit accuracy
                noise_bit_acc = compute_bit_accuracy(original_message, noisy_corrected)
                noise_results[noise_type] = {
                    'bit_accuracy': noise_bit_acc,
                    'image': noisy_stego.cpu().numpy()
                }
                
                print(f"Noisy extraction ({noise_type}) - Bit accuracy: {noise_bit_acc:.4f}")
        
        # Save results
        print("\n=== Step 5: Saving Results ===")
        
        # Save sample images
        stego_path = os.path.join(results_dir, "sample_stego.png")
        plt.imsave(stego_path, stego_image[0, 0].cpu().numpy(), cmap='gray')
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Original and stego images
        plt.subplot(2, 3, 1)
        plt.imshow(original_image[0, 0].cpu().numpy(), cmap='gray')
        plt.title("Original X-ray")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(stego_image[0, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Stego Image (PSNR: {psnr:.2f} dB)")
        plt.axis('off')
        
        # Difference visualization
        plt.subplot(2, 3, 3)
        diff = np.abs(original_image[0, 0].cpu().numpy() - stego_image[0, 0].cpu().numpy())
        scaled_diff = diff / max(diff.max(), 0.001) * 5
        plt.imshow(scaled_diff, cmap='hot')
        plt.title("Difference (×5)")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Feature weights
        plt.subplot(2, 3, 4)
        plt.imshow(feature_weights[0, 0].cpu().numpy(), cmap='viridis')
        plt.title("Feature Weights")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Noise examples (pick two)
        i = 5
        for noise_type in list(noise_results.keys())[:2]:
            plt.subplot(2, 3, i)
            i += 1
            plt.imshow(noise_results[noise_type]['image'][0, 0], cmap='gray')
            plt.title(f"{noise_type} (Acc: {noise_results[noise_type]['bit_accuracy']:.4f})")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "demo_results.png"), dpi=200)
        
        # Create text report
        with open(os.path.join(results_dir, "report.txt"), 'w') as f:
            f.write("=== Medical Image Steganography Demo ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image Resolution: {args.image_width}x{args.image_height}\n")
            f.write(f"Message Length: {args.message_length} bits\n")
            f.write(f"Error Correction: {'Enabled' if args.use_error_correction else 'Disabled'}\n\n")
            
            f.write("=== Image Quality ===\n")
            f.write(f"PSNR: {psnr:.2f} dB\n")
            f.write(f"SSIM: {ssim:.4f}\n\n")
            
            f.write("=== Extraction Results ===\n")
            f.write(f"Clean Bit Accuracy: {bit_accuracy:.4f}\n")
            for noise_type, result in noise_results.items():
                f.write(f"{noise_type}: {result['bit_accuracy']:.4f}\n")
        
        print(f"Results saved to {results_dir}")
    
    elif args.mode == 'embed':
        # Embed data into a specific image
        if not args.input_image or not args.input_text:
            print("Error: Input image and text required for embed mode")
            return
        
        print("\n=== Step 2: Embedding Data ===")
        
        # Build embedding configuration
        embed_config = {
            'resolution': (args.image_width, args.image_height),
            'message_length': args.message_length,
            'feature_analyzer': args.feature_analyzer,
            'feature_backbone': args.feature_backbone,
            'embedding_strength': args.embedding_strength,
            'use_error_correction': args.use_error_correction,
            'ecc_bytes': args.ecc_bytes
        }
        
        # Embed data
        output_path = os.path.join(results_dir, f"stego_{os.path.basename(args.input_image)}")
        result = embed_patient_data(args.input_image, args.input_text, args.model_path, output_path, embed_config)
        
        if not result:
            print("Error: Embedding failed")
            return
        
        print(f"\nEmbedding successful:")
        print(f"  Original image: {args.input_image}")
        print(f"  Stego image: {result['stego_path']}")
        print(f"  PSNR: {result['psnr']:.2f} dB")
        print(f"  SSIM: {result['ssim']:.4f}")
    
    elif args.mode == 'extract':
        # Extract data from a stego image
        if not args.input_image:
            print("Error: Input stego image required for extract mode")
            return
        
        print("\n=== Step 2: Extracting Data ===")
        
        # Build extraction configuration
        extract_config = {
            'resolution': (args.image_width, args.image_height),
            'message_length': args.message_length,
            'multi_scale_decoder': args.multi_scale_decoder,
            'use_dct_decoder': args.use_dct_decoder,
            'use_error_correction': args.use_error_correction,
            'ecc_bytes': args.ecc_bytes
        }
        
        # Extract data
        output_path = os.path.join(results_dir, f"extracted_{os.path.splitext(os.path.basename(args.input_image))[0]}.txt")
        result = extract_patient_data(args.input_image, args.model_path, output_path, extract_config)
        
        if not result:
            print("Error: Extraction failed")
            return
        
        print(f"\nExtraction successful:")
        print(f"  Stego image: {args.input_image}")
        print(f"  Extracted text length: {len(result['text'])} characters")
        print(f"  Output file: {result['output_path']}")
        
        # Show part of the extracted text
        print("\nExtracted text sample:")
        print("-" * 40)
        print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Image Steganography Demo")
    
    # Operation mode
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'embed', 'extract'], 
                       help='Operation mode: sample (use dataset), embed (embed in image), extract (extract from image)')
    
    # Data paths
    parser.add_argument('--xray_dir', type=str, default=None, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, default=None, help='Directory containing patient data text files')
    parser.add_argument('--input_image', type=str, default=None, help='Input image path for embed/extract mode')
    parser.add_argument('--input_text', type=str, default=None, help='Input text or text file path for embed mode')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image_width', type=int, default=512, help='Target image width')
    parser.add_argument('--image_height', type=int, default=512, help='Target image height')
    parser.add_argument('--message_length', type=int, default=4096, help='Length of binary message')
    
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
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'sample' and (not args.xray_dir or not args.label_dir):
        parser.error("Sample mode requires --xray_dir and --label_dir")
    
    demo(args)