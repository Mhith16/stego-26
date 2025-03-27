import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import json
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_analyzer import FeatureAnalysisDenseNet, FeatureAnalysisUNet
from models.encoder import SteganographyEncoder
from utils.error_correction import ErrorCorrection
from utils.metrics import compute_psnr, compute_ssim

def text_to_binary(text, max_length=4096):
    """
    Convert text to binary tensor representation
    
    Args:
        text (str): Text to convert
        max_length (int): Maximum length of binary representation
        
    Returns:
        torch.Tensor: Binary tensor representation
    """
    # Convert each character to its ASCII binary representation
    binary = ''.join([format(ord(c), '08b') for c in text])
    
    # Convert to tensor
    binary_tensor = torch.tensor([int(bit) for bit in binary], dtype=torch.float32)
    
    # Handle maximum length constraint
    if len(binary_tensor) > max_length:
        print(f"Warning: Text too long, truncating to {max_length} bits")
        binary_tensor = binary_tensor[:max_length]
    else:
        # Pad with zeros if needed
        padding = torch.zeros(max_length - len(binary_tensor))
        binary_tensor = torch.cat([binary_tensor, padding])
    
    return binary_tensor

def embed_patient_data(image_path, text, model_path, output_path=None, config=None):
    """
    Embed patient data into an X-ray image
    
    Args:
        image_path (str): Path to the X-ray image
        text (str): Patient data text or path to text file
        model_path (str): Directory containing trained models
        output_path (str, optional): Path to save the stego image
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Results including path to stego image and metrics
    """
    # Use default configuration if none provided
    if config is None:
        config = {
            'resolution': (512, 512),
            'message_length': 4096,
            'feature_analyzer': 'densenet',
            'feature_backbone': 'densenet121', 
            'embedding_strength': 0.02,
            'use_error_correction': True,
            'ecc_bytes': 16
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp for filename if not specified
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(image_path), 'stego')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_stego_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if text is a file path
    if os.path.exists(text) and os.path.isfile(text):
        with open(text, 'r', encoding='utf-8') as f:
            text_content = f.read()
            text_content = ''.join(text_content.split())
    else:
        text_content = text
    
    # Initialize models
    if config['feature_analyzer'] == 'densenet':
        feature_analyzer = FeatureAnalysisDenseNet(
            model_type=config['feature_backbone'],
            pretrained=False,
            in_channels=1
        ).to(device)
    elif config['feature_analyzer'] == 'unet':
        feature_analyzer = FeatureAnalysisUNet(
            in_channels=1,
            base_filters=64,
            pretrained_encoder=False
        ).to(device)
    else:
        raise ValueError(f"Unsupported feature analyzer type: {config['feature_analyzer']}")
    
    encoder = SteganographyEncoder(
        image_channels=1,
        embedding_strength=config['embedding_strength']
    ).to(device)
    
    # Load model weights
    try:
        feature_analyzer.load_state_dict(torch.load(os.path.join(model_path, 'feature_analyzer.pth'), map_location=device))
        encoder.load_state_dict(torch.load(os.path.join(model_path, 'encoder.pth'), map_location=device))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    
    # Set models to evaluation mode
    feature_analyzer.eval()
    encoder.eval()
    
    # Initialize error correction if enabled
    if config['use_error_correction']:
        error_corrector = ErrorCorrection(ecc_bytes=config['ecc_bytes'])
    else:
        error_corrector = None
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Resize to target resolution
        image = image.resize(config['resolution'], Image.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        image_tensor = image_tensor.to(device)
        
        # Convert text to binary
        binary_message = text_to_binary(text_content, config['message_length'])
        binary_message = binary_message.unsqueeze(0).to(device)  # Add batch dimension
        
        # Apply error correction if enabled
        if error_corrector:
            binary_message = error_corrector.encode_message(binary_message)
        
        # Generate stego image
        with torch.no_grad():
            # Generate feature weights
            feature_weights = feature_analyzer(image_tensor)
            
            # Generate stego image
            stego_image = encoder(image_tensor, binary_message, feature_weights)
        
        # Calculate metrics
        psnr = compute_psnr(image_tensor, stego_image)
        ssim = compute_ssim(image_tensor, stego_image)
        
        # Convert back to numpy and save
        stego_numpy = stego_image[0, 0].cpu().numpy() * 255
        stego_numpy = np.clip(stego_numpy, 0, 255).astype(np.uint8)
        
        # Save stego image
        cv2.imwrite(output_path, stego_numpy)
        
        print(f"Stego image created successfully: {output_path}")
        print(f"Image quality metrics - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        
        # Create metadata file with embedding information
        meta_path = f"{os.path.splitext(output_path)[0]}_meta.json"
        metadata = {
            'original_image': image_path,
            'stego_image': output_path,
            'text_length': len(text_content),
            'binary_length': len(binary_message[0]),
            'metrics': {
                'psnr': float(psnr),
                'ssim': float(ssim)
            },
            'config': config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return {
            'stego_path': output_path,
            'meta_path': meta_path,
            'psnr': float(psnr),
            'ssim': float(ssim)
        }
        
    except Exception as e:
        print(f"Error embedding data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed patient data into an X-ray image")
    parser.add_argument('--image', type=str, required=True, help='Path to the X-ray image')
    parser.add_argument('--text', type=str, required=True, help='Patient data text or path to text file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained models')
    parser.add_argument('--output', type=str, help='Path to save the stego image (optional)')
    
    # Model configuration
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution (assumes square image)')
    parser.add_argument('--message_length', type=int, default=4096, help='Maximum message length in bits')
    parser.add_argument('--feature_analyzer', type=str, default='densenet', 
                        choices=['densenet', 'unet'], help='Feature analyzer type')
    parser.add_argument('--feature_backbone', type=str, default='densenet121',
                        choices=['densenet121', 'resnet50', 'efficientnet_b0'], help='Feature backbone')
    parser.add_argument('--embedding_strength', type=float, default=0.02, help='Embedding strength')
    parser.add_argument('--use_error_correction', action='store_true', help='Use error correction')
    parser.add_argument('--ecc_bytes', type=int, default=16, help='Error correction bytes per chunk')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'resolution': (args.resolution, args.resolution),
        'message_length': args.message_length,
        'feature_analyzer': args.feature_analyzer,
        'feature_backbone': args.feature_backbone,
        'embedding_strength': args.embedding_strength,
        'use_error_correction': args.use_error_correction,
        'ecc_bytes': args.ecc_bytes
    }
    
    result = embed_patient_data(args.image, args.text, args.model_path, args.output, config)
    
    if result:
        print(f"Summary:")
        print(f"  Original image: {args.image}")
        print(f"  Stego image: {result['stego_path']}")
        print(f"  PSNR: {result['psnr']:.2f} dB")
        print(f"  SSIM: {result['ssim']:.4f}")
        print(f"  Metadata: {result['meta_path']}")