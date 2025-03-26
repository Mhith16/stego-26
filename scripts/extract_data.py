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

from models.decoder import SteganographyDecoder, MultiScaleDecoder
from utils.error_correction import ErrorCorrection

def binary_to_text(binary_tensor, threshold=0.5):
    """
    Convert binary tensor to text
    
    Args:
        binary_tensor (torch.Tensor): Binary tensor representation
        threshold (float): Threshold for binary decision
        
    Returns:
        str: Extracted text
    """
    # Convert tensor to binary values
    binary_output = (binary_tensor > threshold).cpu().numpy().astype(np.uint8)
    
    # Process 8 bits at a time to convert to characters
    text = ""
    for i in range(0, len(binary_output), 8):
        if i + 8 <= len(binary_output):
            byte = binary_output[i:i+8]
            if np.any(byte):  # Skip if all zeros (likely padding)
                char_code = int(''.join(map(str, byte)), 2)
                if 0 < char_code < 128:  # Ensure valid ASCII
                    text += chr(char_code)
    
    # Find where the actual message ends (before padding starts)
    if '\0' in text:
        text = text.split('\0')[0]
    
    return text

def extract_patient_data(stego_path, model_path, output_path=None, config=None):
    """
    Extract patient data from a stego image
    
    Args:
        stego_path (str): Path to the stego image
        model_path (str): Directory containing trained models
        output_path (str, optional): Path to save the extracted text
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: Results including extracted text and confidence
    """
    # Use default configuration if none provided
    if config is None:
        config = {
            'resolution': (512, 512),
            'message_length': 4096,
            'multi_scale_decoder': False,
            'use_dct_decoder': True,
            'use_error_correction': True,
            'ecc_bytes': 16
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory and path if not specified
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(stego_path), 'extracted')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{os.path.splitext(os.path.basename(stego_path))[0]}_extracted_{timestamp}.txt"
        output_path = os.path.join(output_dir, filename)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize decoder
    if config['multi_scale_decoder']:
        decoder = MultiScaleDecoder(
            image_channels=1,
            message_length=config['message_length']
        ).to(device)
    else:
        decoder = SteganographyDecoder(
            image_channels=1,
            message_length=config['message_length'],
            growth_rate=32,
            num_dense_layers=6,
            with_dct=config['use_dct_decoder']
        ).to(device)
    
    # Initialize error correction if enabled
    if config['use_error_correction']:
        error_corrector = ErrorCorrection(ecc_bytes=config['ecc_bytes'])
    else:
        error_corrector = None
    
    try:
        # Load decoder model
        decoder_path = os.path.join(model_path, 'decoder.pth')
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.eval()
        print("Decoder model loaded successfully.")
        
        # Load stego image
        if isinstance(stego_path, str):
            image = Image.open(stego_path).convert('L')  # Convert to grayscale
            
            # Resize to target resolution
            image = image.resize(config['resolution'], Image.LANCZOS)
            
            # Convert to tensor
            stego_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            stego_tensor = stego_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            stego_tensor = stego_tensor.to(device)
        else:
            # If stego_path is already a tensor
            stego_tensor = stego_path.to(device)
        
        # Extract binary data
        with torch.no_grad():
            if config['multi_scale_decoder']:
                decoded_message, confidence_scores = decoder(stego_tensor)
            else:
                decoder_output = decoder(stego_tensor)
                if isinstance(decoder_output, tuple):
                    decoded_message, confidence_scores = decoder_output
                else:
                    decoded_message = decoder_output
                    confidence_scores = None
            
            # Apply error correction if enabled
            if error_corrector:
                corrected_message = error_corrector.decode_message(decoded_message)
                raw_message = decoded_message
            else:
                corrected_message = decoded_message
                raw_message = decoded_message
            
            # Calculate average confidence if available
            avg_confidence = None
            if confidence_scores is not None:
                avg_confidence = confidence_scores.mean().item()
            
            # Convert binary to text
            extracted_text = binary_to_text(corrected_message[0])
            
            # Calculate bit statistics (ones, zeros)
            bit_stats = {
                'ones': (corrected_message > 0.5).float().mean().item(),
                'zeros': (corrected_message <= 0.5).float().mean().item()
            }
            
            # Save extracted text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            print("Successfully extracted patient data:")
            print("-" * 40)
            print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            print("-" * 40)
            
            # Save metadata
            meta_path = f"{os.path.splitext(output_path)[0]}_meta.json"
            metadata = {
                'stego_image': stego_path if isinstance(stego_path, str) else "tensor_input",
                'extracted_file': output_path,
                'text_length': len(extracted_text),
                'confidence': avg_confidence,
                'bit_stats': bit_stats,
                'config': config,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return {
                'text': extracted_text,
                'confidence': avg_confidence,
                'output_path': output_path,
                'meta_path': meta_path,
                'bit_stats': bit_stats,
                'decoded_message': raw_message[0].cpu().numpy() if raw_message is not None else None,
                'corrected_message': corrected_message[0].cpu().numpy()
            }
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract patient data from a stego image")
    parser.add_argument('--image', type=str, required=True, help='Path to the stego image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained models')
    parser.add_argument('--output', type=str, help='Path to save the extracted text (optional)')
    
    # Model configuration
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution (assumes square image)')
    parser.add_argument('--message_length', type=int, default=4096, help='Maximum message length in bits')
    parser.add_argument('--multi_scale_decoder', action='store_true', help='Use multi-scale decoder')
    parser.add_argument('--use_dct_decoder', action='store_true', help='Use DCT in decoder')
    parser.add_argument('--use_error_correction', action='store_true', help='Use error correction')
    parser.add_argument('--ecc_bytes', type=int, default=16, help='Error correction bytes per chunk')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'resolution': (args.resolution, args.resolution),
        'message_length': args.message_length,
        'multi_scale_decoder': args.multi_scale_decoder,
        'use_dct_decoder': args.use_dct_decoder,
        'use_error_correction': args.use_error_correction,
        'ecc_bytes': args.ecc_bytes
    }
    
    result = extract_patient_data(args.image, args.model_path, args.output, config)
    
    if result:
        print(f"\nSummary:")
        print(f"  Stego image: {args.image}")
        print(f"  Extracted text length: {len(result['text'])} characters")
        print(f"  Confidence: {result['confidence']:.4f}" if result['confidence'] else "  Confidence: N/A")
        print(f"  Output file: {result['output_path']}")
        print(f"  Metadata: {result['meta_path']}")

if __name__ == "__main__":
    main()