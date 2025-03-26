import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio between two images
    
    Args:
        img1 (torch.Tensor): First image tensor
        img2 (torch.Tensor): Second image tensor
        
    Returns:
        float: PSNR value in dB
    """
    if torch.is_tensor(img1) and torch.is_tensor(img2):
        # Make sure both images are on the same device
        if img1.device != img2.device:
            img1 = img1.to(img2.device)
        
        # Check dimensions and handle batch dimension
        if img1.dim() == 4 and img2.dim() == 4:  # Batch of images [B, C, H, W]
            mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])  # MSE per image in batch
            # If any MSE is 0, replace with small value to avoid inf
            mse = torch.clamp(mse, min=1e-10)
            psnr = 10 * torch.log10(1.0 / mse)
            return torch.mean(psnr).item()
        else:
            mse = torch.mean((img1 - img2) ** 2)
            if mse < 1e-10:
                return float('inf')
            return 10 * torch.log10(1.0 / mse).item()
    else:
        # Convert to numpy arrays if not tensors
        if torch.is_tensor(img1):
            img1 = img1.detach().cpu().numpy()
        if torch.is_tensor(img2):
            img2 = img2.detach().cpu().numpy()
        
        mse = np.mean((img1 - img2) ** 2)
        if mse < 1e-10:
            return float('inf')
        return 10 * np.log10(1.0 / mse)

def compute_ssim(img1, img2, data_range=1.0, multichannel=False):
    """
    Compute Structural Similarity Index between two images
    
    Args:
        img1 (torch.Tensor or numpy.ndarray): First image
        img2 (torch.Tensor or numpy.ndarray): Second image
        data_range (float): Data range of the images (difference between max and min)
        multichannel (bool): Whether images are multichannel (RGB vs grayscale)
        
    Returns:
        float: SSIM value between 0 and 1
    """
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # If batched, compute average SSIM across batch
    if img1.ndim == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            if img1.shape[1] == 1:  # Grayscale
                ssim_val = ssim(img1[i, 0], img2[i, 0], data_range=data_range)
            else:  # Multichannel
                # Move channels to last dimension for skimage
                ssim_val = ssim(
                    img1[i].transpose(1, 2, 0),
                    img2[i].transpose(1, 2, 0),
                    data_range=data_range,
                    multichannel=True
                )
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        # Single image case (no batch dimension)
        if img1.ndim == 3 and img1.shape[0] == 1:  # Single grayscale image [1, H, W]
            return ssim(img1[0], img2[0], data_range=data_range)
        elif img1.ndim == 3:  # Multichannel image [C, H, W]
            return ssim(
                img1.transpose(1, 2, 0),
                img2.transpose(1, 2, 0),
                data_range=data_range,
                multichannel=True
            )
        else:  # Single channel image [H, W]
            return ssim(img1, img2, data_range=data_range)

def compute_bit_accuracy(original_msg, decoded_msg, threshold=0.5):
    """
    Compute bit accuracy between original and decoded messages
    
    Args:
        original_msg (torch.Tensor): Original binary message
        decoded_msg (torch.Tensor): Decoded binary message
        threshold (float): Decision threshold for binary values
        
    Returns:
        float: Bit accuracy between 0 and 1
    """
    if torch.is_tensor(original_msg) and torch.is_tensor(decoded_msg):
        # Make sure both are on the same device
        if original_msg.device != decoded_msg.device:
            decoded_msg = decoded_msg.to(original_msg.device)
        
        # Binarize with threshold
        original_bits = (original_msg > threshold).float()
        decoded_bits = (decoded_msg > threshold).float()
        
        # Calculate accuracy
        correct = (original_bits == decoded_bits).float()
        
        # Handle different tensor dimensions
        if original_bits.dim() == 1:  # Single message [L]
            return torch.mean(correct).item()
        elif original_bits.dim() == 2:  # Batch of messages [B, L]
            return torch.mean(correct, dim=1).mean().item()
        elif original_bits.dim() == 3:  # Batch with channel dim [B, 1, L]
            return torch.mean(correct, dim=(1, 2)).mean().item()
        else:
            raise ValueError(f"Unsupported tensor dimension: {original_bits.dim()}")
    else:
        # Convert to numpy if not tensors
        if torch.is_tensor(original_msg):
            original_msg = original_msg.detach().cpu().numpy()
        if torch.is_tensor(decoded_msg):
            decoded_msg = decoded_msg.detach().cpu().numpy()
        
        original_bits = (original_msg > threshold).astype(float)
        decoded_bits = (decoded_msg > threshold).astype(float)
        
        return np.mean(original_bits == decoded_bits)

def compute_embedding_capacity(image_shape, message_length):
    """
    Compute bits per pixel (bpp) embedding capacity
    
    Args:
        image_shape (tuple): Shape of the image tensor [B, C, H, W]
        message_length (int): Length of binary message
        
    Returns:
        float: Bits per pixel capacity
    """
    if len(image_shape) == 4:  # [B, C, H, W]
        _, _, h, w = image_shape
    elif len(image_shape) == 3:  # [C, H, W]
        _, h, w = image_shape
    else:
        h, w = image_shape
    
    total_pixels = h * w
    bits_per_pixel = message_length / total_pixels
    
    return bits_per_pixel

def compute_character_error_rate(original_text, decoded_text):
    """
    Compute character error rate between original and decoded text
    
    Args:
        original_text (str): Original text
        decoded_text (str): Decoded text
        
    Returns:
        float: Character error rate between 0 and 1
    """
    # Levenshtein distance (edit distance)
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
    
    # If either string is empty, handle specially
    if not original_text:
        return 1.0 if decoded_text else 0.0
    
    if not decoded_text:
        return 1.0
    
    # Calculate edit distance
    distance = levenshtein(original_text, decoded_text)
    
    # Normalize by length of original text
    return distance / max(len(original_text), 1)

def compute_perfect_recovery_rate(original_msgs, decoded_msgs, threshold=0.5):
    """
    Compute the rate of perfectly recovered messages
    
    Args:
        original_msgs (torch.Tensor): Batch of original messages [B, L]
        decoded_msgs (torch.Tensor): Batch of decoded messages [B, L]
        threshold (float): Decision threshold for binary values
        
    Returns:
        float: Perfect recovery rate between 0 and 1
    """
    if not torch.is_tensor(original_msgs):
        original_msgs = torch.tensor(original_msgs)
    if not torch.is_tensor(decoded_msgs):
        decoded_msgs = torch.tensor(decoded_msgs)
    
    # Binarize
    original_bits = (original_msgs > threshold).float()
    decoded_bits = (decoded_msgs > threshold).float()
    
    # Calculate per-message exact match
    if original_msgs.dim() == 2:  # [B, L]
        perfect_match = torch.all(original_bits == decoded_bits, dim=1).float()
    elif original_msgs.dim() == 3:  # [B, 1, L]
        perfect_match = torch.all(original_bits == decoded_bits, dim=(1, 2)).float()
    else:
        raise ValueError(f"Unsupported tensor dimension: {original_msgs.dim()}")
    
    # Return perfect recovery rate
    return torch.mean(perfect_match).item()