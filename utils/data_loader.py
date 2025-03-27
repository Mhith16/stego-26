import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

class XrayDataset(Dataset):
    """
    Enhanced dataset for X-ray images and patient data with multi-resolution support
    and increased message capacity.
    """
    def __init__(self, xray_dir, label_dir, transform=None, resolution=(512, 512), max_message_length=4096):
        """
        Args:
            xray_dir (str): Directory with X-ray images
            label_dir (str): Directory with patient data text files
            transform (callable, optional): Optional transform to be applied on the images
            resolution (tuple): Target image resolution (width, height) - supports 256x256, 512x512, 1024x1024
            max_message_length (int): Maximum length of binary message in bits, default 4096
        """
        self.xray_dir = xray_dir
        self.label_dir = label_dir
        self.resolution = resolution
        self.max_message_length = max_message_length
        
        # Validate resolution
        valid_resolutions = [(256, 256), (512, 512), (1024, 1024)]
        if resolution not in valid_resolutions:
            print(f"Warning: Requested resolution {resolution} is not in the recommended list {valid_resolutions}")
            print(f"Proceeding with custom resolution {resolution}")
        
        # Default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.resolution),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # Get list of files
        self.xray_files = sorted([f for f in os.listdir(xray_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        
        # Create mapping from base filename to full filename
        xray_map = {os.path.splitext(f)[0]: f for f in self.xray_files}
        label_map = {os.path.splitext(f)[0]: f for f in self.label_files}
        
        # Find common basenames
        common_names = set(xray_map.keys()) & set(label_map.keys())
        
        # Create paired files
        self.paired_files = [(xray_map[name], label_map[name]) for name in common_names]
        
        print(f"Found {len(self.paired_files)} matching X-ray and label pairs")
        print(f"Using resolution: {self.resolution}, Maximum message length: {self.max_message_length} bits")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        # Load X-ray image
        xray_file, label_file = self.paired_files[idx]
        img_path = os.path.join(self.xray_dir, xray_file)
        
        # Use PIL for better resizing quality
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(np.array(image))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Provide a fallback empty image
            image = torch.zeros((1, *self.resolution))
        
        # Load patient data
        label_path = os.path.join(self.label_dir, label_file)
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                patient_data = f.read()
                patient_data = ''.join(patient_data.split())
        except Exception as e:
            print(f"Error loading text {label_path}: {e}")
            patient_data = ""
        
        # Convert patient data to binary representation
        binary_data = self._text_to_binary(patient_data)
        
        return {
            'image': image,
            'patient_data': binary_data,
            'patient_text': patient_data,  # Keep the text for evaluation
            'file_name': xray_file,
            'original_shape': (image.shape[1], image.shape[2])  # Store original dimensions for reference
        }
    
    def _text_to_binary(self, text):
        """
        Convert text to binary tensor representation with increased capacity
        """
        # Convert each character to its ASCII binary representation
        binary = ''.join([format(ord(c), '08b') for c in text])
        
        # Convert to tensor
        binary_tensor = torch.tensor([int(bit) for bit in binary], dtype=torch.float32)
        
        # Handle variable-length messages with maximum size constraint
        if len(binary_tensor) > self.max_message_length:
            print(f"Warning: Message length ({len(binary_tensor)} bits) exceeds maximum ({self.max_message_length} bits). Truncating.")
            binary_tensor = binary_tensor[:self.max_message_length]
        else:
            # Pad with zeros if needed
            padding = torch.zeros(self.max_message_length - len(binary_tensor))
            binary_tensor = torch.cat([binary_tensor, padding])
        
        return binary_tensor.unsqueeze(0)  # Add channel dimension [1, L]

def get_data_loaders(xray_dir, label_dir, batch_size=4, resolution=(512, 512), 
                    max_message_length=4096, val_split=0.2, num_workers=4):
    """
    Create train and validation data loaders with enhanced options
    
    Args:
        xray_dir (str): Directory containing X-ray images
        label_dir (str): Directory containing patient data text files
        batch_size (int): Batch size for training
        resolution (tuple): Target image resolution (width, height)
        max_message_length (int): Maximum length of binary message
        val_split (float): Validation set ratio (0.0 to 1.0)
        num_workers (int): Number of worker threads for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Debug info
    print(f"Loading data from:")
    print(f"  X-ray directory: {xray_dir}")
    print(f"  Label directory: {label_dir}")
    
    # Check if directories exist
    if not os.path.exists(xray_dir):
        raise FileNotFoundError(f"X-ray directory not found: {xray_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    # Create dataset
    dataset = XrayDataset(xray_dir, label_dir, resolution=resolution, max_message_length=max_message_length)
    
    print(f"Dataset contains {len(dataset)} valid samples")
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found. Please check that X-ray and label files have matching names (without extension).")
    
    # Split into train/validation
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Ensure non-empty splits
    if train_size == 0 or val_size == 0:
        # If dataset is too small, use the same data for both
        train_dataset = dataset
        val_dataset = dataset
        print("Warning: Dataset too small for splitting, using same data for train and validation")
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Created train dataset with {len(train_dataset)} samples and validation dataset with {len(val_dataset)} samples")
    
    # Create data loaders with multiple workers for better performance
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader