import numpy as np
import torch
from reedsolo import RSCodec, ReedSolomonError

class ErrorCorrection:
    """
    Error correction coding using Reed-Solomon codes for robust data embedding and extraction.
    
    Reed-Solomon codes are particularly effective for burst errors (consecutive errors),
    which makes them suitable for image steganography where regions may be corrupted.
    """
    
    def __init__(self, ecc_bytes=16, message_chunk_size=255):
        """
        Initialize the error correction system
        
        Args:
            ecc_bytes (int): Number of error correction bytes per chunk
            message_chunk_size (int): Maximum size of each chunk in bytes (including ecc)
                                     Reed-Solomon limitation is 255 bytes total per chunk
        """
        # Ensure we don't exceed Reed-Solomon's GF(2^8) field size limitation
        self.message_chunk_size = min(message_chunk_size, 255)
        self.ecc_bytes = min(ecc_bytes, self.message_chunk_size - 1)
        
        # Data bytes per chunk (255 - ecc_bytes)
        self.data_bytes_per_chunk = self.message_chunk_size - self.ecc_bytes
        
        # Initialize Reed-Solomon codec
        self.rs_codec = RSCodec(self.ecc_bytes)
        
        print(f"Initialized Reed-Solomon error correction: {self.ecc_bytes} ECC bytes per {self.message_chunk_size} byte chunk")
        print(f"Can correct up to {self.ecc_bytes//2} byte errors per chunk")
    
    def encode_message(self, binary_tensor):
        """
        Encode binary message tensor with Reed-Solomon error correction
        
        Args:
            binary_tensor: torch.Tensor containing binary message (1D or 2D with batch dim)
            
        Returns:
            torch.Tensor: Encoded message with error correction
        """
        if binary_tensor.dim() == 2:
            # Handle batch dimension
            return torch.stack([self.encode_message(msg) for msg in binary_tensor])
        
        # Convert binary tensor to bytes
        binary_np = binary_tensor.cpu().numpy().astype(np.uint8)
        bits_len = len(binary_np)
        
        # Convert binary array to bytes
        # First ensure length is multiple of 8
        padding_bits = (8 - (bits_len % 8)) % 8
        if padding_bits > 0:
            binary_np = np.pad(binary_np, (0, padding_bits))
        
        # Reshape and pack bits into bytes
        bytes_data = np.packbits(binary_np)
        
        # Store original bit length for later reconstruction
        bits_length_bytes = bits_len.to_bytes(4, byteorder='big')
        
        # Prepend the bit length to the message
        message_bytes = bits_length_bytes + bytes_data.tobytes()
        
        # Chunk the message and encode each chunk
        encoded_chunks = []
        for i in range(0, len(message_bytes), self.data_bytes_per_chunk):
            chunk = message_bytes[i:i+self.data_bytes_per_chunk]
            encoded_chunk = self.rs_codec.encode(chunk)
            encoded_chunks.append(encoded_chunk)
        
        # Combine chunks and convert back to binary
        encoded_bytes = b''.join(encoded_chunks)
        encoded_bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))
        
        # Convert back to tensor
        encoded_tensor = torch.from_numpy(encoded_bits).float()
        
        return encoded_tensor
    
    def decode_message(self, binary_tensor, confidence_scores=None):
        """
        Decode binary message tensor with Reed-Solomon error correction
        
        Args:
            binary_tensor: torch.Tensor containing binary message (1D or 2D with batch dim)
            confidence_scores: Optional confidence scores for each bit (used for soft decision)
            
        Returns:
            torch.Tensor: Decoded and error-corrected binary message
        """
        if binary_tensor.dim() == 2:
            # Handle batch dimension
            return torch.stack([self.decode_message(binary_tensor[i], 
                                                   None if confidence_scores is None else confidence_scores[i]) 
                               for i in range(binary_tensor.size(0))])
        
        # Threshold the binary tensor if not already binary
        binary_np = (binary_tensor > 0.5).cpu().numpy().astype(np.uint8)
        
        # Use confidence scores to prioritize correction (if available)
        if confidence_scores is not None:
            # This is an advanced technique: we can use confidence to guide the error correction
            # Implementable as a future enhancement
            pass
        
        # Convert binary array to bytes
        try:
            # Pack bits into bytes
            bytes_data = np.packbits(binary_np)
            
            # Chunk the message and decode each chunk
            decoded_chunks = []
            for i in range(0, len(bytes_data), self.message_chunk_size):
                chunk = bytes_data[i:i+self.message_chunk_size].tobytes()
                try:
                    # Attempt to decode and correct errors
                    decoded_chunk, _ = self.rs_codec.decode(chunk)
                    decoded_chunks.append(decoded_chunk)
                except ReedSolomonError:
                    # If chunk is too corrupted, append zeros or try fallback strategy
                    print(f"Warning: Could not correct errors in chunk {i//self.message_chunk_size}")
                    # Append zeros or partial correction as fallback
                    decoded_chunks.append(bytes(self.data_bytes_per_chunk))
            
            # Combine chunks
            decoded_bytes = b''.join(decoded_chunks)
            
            # Extract the original bit length (first 4 bytes)
            original_bit_length = int.from_bytes(decoded_bytes[:4], byteorder='big')
            
            # Convert remaining bytes back to binary
            message_bytes = decoded_bytes[4:]
            decoded_bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))
            
            # Trim to original length
            decoded_bits = decoded_bits[:original_bit_length]
            
            # Convert back to tensor
            decoded_tensor = torch.from_numpy(decoded_bits).float()
            
            # Ensure we return the correct message length (matching input expectation)
            if len(decoded_tensor) < binary_tensor.size(0):
                decoded_tensor = torch.cat([decoded_tensor, 
                                          torch.zeros(binary_tensor.size(0) - len(decoded_tensor))])
            elif len(decoded_tensor) > binary_tensor.size(0):
                decoded_tensor = decoded_tensor[:binary_tensor.size(0)]
                
            return decoded_tensor
            
        except Exception as e:
            print(f"Error in Reed-Solomon decoding: {e}")
            # Return original tensor on decoding failure
            return binary_tensor
    
    def get_encoded_length(self, original_length):
        """
        Calculate the length of the encoded message given the original length in bits
        
        Args:
            original_length (int): Original message length in bits
            
        Returns:
            int: Length of encoded message in bits
        """
        # Convert bits to bytes (rounding up)
        original_bytes = (original_length + 7) // 8
        
        # Add 4 bytes for storing the original bit length
        message_bytes = original_bytes + 4
        
        # Calculate how many chunks we need
        chunks = (message_bytes + self.data_bytes_per_chunk - 1) // self.data_bytes_per_chunk
        
        # Calculate total encoded bytes (each chunk is message_chunk_size)
        encoded_bytes = chunks * self.message_chunk_size
        
        # Convert back to bits
        encoded_bits = encoded_bytes * 8
        
        return encoded_bits


# Helper functions for redundant encoding
def apply_redundant_encoding(binary_message, redundancy=3):
    """
    Apply redundant encoding to critical bits in the message
    
    Args:
        binary_message: torch.Tensor containing binary message
        redundancy: Number of times to repeat each bit
        
    Returns:
        torch.Tensor: Message with redundant encoding
    """
    # Repeat each bit redundancy times
    if binary_message.dim() == 2:  # Batch dimension
        encoded = binary_message.repeat_interleave(redundancy, dim=1)
    else:
        encoded = binary_message.repeat_interleave(redundancy)
    return encoded

def decode_redundant_encoding(binary_message, redundancy=3):
    """
    Decode a message with redundant encoding using majority voting
    
    Args:
        binary_message: torch.Tensor containing encoded binary message
        redundancy: Number of times each bit was repeated
        
    Returns:
        torch.Tensor: Decoded message
    """
    if binary_message.dim() == 1:
        # Reshape to group redundant bits
        reshaped = binary_message.view(-1, redundancy)
        # Perform majority voting (sum > redundancy/2)
        decoded = (reshaped.sum(dim=1) > redundancy/2).float()
        return decoded
    elif binary_message.dim() == 2:  # Batch dimension
        batch_size = binary_message.size(0)
        reshaped = binary_message.view(batch_size, -1, redundancy)
        decoded = (reshaped.sum(dim=2) > redundancy/2).float()
        return decoded
    else:
        raise ValueError(f"Unsupported tensor dimensions: {binary_message.dim()}")