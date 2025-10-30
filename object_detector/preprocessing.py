"""
DETR preprocessing functions for object detection.
"""
import torch
from PIL import Image
import torchvision.transforms as T


def detr_resnet50_normalize_image(image: Image.Image) -> torch.Tensor:
    """
    Normalize image for DETR ResNet-50 model.
    
    Args:
        image: PIL Image
        
    Returns:
        Normalized tensor ready for DETR model
    """
    # Define the same transforms as used in DETR training
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor

