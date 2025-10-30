"""
Predictor classes for object detection models.
"""
import torch
from typing import Any, List
from PIL import Image


class TorchImagePredictor:
    """
    PyTorch-based image predictor for object detection.
    """
    
    def __init__(self, model, preprocessing, postprocessing):
        self.model = model
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.model.eval()  # Set to evaluation mode
    
    def predict(self, image: Image.Image, min_confidence: float = 0.25) -> List[dict]:
        """
        Predict objects in image.
        
        Args:
            image: PIL Image
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detection dictionaries
        """
        with torch.no_grad():
            # Preprocess image
            if self.preprocessing:
                for preprocess_func in self.preprocessing:
                    image_tensor = preprocess_func(image)
            
            # Run inference
            outputs = self.model(image_tensor)
            
            # Postprocess results
            if self.postprocessing:
                for postprocess_func in self.postprocessing:
                    detections = postprocess_func(outputs, min_confidence)
            
            return detections

