"""
AI model wrapper for object detection.
"""
from typing import Any, List
from PIL import Image


class AiModel:
    """
    AI model wrapper for object detection.
    """
    
    def __init__(self, predictor, model, preprocessing=None, postprocessing=None):
        self.predictor = predictor
        self.model = model
        self.preprocessing = preprocessing or []
        self.postprocessing = postprocessing or []
        
        # Initialize predictor with model and processing functions
        self.predictor_instance = self.predictor(
            model=self.model,
            preprocessing=self.preprocessing,
            postprocessing=self.postprocessing
        )
    
    def predict(self, image: Image.Image, min_confidence: float = 0.25) -> List[dict]:
        """
        Predict objects in image.
        
        Args:
            image: PIL Image
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detection dictionaries
        """
        return self.predictor_instance.predict(image, min_confidence)

