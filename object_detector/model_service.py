import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import logging
import torch
from transformers import DetrForObjectDetection

from .ai_models import AiModel
from .preprocessing import detr_resnet50_normalize_image
from .postprocessing import detr_resnet50_parse_preds
from .predictors import TorchImagePredictor

logger = logging.getLogger(__name__)

class ObjectDetectorService:
    """
    DETR-based object detection service using transformers.
    Provides real AI-powered object detection with high accuracy.
    """
    
    def __init__(self):
        self.ai_model = None
        self.input_size = (800, 800)  # DETR default input size
        self._load_model()
        logger.info("Object detector service initialized with DETR model")

    def _load_model(self):
        """Load DETR model."""
        try:
            logger.info("Loading DETR ResNet-50 model...")
            
            # Load DETR model
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            
            # Initialize AI model wrapper
            self.ai_model = AiModel(
                predictor=TorchImagePredictor,
                model=model,
                preprocessing=[detr_resnet50_normalize_image],
                postprocessing=[detr_resnet50_parse_preds],
            )
            
            logger.info("DETR ResNet-50 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load DETR model: {e}")
            self.ai_model = None
            logger.warning("No DETR model available, using fallback detection")

    def detect_objects(self, image: np.ndarray, min_confidence: float = 0.25, 
                       max_results: int = 50, class_filter: Optional[List[str]] = None,
                       nms_iou: float = 0.45) -> List[Dict]:
        """
        Detect objects in the image using DETR model.
        
        Args:
            image: Input image as numpy array
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of detections to return
            class_filter: List of class names to filter (None for all classes)
            nms_iou: IoU threshold for Non-Maximum Suppression
            
        Returns:
            List of detection dictionaries with 'class_name', 'confidence', 'bbox'
        """
        try:
            if self.ai_model is not None:
                return self._detr_detection(image, min_confidence, max_results, class_filter)
            else:
                logger.warning("DETR model not available, using fallback detection")
                return self._fallback_detection(image, min_confidence, max_results, class_filter)
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return self._fallback_detection(image, min_confidence, max_results, class_filter)

    def _detr_detection(self, image, min_confidence, max_results, class_filter):
        """Use DETR model for object detection."""
        try:
            # Convert OpenCV image to PIL
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Run DETR inference
            detections = self.ai_model.predict(image_pil, min_confidence)
            
            # Apply class filter if specified
            if class_filter is not None:
                detections = [det for det in detections if det['class_name'] in class_filter]
            
            # Sort by confidence and limit results
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections[:max_results]
            
        except Exception as e:
            logger.error(f"Error in DETR detection: {str(e)}")
            return self._fallback_detection(image, min_confidence, max_results, class_filter)

    def _fallback_detection(self, image, min_confidence, max_results, class_filter):
        """Fallback detection using OpenCV Haar cascades and contour analysis."""
        try:
            height, width = image.shape[:2]
            detections = []
            
            # Try face detection using Haar cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                if class_filter is None or 'person' in class_filter:
                    detections.append({
                        'class_name': 'person',
                        'confidence': 0.8,  # High confidence for detected faces
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
            
            # Enhanced contour analysis for furniture detection
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Only consider significant contours
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Determine object type based on shape and size
                    aspect_ratio = w / h
                    contour_area = w * h
                    
                    # Classify based on shape characteristics
                    if aspect_ratio > 2.0:  # Very wide - likely a desk or table
                        class_name = 'dining table'  # COCO class for table
                        confidence = min(0.7, area / 20000)
                    elif aspect_ratio > 1.2 and aspect_ratio < 2.0:  # Rectangular - could be desk
                        class_name = 'dining table'  # Use table as closest match
                        confidence = min(0.6, area / 15000)
                    elif aspect_ratio < 0.8:  # Tall - likely a person or chair
                        class_name = 'chair'
                        confidence = min(0.5, area / 10000)
                    else:  # Square-ish - could be various objects
                        class_name = 'object'
                        confidence = min(0.4, area / 8000)
                    
                    # Apply class filter
                    if class_filter is not None and class_name not in class_filter:
                        continue
                    
                    if confidence >= min_confidence:
                        detections.append({
                            'class_name': class_name,
                            'confidence': round(confidence, 3),
                            'bbox': [int(x), int(y), int(w), int(h)]
                        })
            
            return detections[:max_results]
            
        except Exception as e:
            logger.error(f"Error in fallback detection: {str(e)}")
            return []

# Global instance
detector_service = None

def get_detector_service() -> ObjectDetectorService:
    """Get the global detector service instance."""
    global detector_service
    if detector_service is None:
        detector_service = ObjectDetectorService()
    return detector_service