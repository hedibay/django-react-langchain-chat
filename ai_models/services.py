import torch
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time
from transformers import pipeline
import base64
import io

class ImageClassificationService:
    def __init__(self):
        # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # Define preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load ImageNet class labels
        with open('ai_models/imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def classify_image(self, image):
        """
        Classify an image and return top 5 predictions
        """
        start_time = time.time()
        
        if isinstance(image, str):  # base64 string
            image = self.decode_base64_image(image)
        
        # Preprocess image
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        results = []
        for i in range(5):
            class_name = self.classes[top5_catid[i]]
            confidence = float(top5_prob[i])
            results.append({
                'class': class_name,
                'confidence': confidence
            })
        
        processing_time = time.time() - start_time
        
        return {
            'predictions': results,
            'processing_time': processing_time
        }

    def decode_base64_image(self, base64_string):
        """Decode base64 string to PIL Image"""
        image_data = base64.b64decode(base64_string.split(',')[1])
        return Image.open(io.BytesIO(image_data)).convert('RGB')


class ObjectDetectionService:
    def __init__(self):
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # nano version for faster inference
        
    def detect_objects(self, image):
        """
        Detect objects in an image using YOLOv8
        """
        start_time = time.time()
        
        if isinstance(image, str):  # base64 string
            image = self.decode_base64_image(image)
            
        # Convert PIL to cv2 format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = self.model(cv_image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': {
                            'x1': x1, 'y1': y1,
                            'x2': x2, 'y2': y2
                        }
                    })
        
        processing_time = time.time() - start_time
        
        return {
            'detections': detections,
            'processing_time': processing_time,
            'image_size': {'width': cv_image.shape[1], 'height': cv_image.shape[0]}
        }

    def decode_base64_image(self, base64_string):
        """Decode base64 string to PIL Image"""
        image_data = base64.b64decode(base64_string.split(',')[1])
        return Image.open(io.BytesIO(image_data)).convert('RGB')