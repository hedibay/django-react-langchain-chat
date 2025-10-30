from rest_framework import viewsets
from rest_framework import status
from rest_framework.response import Response
from PIL import Image
import logging
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .serializers import ObjectDetectionResponseSerializer
from .ai_models import AiModel
from .preprocessing import detr_resnet50_normalize_image
from .postprocessing import detr_resnet50_parse_preds
from .predictors import TorchImagePredictor
from transformers import DetrForObjectDetection

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class ObjectDetectionViewSet(viewsets.ModelViewSet):
    """
    ViewSet for object detection using DETR model.
    """
    serializer_class = ObjectDetectionResponseSerializer

    # CURRENT MODEL: DETR ResNet-101 (200MB) - High accuracy, good for furniture detection
    ai_model = AiModel(
        predictor=TorchImagePredictor,
        model=DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101"),
        preprocessing=[detr_resnet50_normalize_image],
        postprocessing=[detr_resnet50_parse_preds],
    )
    
    # ALTERNATIVE BIG MODELS YOU CAN USE:
    
    # 1. YOLO MODELS (Ultralytics) - Fast and accurate
    # from ultralytics import YOLO
    # model = YOLO('yolov8x.pt')  # 70MB - Largest YOLO, best accuracy
    # model = YOLO('yolov8l.pt')  # 50MB - Large YOLO
    # model = YOLO('yolov8m.pt')  # 25MB - Medium YOLO
    # model = YOLO('yolov5x.pt')  # 90MB - Very large YOLOv5
    # model = YOLO('yolov5l.pt')  # 50MB - Large YOLOv5
    
    # 2. DETR MODELS (Transformers) - State-of-the-art accuracy
    # model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")  # 102MB - Original
    # model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101-dc5")  # 300MB - Largest DETR
    # model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50-dc5")  # 150MB - Better for small objects
    
    # 3. FASTER R-CNN MODELS - High accuracy, slower
    # from transformers import AutoModelForObjectDetection
    # model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-101")  # 200MB
    # model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-152")  # 300MB - Largest
    
    # 4. MASK R-CNN MODELS - Best for detailed detection
    # model = AutoModelForObjectDetection.from_pretrained("facebook/maskrcnn-resnet-101")  # 250MB
    # model = AutoModelForObjectDetection.from_pretrained("facebook/maskrcnn-resnet-152")  # 350MB - Largest
    
    # 5. EFFICIENTDET MODELS - Most efficient large models
    # from transformers import AutoModelForObjectDetection
    # model = AutoModelForObjectDetection.from_pretrained("google/efficientdet-d7")  # 50MB - Largest EfficientDet
    # model = AutoModelForObjectDetection.from_pretrained("google/efficientdet-d6")  # 40MB
    # model = AutoModelForObjectDetection.from_pretrained("google/efficientdet-d5")  # 30MB
    
    # 6. SOTA MODELS (State-of-the-Art) - Latest and greatest
    # model = AutoModelForObjectDetection.from_pretrained("microsoft/swin-base-patch4-window7-224")  # 300MB - Swin Transformer
    # model = AutoModelForObjectDetection.from_pretrained("facebook/convnext-base-224")  # 200MB - ConvNeXt
    # model = AutoModelForObjectDetection.from_pretrained("facebook/pvt-base-224")  # 250MB - Pyramid Vision Transformer
    
    # RECOMMENDED FOR FURNITURE DETECTION:
    # - YOLOv8x: Best balance of speed and accuracy (70MB)
    # - DETR ResNet-101-DC5: Highest accuracy, slower (300MB)
    # - Faster R-CNN ResNet-152: Very high accuracy (300MB)
    # - Mask R-CNN ResNet-152: Best for detailed detection (350MB)

    def create(self, request, *args, **kwargs):
        """
        Detect objects in uploaded image using DETR model.
        
        Expected form data:
        - image: Image file (required)
        - min_confidence: Float (optional, default 0.25)
        - max_results: Integer (optional, default 50)
        - class_filter: Comma-separated string (optional)
        
        Returns:
        - JSON with detected objects, bounding boxes, and confidence scores
        """
        try:
            # Validate required fields
            if 'image' not in request.FILES:
                return Response(
                    {'error': 'No image file provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get the uploaded image
            image_file = request.FILES['image']
            
            # Parse optional parameters
            min_confidence = float(request.data.get('min_confidence', 0.25))
            max_results = int(request.data.get('max_results', 50))
            
            # Parse class filter
            class_filter = None
            if 'class_filter' in request.data and request.data['class_filter']:
                class_filter = [name.strip() for name in request.data['class_filter'].split(',')]
            
            # Validate and open image
            try:
                img = Image.open(image_file)
                img.verify()  # Verify it's a valid image
                image_file.seek(0)  # Reset file pointer
                img = Image.open(image_file)  # Reopen after verify
            except Exception as e:
                logger.error(f"Invalid image file: {str(e)}")
                return Response(
                    {'error': 'Invalid image file'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Predict objects using DETR model
            preds = self.ai_model.predict(img, min_confidence)
            
            # Apply class filter if specified
            if class_filter is not None:
                preds = [pred for pred in preds if pred['class_name'] in class_filter]
            
            # Limit results
            preds = preds[:max_results]
            
            # Prepare response data
            response_data = {
                'detections': preds,
                'image_size': [img.width, img.height],  # [width, height]
                'processing_time': 0.0,  # DETR processing time
                'model_info': {
                    'model_type': 'DETR ResNet-101',
                    'input_size': [800, 800],
                    'confidence_threshold': min_confidence,
                    'nms_threshold': 0.45
                }
            }
            
            # Validate response with serializer
            serializer = ObjectDetectionResponseSerializer(data=response_data)
            if serializer.is_valid():
                return Response(serializer.validated_data, status=status.HTTP_200_OK)
            else:
                logger.error(f"Serialization error: {serializer.errors}")
                return Response(
                    {'error': 'Internal server error'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return Response(
                {'error': 'Object detection failed'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )