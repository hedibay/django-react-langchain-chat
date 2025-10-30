from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image
import logging
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .model_service import get_classifier_service
from .serializers import DogClassificationResponseSerializer

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class DogClassificationView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        try:
            if 'image' not in request.FILES:
                return Response(
                    {'error': 'No image file provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            image_file = request.FILES['image']
            species = request.data.get('species', 'dog')
            
            try:
                image = Image.open(image_file)
                image.verify()
                image_file.seek(0)
                image = Image.open(image_file)
                
            except Exception as e:
                logger.error(f"Invalid image file: {str(e)}")
                return Response(
                    {'error': 'Invalid image file'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            classifier = get_classifier_service()
            predictions = classifier.predict(image, top_k=3)
            
            response_data = {
                'predictions': predictions,
                'species': species
            }
            
            serializer = DogClassificationResponseSerializer(data=response_data)
            if serializer.is_valid():
                return Response(serializer.validated_data, status=status.HTTP_200_OK)
            else:
                logger.error(f"Serialization error: {serializer.errors}")
                return Response(
                    {'error': 'Internal server error'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return Response(
                {'error': 'Classification failed'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )