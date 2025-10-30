# Dog Breed Classification Feature

This document describes the complete implementation of the dog breed classification feature added to the Django-React project.

## Overview

The dog classification feature uses PyTorch with a pretrained EfficientNet model to classify dog breeds from uploaded images. It provides both a REST API endpoint and a React frontend interface.

## Architecture

### Backend (Django)
- **App**: `dog_classifier`
- **Model**: EfficientNet-B0 via `timm` library
- **API**: Django REST Framework
- **Endpoint**: `POST /api/dog-classify/`

### Frontend (React)
- **Component**: `DogClassifier.jsx`
- **Integration**: Added as new mode in main app
- **Features**: Image upload, preview, classification results

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `djangorestframework==3.15.2`
- `django-cors-headers==4.3.1`
- `torch==2.1.0`
- `torchvision==0.16.0`
- `timm==0.9.12`
- `Pillow==10.1.0`
- `numpy==1.24.3`

### 2. Django Configuration

The following changes were made to Django settings:

**settings.py**:
```python
INSTALLED_APPS = [
    # ... existing apps ...
    'rest_framework',
    'dog_classifier',
]

# REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# CORS settings for frontend integration
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
```

**urls.py**:
```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('dog_classifier.urls')),
]
```

### 3. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

## API Documentation

### Endpoint: `POST /api/dog-classify/`

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Fields:
  - `image` (required): Image file
  - `species` (optional): String, defaults to 'dog'

**Response**:
```json
{
  "predictions": [
    {
      "breed": "Golden Retriever",
      "probability": 85.23
    },
    {
      "breed": "Labrador Retriever", 
      "probability": 12.45
    },
    {
      "breed": "German Shepherd",
      "probability": 2.32
    }
  ],
  "species": "dog"
}
```

**Error Responses**:
- `400 Bad Request`: Missing image file or invalid image
- `500 Internal Server Error`: Classification failed

## Frontend Usage

### React Component

The `DogClassifier` component provides:

1. **Image Upload**: File input with drag-and-drop support
2. **Image Preview**: Shows uploaded image before classification
3. **Classification**: Sends image to API and displays results
4. **Results Display**: Shows top-3 predictions with probabilities
5. **Error Handling**: User-friendly error messages

### Integration

The component is integrated into the main app as a new mode:

```jsx
// In App.jsx
{mode === 'dog_classifier' && (
    <DogClassifier />
)}
```

## Model Details

### EfficientNet-B0
- **Architecture**: EfficientNet-B0 from `timm` library
- **Input Size**: 224x224 pixels
- **Preprocessing**: 
  - Resize to 224x224
  - Normalize with ImageNet stats
  - Convert to tensor
- **Output**: Logits converted to probabilities via softmax

### Class Names
The model uses a subset of ImageNet classes mapped to common dog breeds:
- Afghan hound, Beagle, Border Collie, Golden Retriever, etc.
- 100+ dog breed classes supported

## Performance Considerations

### Model Loading
- Model is loaded once at Django startup
- Reused for all classification requests
- No per-request model loading overhead

### Memory Usage
- EfficientNet-B0: ~5MB model size
- GPU acceleration if CUDA available
- CPU fallback for systems without GPU

### Response Time
- First request: ~2-3 seconds (model loading)
- Subsequent requests: ~200-500ms
- Depends on image size and hardware

## Testing

### API Testing
```bash
python test_dog_classifier.py
```

### Manual Testing
1. Start Django server: `python manage.py runserver`
2. Start React dev server: `cd frontend && npm run dev`
3. Navigate to http://localhost:3000
4. Select "Dog Breed Classifier" mode
5. Upload a dog image and test classification

## File Structure

```
Django_React_Langchain_Stream/
├── dog_classifier/           # New Django app
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py              # Model initialization
│   ├── models.py
│   ├── model_service.py     # PyTorch model service
│   ├── serializers.py       # DRF serializers
│   ├── urls.py              # API routes
│   └── views.py             # API views
├── frontend/src/
│   ├── DogClassifier.jsx     # React component
│   ├── DogClassifier.css    # Component styles
│   └── App.jsx              # Updated with new mode
└── test_dog_classifier.py   # Test script
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure PyTorch and timm are installed
   - Check CUDA availability
   - Verify model download permissions

2. **CORS Errors**
   - Ensure django-cors-headers is installed
   - Check CORS_ALLOWED_ORIGINS settings
   - Verify frontend URL matches CORS settings

3. **Image Upload Issues**
   - Check file size limits
   - Verify image format support
   - Ensure proper multipart/form-data encoding

4. **Classification Errors**
   - Check image preprocessing
   - Verify model is loaded
   - Check server logs for detailed errors

### Debug Mode

Enable Django debug logging:
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'dog_classifier': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}
```

## Future Enhancements

1. **Model Improvements**
   - Fine-tune on dog-specific dataset
   - Add more breed classes
   - Implement confidence thresholds

2. **UI Enhancements**
   - Drag-and-drop file upload
   - Batch image processing
   - Result history/export

3. **Performance**
   - Model quantization
   - Async processing
   - Caching mechanisms

4. **Features**
   - Breed information display
   - Similar breed suggestions
   - Image quality assessment
