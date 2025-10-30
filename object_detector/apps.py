from django.apps import AppConfig

class ObjectDetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'object_detector'
    
    def ready(self):
        """Initialize the detector service when the app is ready."""
        try:
            from .model_service import get_detector_service
            # Load the detector at startup
            get_detector_service()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize object detector service: {str(e)}")