from django.apps import AppConfig

class DogClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dog_classifier'
    
    def ready(self):
        """Initialize the model service when the app is ready."""
        try:
            from .model_service import get_classifier_service
            # Load the model at startup
            get_classifier_service()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize dog classifier service: {str(e)}")