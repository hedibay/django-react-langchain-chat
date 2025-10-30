from django.db import models
from django.contrib.auth.models import User
import uuid

class ImagePrediction(models.Model):
    PREDICTION_TYPES = (
        ('classification', 'Image Classification'),
        ('detection', 'Object Detection'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='predictions/')
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    results = models.JSONField()
    confidence_scores = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']