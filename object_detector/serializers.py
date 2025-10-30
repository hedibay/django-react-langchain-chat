from rest_framework import serializers

class DetectionSerializer(serializers.Serializer):
    """Serializer for individual detection results."""
    class_name = serializers.CharField()
    confidence = serializers.FloatField()
    bbox = serializers.ListField(child=serializers.IntegerField())

class ObjectDetectionResponseSerializer(serializers.Serializer):
    """Serializer for the complete detection response."""
    detections = DetectionSerializer(many=True)
    image_size = serializers.ListField(child=serializers.IntegerField())
    processing_time = serializers.FloatField()
    model_info = serializers.DictField()
