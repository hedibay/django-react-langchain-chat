from rest_framework import serializers

class DogClassificationSerializer(serializers.Serializer):
    """Serializer for dog classification results."""
    breed = serializers.CharField()
    probability = serializers.FloatField()

class DogClassificationResponseSerializer(serializers.Serializer):
    """Serializer for the complete classification response."""
    predictions = DogClassificationSerializer(many=True)
    species = serializers.CharField(default='dog')
