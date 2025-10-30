from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ObjectDetectionViewSet

router = DefaultRouter()
router.register(r'detect', ObjectDetectionViewSet, basename='detect')

urlpatterns = [
    path('', include(router.urls)),
]
