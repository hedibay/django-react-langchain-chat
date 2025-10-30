# cv/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("classify/", views.classify_view, name="cv-classify"),
    path("detect/", views.detect_view, name="cv-detect"),
]
