from django.urls import path
from . import views

urlpatterns = [
    path('dog-classify/', views.DogClassificationView.as_view(), name='dog-classify'),
]


