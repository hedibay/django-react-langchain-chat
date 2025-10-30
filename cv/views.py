# cv/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import predictor

@csrf_exempt
@require_POST
def classify_view(request):
    imgfile = request.FILES.get("image")
    if not imgfile:
        return JsonResponse({"error":"no image uploaded"}, status=400)
    image_bytes = imgfile.read()
    try:
        results = predictor.classify_image_bytes(image_bytes, topk=5)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"predictions": results})

@csrf_exempt
@require_POST
def detect_view(request):
    imgfile = request.FILES.get("image")
    if not imgfile:
        return JsonResponse({"error":"no image uploaded"}, status=400)
    image_bytes = imgfile.read()
    try:
        threshold = float(request.POST.get("threshold", 0.5))
    except Exception:
        threshold = 0.5
    try:
        results = predictor.detect_image_bytes(image_bytes, score_threshold=threshold)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"predictions": results})
