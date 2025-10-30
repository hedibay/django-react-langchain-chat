# cv/predictor.py
import io
import torch
from torchvision import transforms, models
from PIL import Image

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- classification model (ResNet50 pretrained on ImageNet) ----------
try:
    # newer torchvision: weights enums
    weights = models.ResNet50_Weights.DEFAULT
    cls_model = models.resnet50(weights=weights)
    cls_weights = weights
except Exception:
    # fallback for older torchvision versions
    cls_model = models.resnet50(pretrained=True)
    cls_weights = None

cls_model.eval().to(device)

# transform for ImageNet models
if cls_weights is not None:
    cls_transform = cls_weights.transforms()
else:
    cls_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# load ImageNet labels (downloaded once)
import os
LABELS_PATH = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
if not os.path.exists(LABELS_PATH):
    # download from pytorch/hub raw
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        LABELS_PATH
    )
with open(LABELS_PATH, "r") as f:
    IMAGENET_LABELS = [l.strip() for l in f.readlines()]

# ---------- detection model (Faster R-CNN pretrained on COCO) ----------
try:
    det_weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    det_model = models.detection.fasterrcnn_resnet50_fpn(weights=det_weights)
except Exception:
    det_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

det_model.eval().to(device)
# detection only needs ToTensor
det_transform = transforms.Compose([transforms.ToTensor()])

# Simple helper to run classification
def classify_image_bytes(image_bytes, topk=5):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = cls_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = cls_model(input_tensor)[0]
        probs = torch.nn.functional.softmax(out, dim=0)
        topk_probs, topk_idxs = torch.topk(probs, topk)
    results = []
    for p, idx in zip(topk_probs.cpu().numpy(), topk_idxs.cpu().numpy()):
        label = IMAGENET_LABELS[idx] if idx < len(IMAGENET_LABELS) else str(int(idx))
        results.append({"label": label, "score": float(p)})
    return results

# COCO class names (torchvision uses COCO labels; this is the common list).
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_image_bytes(image_bytes, score_threshold=0.5):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = det_transform(img).to(device)
    with torch.no_grad():
        preds = det_model([tensor])[0]   # dict with boxes, labels, scores
    boxes = preds['boxes'].cpu().numpy().tolist()
    labels = preds['labels'].cpu().numpy().tolist()
    scores = preds['scores'].cpu().numpy().tolist()
    results = []
    for box, lbl, scr in zip(boxes, labels, scores):
        if scr < score_threshold:
            continue
        name = COCO_INSTANCE_CATEGORY_NAMES[lbl] if lbl < len(COCO_INSTANCE_CATEGORY_NAMES) else str(int(lbl))
        results.append({"box": [float(v) for v in box], "label": name, "score": float(scr)})
    return results
