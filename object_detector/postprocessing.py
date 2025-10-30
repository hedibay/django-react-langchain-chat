"""
DETR postprocessing functions for object detection.
"""
import torch
from typing import List, Dict


def detr_resnet50_parse_preds(outputs, min_confidence: float = 0.25) -> List[Dict]:
    """
    Parse DETR model outputs into detection format.
    
    Args:
        outputs: DETR model outputs
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detection dictionaries
    """
    # Get predictions
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes
    
    # Apply softmax to get probabilities
    probas = torch.nn.functional.softmax(logits, -1)
    
    # Get top predictions
    scores, labels = probas[..., :-1].max(-1)
    
    # Convert to numpy for easier handling
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    boxes_np = pred_boxes.cpu().numpy()
    
    detections = []
    
    # Process each detection - handle batch dimension properly
    batch_size = scores_np.shape[0] if len(scores_np.shape) > 0 else 1
    num_queries = scores_np.shape[1] if len(scores_np.shape) > 1 else len(scores_np)
    
    # Flatten arrays if needed
    if len(scores_np.shape) > 1:
        scores_flat = scores_np.flatten()
        labels_flat = labels_np.flatten()
        boxes_flat = boxes_np.reshape(-1, 4)
    else:
        scores_flat = scores_np
        labels_flat = labels_np
        boxes_flat = boxes_np
    
    # Process each detection
    for i in range(len(scores_flat)):
        if scores_flat[i] > min_confidence:
            # Get bounding box coordinates
            box = boxes_flat[i]
            x_center, y_center, width, height = box
            
            # Convert from center format to corner format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Convert to integer coordinates
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # Get class name
            class_id = int(labels_flat[i])
            class_name = get_coco_class_name(class_id)
            
            detections.append({
                'class_name': class_name,
                'confidence': round(float(scores_flat[i]), 3),
                'bbox': [x, y, w, h]
            })
    
    return detections


def get_coco_class_name(class_id: int) -> str:
    """
    Get COCO class name from class ID.
    
    Args:
        class_id: Class ID from DETR model
        
    Returns:
        Class name string
    """
    # COCO class names (91 classes including background)
    coco_classes = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    if 0 <= class_id < len(coco_classes):
        return coco_classes[class_id]
    else:
        return f"class_{class_id}"
