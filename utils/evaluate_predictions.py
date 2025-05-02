import cv2
import os
import numpy as np

FILTERED_CLASSES = {
    0: "Hardhat",
    2: "NO-Hardhat",
    4: "NO-Safety Vest",
    7: "Safety Vest"
}

def read_labels(file_path):
    with open(file_path, 'r') as f:
        labels = [line.strip().split()[:5] for line in f.readlines()]
    return np.array(labels, dtype=float)

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)
    if ix_min >= ix_max or iy_min >= iy_max:
        return 0.0
    intersection_area = (ix_max - ix_min) * (iy_max - iy_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    return intersection_area / (box1_area + box2_area - intersection_area)

def evaluate_frame(pred_file, gt_file, frame_path, iou_threshold=0.4, class_fp=None, class_fn=None):
    pred_labels = read_labels(pred_file)
    gt_labels = read_labels(gt_file)

    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    frame_img = cv2.imread(frame_path)
    gt_classes = {int(gt[0]) for gt in gt_labels if int(gt[0]) in FILTERED_CLASSES}

    for pred in pred_labels:
        pred_class = int(pred[0])
        if pred_class not in FILTERED_CLASSES:
            continue
        if pred_class not in gt_classes:
            fp += 1
            if class_fp is not None:
                class_fp[pred_class] += 1
            continue
        pred_box = pred[1:5]
        matched = False
        for idx, gt in enumerate(gt_labels):
            if idx not in matched_gt and int(gt[0]) == pred_class:
                gt_box = gt[1:5]
                if compute_iou(pred_box, gt_box) >= iou_threshold:
                    tp += 1
                    matched_gt.add(idx)
                    matched = True
                    break
        if not matched:
            fn += 1
            if class_fn is not None:
                class_fn[pred_class] += 1
    fn = len(gt_labels) - len(matched_gt)
    return tp, fp, fn

def evaluate_all_frames(frame_dir, gt_dir, pred_dir, iou_threshold=0.4):
    tp, fp, fn = 0, 0, 0
    class_fp = {k: 0 for k in FILTERED_CLASSES}
    class_fn = {k: 0 for k in FILTERED_CLASSES}

    for frame in os.listdir(frame_dir):
        frame_name = os.path.splitext(frame)[0]
        pred_file = os.path.join(pred_dir, f"{frame_name}.txt")
        gt_file = os.path.join(gt_dir, f"{frame_name}.txt")
        frame_path = os.path.join(frame_dir, frame)

        if os.path.exists(pred_file) and os.path.exists(gt_file):
            tp_frame, fp_frame, fn_frame = evaluate_frame(pred_file, gt_file, frame_path, iou_threshold, class_fp, class_fn)
            tp += tp_frame
            fp += fp_frame
            fn += fn_frame

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1Score": f"{f1_score:.2f}",
        "FPperClass": {
            FILTERED_CLASSES[k]: v for k, v in class_fp.items()
        },
        "FNperClass": {
            FILTERED_CLASSES[k]: v for k, v in class_fn.items()
        }
    }

    return metrics