import cv2
import os
import numpy as np

# Filtered classes with their names and IDs
FILTERED_CLASSES = {
    0: "Hardhat",
    2: "NO-Hardhat",
    4: "NO-Safety Vest",
    7: "Safety Vest"
}

# For tracking False Positive and False Negative per class
class_fp = {0: 0, 2: 0, 4: 0, 7: 0}
class_fn = {0: 0, 2: 0, 4: 0, 7: 0}

def read_labels(file_path):
    with open(file_path, 'r') as f:
        labels = [line.strip().split()[:5] for line in f.readlines()]  # Take only the first 5 values
    return np.array(labels, dtype=float)

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (xmin, ymin, xmax, ymax)
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    # Compute the intersection area
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)

    if ix_min >= ix_max or iy_min >= iy_max:
        return 0.0  # No overlap

    intersection_area = (ix_max - ix_min) * (iy_max - iy_min)

    # Compute the union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def evaluate_frame(pred_file, gt_file, frame_path, iou_threshold=0.4):
    # Read the predicted and ground truth labels
    pred_labels = read_labels(pred_file)
    gt_labels = read_labels(gt_file)

    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    # Load the frame image for drawing
    frame_img = cv2.imread(frame_path)

    # Get the classes present in the ground truth that are in the filtered classes
    gt_classes = {int(gt[0]) for gt in gt_labels if int(gt[0]) in FILTERED_CLASSES}

    # For each predicted box, check if there's a matching ground truth box
    for pred in pred_labels:
        pred_class = int(pred[0])

        # Check if the predicted class is in the filtered classes
        if pred_class not in FILTERED_CLASSES:
            continue  # Skip predictions not in filtered classes

        # Only evaluate for classes that are in the ground truth
        if pred_class not in gt_classes:
            # If class is in filtered classes but not in GT, it's a False Positive
            fp += 1
            class_fp[pred_class] += 1

            # Draw bounding box for False Positive
            x, y, w, h = pred[1:5]
            x1, y1 = int((x - w / 2) * frame_img.shape[1]), int((y - h / 2) * frame_img.shape[0])
            x2, y2 = int((x + w / 2) * frame_img.shape[1]), int((y + h / 2) * frame_img.shape[0])
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for FP

            # Add label "FP: [Class Name]"
            label_position = (x1, y1 - 10)  # Positioning the label above the box
            class_name = FILTERED_CLASSES.get(pred_class, "Unknown")
            cv2.putText(frame_img, f"FP: {class_name}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            continue

        pred_box = pred[1:5]  # x_center, y_center, width, height

        matched = False
        for idx, gt in enumerate(gt_labels):
            if idx not in matched_gt and int(gt[0]) == pred_class:
                gt_box = gt[1:5]  # x_center, y_center, width, height
                iou = compute_iou(pred_box, gt_box)

                if iou >= iou_threshold:  # Match if IoU is above the threshold
                    tp += 1
                    matched_gt.add(idx)
                    matched = True
                    break

        if not matched:
            fn += 1
            class_fn[pred_class] += 1

            # Draw bounding box for False Negative
            x, y, w, h = pred[1:5]
            x1, y1 = int((x - w / 2) * frame_img.shape[1]), int((y - h / 2) * frame_img.shape[0])
            x2, y2 = int((x + w / 2) * frame_img.shape[1]), int((y + h / 2) * frame_img.shape[0])
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for FN

            # Add label "FN: [Class Name]"
            label_position = (x1, y1 - 10)  # Positioning the label above the box
            class_name = FILTERED_CLASSES.get(pred_class, "Unknown")
            cv2.putText(frame_img, f"FN: {class_name}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # False Negatives are the ground truth boxes not matched
    fn = len(gt_labels) - len(matched_gt)

    # Save the frame image with bounding boxes
    output_path = os.path.join("output_frames", os.path.basename(frame_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame_img)

    return tp, fp, fn

# Loop through all frames to evaluate
frame_dir = "./data/images"
gt_dir = "./data/labels/ground_truth/"
pred_dir = "./data/labels/predictions/"

tp, fp, fn = 0, 0, 0

for frame in os.listdir(frame_dir):
    frame_name = frame.split(".")[0]  # Extract the frame number (e.g., frame_0000)

    pred_file = os.path.join(pred_dir, f"{frame_name}.txt")
    gt_file = os.path.join(gt_dir, f"{frame_name}.txt")
    frame_path = os.path.join(frame_dir, frame)

    if os.path.exists(pred_file) and os.path.exists(gt_file):
        tp_frame, fp_frame, fn_frame = evaluate_frame(pred_file, gt_file, frame_path)
        tp += tp_frame
        fp += fp_frame
        fn += fn_frame

# Final evaluation metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Print False Positives and False Negatives per class with class names
print("\nFalse Positives per Class:")
for class_id, count in class_fp.items():
    class_name = FILTERED_CLASSES.get(class_id, "Unknown")
    print(f"Class {class_name} ({class_id}): {count} False Positives")

print("\nFalse Negatives per Class:")
for class_id, count in class_fn.items():
    class_name = FILTERED_CLASSES.get(class_id, "Unknown")
    print(f"Class {class_name} ({class_id}): {count} False Negatives")
