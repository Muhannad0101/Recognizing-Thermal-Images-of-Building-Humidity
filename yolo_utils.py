import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def draw_bounding_boxes(image, labels, color=(0, 255, 0), thickness=2, text_color=(255, 255, 255)):
    for label in labels:
        # Unpack the label. Assuming label is a tuple in the form:
        # (class_name, x_center, y_center, width, height)
        class_name, x_center, y_center, width, height = label

        # Convert center coordinates to absolute coordinates
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]

        # Calculate the top left corner (x_min, y_min) and bottom right corner (x_max, y_max)
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw the rectangle around the detected object
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Put the class name text on the image above the bounding box
        cv2.putText(image, class_name, (x_min, y_max if y_min - 15 < 0 else y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return image


def get_class_name_by_id(class_id):
    class_names = {0: 'lamp', 1: 'moisture', 2: 'vent', 3: 'window'}
    return class_names.get(class_id, 'Unknown')


def create_segmentation_mask(labels, target_size=(640, 640)):
    mask = np.zeros(target_size, dtype=np.uint8)
    for label in labels:
        class_id, vertices = label
        if len(vertices) % 2 != 0:
            raise ValueError(f"Polygon coords should be a multiple of 2: {vertices}")

        pixel_vertices = []
        for i in range(0, len(vertices), 2):
            x = int(vertices[i] * target_size[0])
            y = int(vertices[i+1] * target_size[1])
            pixel_vertices.append([x, y])

        np_vertices = np.array([pixel_vertices], dtype=np.int32)
        cv2.fillPoly(mask, np_vertices, (255))  
    return mask


def process_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        class_name = get_class_name_by_id(class_id)  # Get the class name using the ID
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)  
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        bbox_coords = [x_center, y_center, width, height]
        # Append a tuple of the class name and bounding box coordinates
        labels.append((class_name, *bbox_coords))  # Unpack bbox_coords into the tuple
    
    return labels


def load_and_resize(image_path, target_size=(640, 640)):
    image = cv2.imread(str(image_path))
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image


def convert_bbox_format(bbox, from_format="YOLO", to_format="COCO"):
    """
    Convert the bounding box format from one type to another.

    YOLO format: [x_center, y_center, width, height]
    COCO format: [x_min, y_min, width, height]
    """
    if from_format == "YOLO" and to_format == "COCO":
        x_center, y_center, width, height = bbox
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        return [x_min, y_min, width, height]
    elif from_format == "COCO" and to_format == "YOLO":
        x_min, y_min, width, height = bbox
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)
        return [x_center, y_center, width, height]
    else:
        raise ValueError("Unsupported conversion format.")

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    intersect_min_x = max(x1_min, x2_min)
    intersect_min_y = max(y1_min, y2_min)
    intersect_max_x = min(x1_max, x2_max)
    intersect_max_y = min(y1_max, y2_max)

    intersect_area = max(intersect_max_x - intersect_min_x, 0) * max(intersect_max_y - intersect_min_y, 0)
    
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = bbox1_area + bbox2_area - intersect_area
    
    # compute the IoU
    iou = intersect_area / union_area if union_area != 0 else 0
    
    return iou

def non_maximum_suppression(bboxes, iou_threshold):
    """
    Perform non-maximum suppression given bounding boxes and an IoU threshold.
    """
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # Sort by confidence score
    nms_bboxes = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if calculate_iou(chosen_box[2], box[2]) < iou_threshold]
        nms_bboxes.append(chosen_box)

    return nms_bboxes


def visualize_samples(images_dir, labels_dir, num_samples=5):
    image_paths = list(images_dir.glob('*.jpg'))
    random_sample_paths = random.sample(image_paths, num_samples)  

    for image_path in random_sample_paths:
        file_name = image_path.stem
        label_path = labels_dir / (file_name + '.txt')

        image = load_and_resize(image_path)
        labels = process_labels(label_path)
        image_with_boxes = draw_bounding_boxes(image.copy(), labels)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title('Image with Bounding Boxes')
        plt.axis('off')

        plt.show()
