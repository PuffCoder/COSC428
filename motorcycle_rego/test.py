from ultralytics import YOLO

# load models
coco_model = YOLO('yolov8n-seg.pt')
license_plate_detector = YOLO