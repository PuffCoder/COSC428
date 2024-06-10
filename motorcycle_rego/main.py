import os
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

results = {}
mot_tracker = Sort()

# 加载模型
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('/Users/david/HProject/Demo/motorcycle_rego/model/license_plate_detector.pt')

# 加载视频
# cap = cv2.VideoCapture('/Users/david/HProject/Demo/testSet/sample.mp4')
cap = cv2.VideoCapture('/Users/david/HProject/Demo/testSet/engroad.MOV')

vehicles = [2, 3, 5, 7]  # 定义车辆类别ID

# 读取视频帧
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:    
        results[frame_nmr] = {}
        # 检测车辆
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        # 跟踪车辆        
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        # 检测车牌
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
        
            # 分配车牌给车辆
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:
                # 裁剪车牌区域
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
        
                # 处理车牌图像
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                # 读取车牌号码
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                
# 创建目录（如果不存在）
output_dir = '/Users/david/HProject/Demo/excel_result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 写入结果到CSV文件
write_csv(results, os.path.join(output_dir, 'test.csv'))