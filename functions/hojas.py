import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def model_hojas(path,model,model2):
    img_rgb = cv2.imread(path,1)
# Predict with the model
    #results = model(path)  # predict on an image
    result_detection = model2(path,iou=0.4, conf=0.47) #conf=0.47
    list_direc = []
    detection_id = 0

# Realiza la detección del primer modelo
    for result in result_detection:
        boxes = result.boxes
        
        for box in boxes:
            detection_id += 1
            # Extraer las coordenadas de la caja delimitadora
            y_1 = int(box.xyxy[0][0])
            x_1 = int(box.xyxy[0][1])
            y_2 = int(box.xyxy[0][2])
            x_2 = int(box.xyxy[0][3])

            roi = img_rgb[x_1:x_2, y_1:y_2, :]

            # Aplicamos el segundo modelo de segmentación en la ROI sin cortar la imagen original
            results_segment = model(roi, conf=0.3)
            count_leaf = len(results_segment[0].boxes)
            list_direc.append({ 'x_2': x_2, 
                                'delt': y_1,
                                'id': detection_id,
                                'count_leafl': count_leaf})
            
            
    response = {'result_detection':result_detection, 
                'list_direc':list_direc }
    return response