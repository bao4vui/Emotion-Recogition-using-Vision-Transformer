import PIL.Image
import cv2
import numpy as np
from time import time
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
from datasets import *
from transformers import ViTImageProcessor

import PIL

#Import custom module
from detection import detect_faces
from VitModelcustom import ViTForImageClassification2, string_labels

def get_model(weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    
    if device == 'cuda': cudnn.benchmark = True
    
    model = YOLO(weights)                
    model.to(device)
    return model

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    
# Initialize YOLO on GPU device.
detector = get_model('yolov8m-face/train_log/weights/best.pt')
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
VITmodel = ViTForImageClassification2.from_pretrained('./FER_VIT_model')   
total = []

first_time =  time()
cap = cv2.VideoCapture(0)

#Font setting
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

while cap.isOpened():
    success, image = cap.read()
    
    start_time = time()
    # Detect face boxes on image.
    crops, boxes, scores, cls = detect_faces(detector, [PIL.Image.fromarray(image)], box_format='xywh',th=0.4)
    #faces = face_cascade.detectMultiScale(image, 1.1, 4)
    fps = 1/(time() - start_time)
    total.append(fps)
    
    # Draw detected faces on image.
    for (left, top, right, bottom), score in zip(boxes[0], scores[0]):
    #for (left, top, right, bottom) in faces:
        #Detect face
        cv2.rectangle(image, (int(left), int(top)), (int(left+right), int(top+bottom)), (255, 0, 0), 2) 
        #Crop face to recognize emotion
        roi = image[int(top):int(top+bottom), int(left):int(left+right)]
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR)
        imgg = image_processor(roi, return_tensors='pt')
        with torch.no_grad():
            logits = VITmodel(**imgg).logits
        prob = torch.softmax(logits, dim=-1).max(dim=-1)[0].item()    
        predicted_label = logits.argmax(-1).item()  
        #Put text detection face phase
        #cv2.putText(image, f"Face {score:.2f}",(int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  
        #Put text recognize emotion phase
        cv2.putText(image, f"Emotion {f'{string_labels[predicted_label]} {prob:.2f}'}",(int(left), int(top) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)        
        #Other text
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(image, f'Avg. FPS: {np.mean(total):.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(image, f'Max. FPS: {max(total):.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(image, f'Min. FPS: {min(total):.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
    
    
    cv2.putText(image, f'{time()-first_time:.2f}s', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    cv2.imshow('DEMO', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()