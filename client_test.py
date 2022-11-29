import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torch.backends.cudnn as cudnn

from utils.torch_utils import select_device
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

#--------------------------load model--------------------------
imgsz = (640, 640)
device = select_device()

#model for plate detect
P_model = DetectMultiBackend('./custom_model/P_model.pt', device=device, dnn=False, data='./data/plate_dataset.yaml', fp16=False)
P_model.warmup(imgsz=(1 if P_model.pt else 1, 3, *imgsz))

#model for character recognition
L_model = DetectMultiBackend('./custom_model/L_model.pt', device=device, dnn=False, data='./data/plate_dataset.yaml', fp16=False)
L_model.warmup(imgsz=(1 if L_model.pt else 1, 3, *imgsz))

letters = ["0", "1", "2", "3", "4", "5","6", "7","8", "9","A", "B","C", "D","E", "F","G", "H", "I", "J","K", "L","M", "N", "", "P","Q", "R","S", "T","U", "V","W", "X","Y", "Z"]
crop_frame = []

#--------------------------def function--------------------------
def plate_yoloPred(frame):
    #frame = cv2.resize(frame, (640, 480))

    img0 = frame

    img = letterbox(img0, imgsz, P_model.stride, P_model.pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
    im = torch.from_numpy(im).to(device)
    im = im.half() if P_model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]

    pred = P_model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.5, 0.25, 0, False, max_det=1)

    return pred

def license_yoloPred(frame):
    #frame = cv2.resize(frame, (640, 480))

    img0 = frame

    img = letterbox(img0, imgsz, L_model.stride, L_model.pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
    im = torch.from_numpy(im).to(device)
    im = im.half() if L_model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]

    pred = L_model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.5, 0.45, None, True, max_det=8)

    return pred

def convert_LicenseToWords(frame):
    global crop_frame

    #Step0. Preprocess (equal scaling)
    h, w, c = frame.shape
    n_h = int((640*h)/w)
    frame = cv2.resize(frame, (640, n_h))

    #Step1. Get the position of license plate (using "plate_yoloPred" function)
    pred = plate_yoloPred(frame)
    pred = pred[0].tolist()

    #Step2. Crop image and resize
    if len(pred)>0:
        for i in range(len(pred)):
            pos = pred[i]
            
            for j in range(len(pos)):
                if pos[j]<0:
                    pos[j] = 0

            #frame = cv2.rectangle(frame, (int(pos[0]), int(pos[1])), (int(pos[2]), int(pos[3])), (0, 0, 255), 1)
            crop_frame = frame[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2])]
    else:
        return "no plate!"

    
    h, w, c = crop_frame.shape
    n_h = int((640*h)/w)
    crop_frame = cv2.resize(crop_frame, (640, n_h))

    #Step3. Optical Character Recognition (using "license_yoloPred" function)
    pred = license_yoloPred(crop_frame)
    pred = pred[0].tolist()
    pred.sort()

    number = ""
    if len(pred)>0:
        for i in range(len(pred)):
            pos = pred[i]
            word = letters[int(pos[5])]
            ww = int(pos[2])-int(pos[0])

            if ww > 55:
                crop_frame = cv2.rectangle(crop_frame, (int(pos[0]), int(pos[1])), (int(pos[2]), int(pos[3])), (0, 0, 255), 1)
                number += word

        crop_frame = cv2.putText(crop_frame, number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        return "no letters!"

    return number

def IoU(box1, box2):
    Area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    Area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    if ( min(box1[2], box2[2]) - max(box1[0], box2[0]) < 0) or (min(box1[3], box2[3]) - max(box1[1], box2[1])  <0 ):
        return 0 
    Intersection =(min(box1[2], box2[2]) - max(box1[0], box2[0]))*(min(box1[3], box2[3]) - max(box1[1], box2[1]))
    #Union = Area1 + Area2 - Intersection
    Union = Area1
    return Intersection/Union

#
import requests
import socket
import json

HOST = '127.0.0.1'
PORT = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    #outdata = input('please input message: ')
    #print('send: ' + outdata)
    #s.send(outdata.encode())

    print("waiting")
    indata = s.recv(1024)
    rs = indata.decode()
    
    if rs == "start\n":
        print("img received")
        today = time.strftime("%Y-%m-%d", time.localtime())
        #path = "/home/lpr/openalpr/logs/"+str(today)+"/1.jpg"
        path = "/home/lpr/openalpr/testdata/photo.jpg"
        frame = cv2.imread(path)
        number = convert_LicenseToWords(frame)
        print("imgPath:", path)
        obj = { "results": [{ "candidates": [{ "plate":str(number) }] }] }
        se = json.dumps(obj)
        print("obj:", obj)
        print("se:", se)
        print("se_en:", bytes(se,encoding="utf-8"))
        print(number)

        s.sendall(bytes(se,encoding="utf-8"))
        se = "\n"
        s.send(se.encode())


    else:
        print('recv: ' + rs)
        se = "nono"
        s.send(se.encode())

    if len(indata) == 0:
        s.close()
        print('server closed connection.')
        break
