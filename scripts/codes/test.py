#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import cv2
from torchsummary import summary
import segmentation_models_pytorch as smp
import rospy



config_file = 'configs/pspnet/pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')
'''
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import cv2

config_file = 'configs/pspnet/pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
# visualize the results in a new window
result_img = show_result_pyplot(model, img, result)
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
cv2.imshow('window', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''





def predict (img) :
    
    image_np = np.array(img)  # Convert image to a numpy array
    #print(image_np.shape)
    
    result = inference_model(model, img)
    # visualize the results in a new window
    result_img = show_result_pyplot(model, img, result)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('window', result_img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return result_img

def binary (img) :

    # Mouse callback function

    # def get_intensity(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
    #         intensity = image[y, x]
    #         print(f"Intensity at pixel ({x}, {y}): {intensity}")

    # # Create a window and set the mouse callback
    # cv2.namedWindow('Image')
    # cv2.setMouseCallback('Image', get_intensity)

    lower = 0
    upper = 5

    result = cv2.inRange(img, lower, upper)

     

    return result

    # cv2.waitKey(1)

    # while True :
    #     cv2.imshow('Image', image)
    #     cv2.imshow('result', result)
    #     # cv2.imshow('ori', img)
    #     if cv2.waitKey(1) == ord('q') : 
    #         break

    # cv2.destroyAllWindows()

def noise (img) :
    kernel_size = 5  # Adjust the kernel size as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_image, connectivity=8)
    min_area_threshold = 100  # Adjust the threshold as needed

    for label in range(1, num_labels):  # Skip background label (0)
        if stats[label, cv2.CC_STAT_AREA] < min_area_threshold:
            opened_image[labels == label] = 0  # Set small components to background

    kernel_size = 10  # Adjust the kernel size as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    return(closed_image)


def drive_rgb (img : Image):
    
    bridge = CvBridge()

    # image = np.frombuffer(img.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('ori.png',image)
    # save = cv2.imread('ori.png')
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    copy = image_rgb
    #print(image_rgb.shape)
    mod_result = predict(image_rgb)
    result = binary(mod_result)
    #result = noise(result)
    # print(mod_result.shape)
    # Convert mod_result to uint8 data type
    # mod_result = cv2.resize(image, (320, 640))
    mod_result = mod_result.astype(np.uint8)
    result = cv2.resize(result, (640, 360))
    
    global flip
    flip = cv2.bitwise_not(result)
    #print(flip.shape)
    rgb_masked_img = cv2.bitwise_and(copy, copy, mask=flip)
    
    # cv2.imshow('masked_image', masked_img)
    final = bridge.cv2_to_imgmsg(result, encoding='8UC1')   
    mod_result = bridge.cv2_to_imgmsg(mod_result, encoding='8UC1') 
    rgb_masked_img = bridge.cv2_to_imgmsg(rgb_masked_img, encoding='bgr8')
    pub.publish(final)
    pub1.publish(mod_result)
    pub2.publish(rgb_masked_img)
    global publishing_started
    
    if not publishing_started:
        publishing_started = True
        sub2 = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback=drive_depth)

    # cv2.waitKey(1)


def drive_depth (img : Image):
    
    bridge = CvBridge()
    global flip
    # image = np.frombuffer(img.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
    # print(image.shape)
    copy = image
    depth_masked_img = cv2.bitwise_and(copy, copy, mask=flip)
    # cv2.imshow('depth_mask', depth_masked_img)
    # cv2.waitKey(1)
    depth_masked_img = bridge.cv2_to_imgmsg(depth_masked_img, encoding='32FC1')
    pub3.publish(depth_masked_img)
    



if __name__ == '__main__' :
    
    rospy.init_node("DRIVE")
    pub=rospy.Publisher("/zed2i/drivable_region", Image, queue_size=10)
    pub1=rospy.Publisher("/zed2i/model_result", Image, queue_size=10)
    pub2=rospy.Publisher("/zed2i/rgb_masked_image", Image, queue_size=10)
    pub3=rospy.Publisher("/zed2i/depth_masked_image", Image, queue_size=10)
    sub = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, callback = drive_rgb)
    # sub2 = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback = drive_depth)
    publishing_started = False
    
    rospy.spin()


