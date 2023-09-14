#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import numpy as np 
import pandas as pd
import cv2
from mmseg.apis import inference_model, init_model, show_result_pyplot
# from mmseg.models import build_segmentor




config_file = 'configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py'
checkpoint_file = 'MODELS!/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')


def predict (img) :
    
    image_np = np.array(img)  # Convert image to a numpy array
    #print(image_np.shape)
    
    result = inference_model(model, image_np)
    # visualize the results in a new window
    result_img = show_result_pyplot(model, img, result)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
  
    return result_img


def drive_rgb (img : Image):
    
    bridge = CvBridge()

    image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    #print(image_rgb.shape)
    target_size = (1024,1024)
    image_rgb = cv2.resize(image_rgb, target_size)
    mod_result = predict(image_rgb)
    target_size = (1920,1080)
    mod_result = cv2.resize(mod_result, target_size)
    mod_result = bridge.cv2_to_imgmsg(mod_result, encoding='rgb8') 

    pub1.publish(mod_result)




if __name__ == '__main__' :
    
    rospy.init_node("DRIVE")
    # pub=rospy.Publisher("/zed2i/drivable_region", Image, queue_size=10)
    pub1=rospy.Publisher("/zed2i/model_result", Image, queue_size=10)
    # pub2=rospy.Publisher("/zed2i/rgb_masked_image", Image, queue_size=10)
    # pub3=rospy.Publisher("/zed2i/depth_masked_image", Image, queue_size=10)
    sub = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, callback = drive_rgb)
    # sub2 = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback = drive_depth)
    # publishing_started = False
    
    rospy.spin()
