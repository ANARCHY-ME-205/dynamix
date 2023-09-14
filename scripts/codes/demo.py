#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import numpy as np 
import pandas as pd
import cv2
from mmseg.apis import inference_model, init_model, show_result_pyplot
# from mmseg.models import build_segmentor




config_file = 'configs/pspnet/pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')
#model = build_segmentor(config_file, checkpoint_file)

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
    mod_result = predict(image_rgb)
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
