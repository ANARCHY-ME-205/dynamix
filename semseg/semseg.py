#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import numpy as np 
import cv2
from mmseg.apis import inference_model, init_model, show_result_pyplot

config_file = '/home/tamoghna/catkin_ws/src/dynamix/semseg/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py'
checkpoint_file = '/home/tamoghna/catkin_ws/src/dynamix/semseg/MODELS!/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')


def predict (img) :
    
    image_np = np.array(img)  
    #print(image_np.shape)
    
    result = inference_model(model, image_np)
    
    result_img = show_result_pyplot(model, img, result)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
  
    return result_img

def binary (img) :

    
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #ori = img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Mouse callback function
    # def get_intensity(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
    #         intensity = img[y, x]
    #         print(f"Intensity at pixel ({x}, {y}): {intensity}")

    
    # Create a window and set the mouse callback
    # cv2.namedWindow('Image')
    # cv2.setMouseCallback('Image', get_intensity)

    lower = np.array([150, 128, 128])
    upper = np.array([150, 128, 128])

    result = cv2.inRange(img, lower, upper)
    

    return result

    # cv2.waitKey(1)

    # while True :
    #     cv2.imshow('Image', img)
    #     cv2.imshow('result', result)
    #     cv2.imshow('ori', ori)
    #     if cv2.waitKey(1) == ord('q') : 
    #         break

    # cv2.destroyAllWindows()

def drive_rgb (img : Image):
    
    bridge = CvBridge()

    image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    copy = image_rgb
    #print(image_rgb.shape)
    target_size = (1024,1024)
    image_rgb = cv2.resize(image_rgb, target_size)
    mod_result = predict(image_rgb)
    mod_result = cv2.resize(mod_result, (copy.shape[1],copy.shape[0]))
    result = binary(mod_result)
    global flip
    flip = cv2.bitwise_not(result)
    #print(flip.shape)
    rgb_masked_img = cv2.bitwise_and(copy, copy, mask=flip)
    mod_result = bridge.cv2_to_imgmsg(mod_result, encoding='rgb8') 
    result = bridge.cv2_to_imgmsg(result, encoding='8UC1')
    rgb_masked_img = bridge.cv2_to_imgmsg(rgb_masked_img, encoding='bgr8')
    pub.publish(result)
    pub1.publish(mod_result)
    pub2.publish(rgb_masked_img)
    global publishing_started
    
    if not publishing_started:
        publishing_started = True
        sub2 = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback=drive_depth)


def drive_depth (img : Image):
    
    bridge = CvBridge()
    global flip
    flip1 = flip
    # image = np.frombuffer(img.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
    # print(image.shape)
    copy = image
    flip1 = cv2.resize(flip1, (copy.shape[1],copy.shape[0]))
    depth_masked_img = cv2.bitwise_and(copy, copy, mask=flip1)
    # cv2.imshow('depth_mask', depth_masked_img)
    # cv2.waitKey(1)
    depth_masked_img = bridge.cv2_to_imgmsg(depth_masked_img, encoding='32FC1')
    pub3.publish(depth_masked_img)



if __name__ == '__main__' :
    
    rospy.init_node("DRIVE")
    pub=rospy.Publisher("/semseg/drivable_region", Image, queue_size=10)
    pub1=rospy.Publisher("/semseg/model_result", Image, queue_size=10)
    pub2=rospy.Publisher("/semseg/rgb_masked_image", Image, queue_size=10)
    pub3=rospy.Publisher("/semseg/depth_masked_image", Image, queue_size=10)
    sub = rospy.Subscriber("/lowfps", Image, callback = drive_rgb)
    #sub2 = rospy.Subscriber("/zed2i/zed_node/depth/depth_registered", Image, callback = drive_depth)
    publishing_started = False
    
    rospy.spin()
