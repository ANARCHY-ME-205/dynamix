#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge 

def seg (image : Image):
    bridge = CvBridge()
    dep_img = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    
    