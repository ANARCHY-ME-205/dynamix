#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge 
import numpy as np
import torch
from torchvision import transforms as T
import cv2
import albumentations as A
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('/home/tamoghna/catkin_ws/src/dynamix/scripts/models/seggs_v0.pt', map_location=device)

def predict (img) :
    
    image_np = np.array(img)  # Convert image to a numpy array

    transform = A.Resize(608, 608, interpolation=cv2.INTER_NEAREST)  # Resize image if needed
    augmented = transform(image=image_np)  # Pass the numpy array as a named argument
    augmented_image = augmented['image']  # Retrieve the augmented image

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image_tensor = t(augmented_image)

    model.eval()
    with torch.no_grad():
        # Convert the augmented image to a PyTorch tensor
        image_tensor = t(augmented_image).unsqueeze(0)  # Add batch dimension

        # Move the tensor to the device (CPU or GPU)
        image_tensor = image_tensor.to(device)

        # Make predictions
        output = model(image_tensor)

        # Process the output as needed
        pred_mask = torch.argmax(output, dim=1).squeeze(0)


    pred_mask_np = pred_mask.cpu().numpy()

    return pred_mask_np

def binary (img) :

    lower = 0
    upper = 5

    result = cv2.inRange(img, lower, upper)

    return result


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
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    copy = image_rgb
    
    mod_result = predict(image_rgb)
    result = binary(mod_result)

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
    

# height: 360
# width: 640
# distortion_model: "plumb_bob"
# D: [0.0, 0.0, 0.0, 0.0, 0.0]
# K: [265.56085205078125, 0.0, 320.9938049316406, 0.0, 265.56085205078125, 183.0654296875, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [265.56085205078125, 0.0, 320.9938049316406, 0.0, 0.0, 265.56085205078125, 183.0654296875, 0.0, 0.0, 0.0, 1.0, 0.0]

def create_point_cloud(depth_image, intrinsic_matrix, frame_id):
    # Convert depth image to numpy array
    depth_array = np.array(depth_image, dtype=np.float32)

    # Get image dimensions
    height, width = depth_array.shape

    # Create array of indices for each pixel
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    pixel_indices = np.vstack((cols.flatten(), rows.flatten())).T

    # Convert depth values to 3D point coordinates
    depths = depth_array.flatten()
    pixel_indices = np.pad(pixel_indices, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    points = np.matmul(intrinsic_matrix, pixel_indices.T) * depths
    points = points.T[:, :3]

    # Create PointCloud2 fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    # Create PointCloud2 message
    header = rospy.Header()
    header.frame_id = frame_id
    pc2_msg = pc2.create_cloud_xyz32(header, points)

    return pc2_msg

def depth_masked_callback(depth_msg):
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')


    # Apply necessary processing to depth_image and rgb_image as needed
    global intrinsic_matrix
    # Create point cloud using depth_masked_image
    point_cloud_msg = create_point_cloud(depth_image, intrinsic_matrix, frame_id="camera_frame")

    # Publish the point cloud
    pub4.publish(point_cloud_msg)


if __name__ == '__main__' :
    
    rospy.init_node("DRIVE")
    pub=rospy.Publisher("/zed2i/drivable_region", Image, queue_size=10)
    pub1=rospy.Publisher("/zed2i/model_result", Image, queue_size=10)
    pub2=rospy.Publisher("/zed2i/rgb_masked_image", Image, queue_size=10)
    pub3=rospy.Publisher("/zed2i/depth_masked_image", Image, queue_size=10)
    pub4=rospy.Publisher("zed2i/mypoints", PointCloud2, queue_size=10)

    sub = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, callback = drive_rgb)
    depth_mask_sub = rospy.Subscriber("/zed2i/depth_masked_image", Image, depth_masked_callback)

    publishing_started = False
    
    # Retrieve camera intrinsic matrix from camera information
    intrinsic_matrix = np.array([265.56085205078125, 0.0, 320.9938049316406,
                                 0.0, 265.56085205078125, 183.0654296875,
                                 0.0, 0.0, 1.0]).reshape(3, 3)

    rospy.spin()


