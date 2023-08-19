import cv2
import numpy as np

image = cv2.imread('/home/tamoghna/catkin_ws/src/dynamix/scripts/zed_img.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/home/tamoghna/catkin_ws/src/dynamix/scripts/test14.png')
# blank = np.zeros(image.shape, dtype= 'uint8')

# Mouse callback function

def sobel_filter(image,  min_patch_size):
    

    # Apply Sobel filtering
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude)

    # # Apply thresholding to highlight edges
    # thresholded = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)[1]

    # Perform morphological opening to remove small patches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_patch_size, min_patch_size))
    opened = cv2.morphologyEx(magnitude, cv2.MORPH_OPEN, kernel)

    return opened

def get_intensity(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
        intensity = image[y, x]
        print(f"Intensity at pixel ({x}, {y}): {intensity}")

# Create a window and set the mouse callback
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_intensity)

# Set the minimum patch size to ignore (adjust as needed)
min_patch_size = 1

# Apply Sobel filtering with thresholding and patch removal
filtered_image = sobel_filter(image, min_patch_size)


lower = 6
upper = 8


result = cv2.inRange(image, lower, upper)

cv2.waitKey(1)

while True :
    cv2.imshow('Image', image)
    cv2.imshow('result', result)
    # cv2.imshow('Original Image', image)
    cv2.imshow('Sobel Filtered Image', filtered_image)
    # cv2.imshow('ori', img)
    if cv2.waitKey(1) == ord('q') : 
        break

cv2.destroyAllWindows()