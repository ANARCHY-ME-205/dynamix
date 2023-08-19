import os
import cv2

# Specify the input and output folders
input_folder = '/home/tamoghna/seggs/data/original_images'
output_folder = '/home/tamoghna/seggs/data/original_images'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of files in the input folder
file_list = os.listdir(input_folder)

# Iterate over each file in the input folder
for file_name in file_list:
    # Check if the file is an image (JPG format)
    if file_name.endswith('.png') :
        # Read the image
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)

        # Generate the output file path with the same name but with PNG extension
        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')

        # Convert the image from BGR to RGB (optional, depending on your use case)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save the image in PNG format
        cv2.imwrite(output_path, img_rgb)

print('Image conversion completed.')
