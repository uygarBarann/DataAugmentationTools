from ultralytics import YOLO
import cv2 as cv
import numpy as np
import random

def identify_color(hsv_value, threshold=30):
    h, s, v = hsv_value
    # Check for white color (high value and low saturation)
    if v >= 255 - threshold and s <= threshold:
        return "White"

    # Check for black color (low value)
    if v <= threshold:
        return "Black"

    # If not white or black, consider it as another color
    return "Other"



img = cv.imread(r'')

model = YOLO("yolov8s-seg.pt")

results = model.predict(source=img, classes=2)

polygon = results[0].masks.xy
print(polygon[0])

img_shape = (img.shape[0], img.shape[1])  # Use the dimensions of your input image
car_mask = np.zeros(img_shape, dtype=np.uint8)

# Draw the polygon on the mask
cv.fillPoly(car_mask, [np.array(polygon[0], dtype=np.int32)], color=255)
inv_mask = cv.bitwise_not(car_mask)

# Apply the mask to the image
roi = cv.bitwise_and(img, img, mask=car_mask)
# Convert ROI to HSV color space
roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# Create a mask to filter out black points
non_black_mask = (roi_hsv[:, :, 2] > 10)  # Adjust the threshold as needed
# Apply the non-black mask to the HSV ROI
non_black_hsv = roi_hsv[non_black_mask]
# Count occurrences of each non-black HSV color value
unique_hsv, counts = np.unique(non_black_hsv, axis=0, return_counts=True)
# Find the index of the most encountered color
most_common_index = np.argmax(counts)
# Get the most encountered HSV color value
most_common_hsv = unique_hsv[most_common_index]
print("Most encountered HSV color value:", most_common_hsv)


#Determine color label
color_label = identify_color(most_common_hsv, threshold=30)

if color_label == "Other":
    random_hue = random.randint(0, 360)
    # Convert the image from RGB space to HSV space
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Extract the hue channel from the HSV image
    hue = img_hsv[:, :, 0]

    # Calculate the modified hue values
    scale_factor = 0.7  # Adjust this value for the desired color change intensity
    modified_hue = (hue + scale_factor * random_hue) % 180

    # Assign the modified hue channel to the HSV image
    img_hsv[:, :, 0] = modified_hue

    bgr_img = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)

    result = cv.bitwise_and(bgr_img, bgr_img, mask=car_mask)
    background = cv.bitwise_and(img, img, mask=inv_mask)

    added_img = cv.add(result, background)

elif color_label == "White":
    random_hue = random.randint(0, 360)
    # Convert the image from RGB space to HSV space
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Extract the hue channel from the HSV image
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    

    # Calculate the modified hue values
    scale_factor_R = 0.6  # Adjust this value for the desired color change intensity
    scale_factor_G = 0.4
    scale_factor_B = 0.1
    
    modified_R = R * scale_factor_R
    # Set the saturation channel to the constant value
    modified_G = G * scale_factor_G
    
    modified_B = B * scale_factor_B
    
    # Assign the modified hue channel to the HSV image
    img_rgb[:, :, 0] = modified_R
    img_rgb[:, :, 1] = modified_G
    img_rgb[:, :, 2] = modified_B

    

    bgr_img = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

    result = cv.bitwise_and(bgr_img, bgr_img, mask=car_mask)
    background = cv.bitwise_and(img, img, mask=inv_mask)

    added_img = cv.add(result, background)



cv.imshow("original", img)
cv.imshow("final", added_img)

cv.waitKey(0)
cv.destroyAllWindows() #NOT FINISHED
