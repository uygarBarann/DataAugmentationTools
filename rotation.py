import cv2
import numpy as np
import os
import math
from functions import find_intersection, extract_polygons, is_between, in_interval


images_directory = "images"
labels_directory = "labels"
output_images_directory = "images_rotated"
output_labels_directory = "labels_rotated"


def save_rotated_data(rotated_image, updated_polygons, output_image_path, output_label_path):
    # Save the cropped image
    cv2.imwrite(output_image_path, rotated_image)

    # Save the updated polygons as label file
    with open(output_label_path, 'w') as label_file:
        for class_value, polygon in updated_polygons:
            polygon_line = f"{class_value} " + " ".join([f"{coord:.6f}" for vertex in polygon for coord in vertex])
            label_file.write(polygon_line + "\n")


def warpAffine(src, M, dsize, from_bounding_box_only=False):
    """
    Applies cv2 warpAffine, marking transparency if bounding box only
    The last of the 4 channels is merely a marker. It does not specify opacity in the usual way.
    """
    return cv2.warpAffine(src, M, dsize)

def rotate_image(image, angle):
    """Rotate the image counterclockwise.
    Rotate the image such that the rotated image is enclosed inside the
    tightest rectangle. The area not occupied by the pixels of the original
    image is colored black.
    Parameters
    ----------
    image : numpy.ndarray
        numpy image
    angle : float
        angle by which the image is to be rotated. Positive angle is
        counterclockwise.
    Returns
    -------
    numpy.ndarray
        Rotated Image
    """
    # get dims, find center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = warpAffine(image, M, (nW, nH), False)

    # image = cv2.resize(image, (w,h))

    return image

def crop_to_center(old_img, new_img):
    """
    Crops `new_img` to `old_img` dimensions
    :param old_img: <numpy.ndarray> or <tuple> dimensions
    :param new_img: <numpy.ndarray>
    :return: <numpy.ndarray> new image cropped to old image dimensions
    """

    if isinstance(old_img, tuple):
        original_shape = old_img
    else:
        original_shape = old_img.shape
    original_width = original_shape[1]
    original_height = original_shape[0]
    original_center_x = original_shape[1] / 2
    original_center_y = original_shape[0] / 2

    new_width = new_img.shape[1]
    new_height = new_img.shape[0]
    new_center_x = new_img.shape[1] / 2
    new_center_y = new_img.shape[0] / 2

    new_left_x = int(max(new_center_x - original_width / 2, 0))
    new_right_x = int(min(new_center_x + original_width / 2, new_width))
    new_top_y = int(max(new_center_y - original_height / 2, 0))
    new_bottom_y = int(min(new_center_y + original_height / 2, new_height))

    # create new img canvas
    canvas = np.zeros(original_shape)

    left_x = int(max(original_center_x - new_width / 2, 0))
    right_x = int(min(original_center_x + new_width / 2, original_width))
    top_y = int(max(original_center_y - new_height / 2, 0))
    bottom_y = int(min(original_center_y + new_height / 2, original_height))

    canvas[top_y:bottom_y, left_x:right_x] = new_img[new_top_y:new_bottom_y, new_left_x:new_right_x]

    return canvas


def rotate_point(origin, point, degree):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    :param degree: <float> Angle in degree.
        Positive angle is counterclockwise.
    """
    
    angle = math.radians(degree)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def extract_and_rotate_polygons(image_width, image_height, image_path, label_path, rotation_degree):

    polygons = extract_polygons(image_path, label_path)  # Assuming you have the extract_polygons function

    updated_polygons = []
    for class_value, polygon in polygons:
        origin = (image_width / 2, -(image_height / 2))
        
        
        updated_polygon = []
        for vertex in polygon:
            x_vertex = int(vertex[0] * image_width) #Anti normalize
            y_vertex = int(vertex[1] * image_height)#Anti normalize
            rotated_x, rotated_y = map(lambda x: round(x * 2) / 2, rotate_point(origin, (x_vertex, -y_vertex), rotation_degree))
            
            updated_polygon.append((rotated_x , -rotated_y) ) 

        updated_polygons.append((class_value, updated_polygon))

    print(updated_polygons)
    return updated_polygons


def handle_edges(polygons, image_width, image_height): #Function that takes the rotated polygons, and check if there are any points outside the frame.
    
    
    # Update polygon coordinates for the cropped region
    updated_polygons = []
    for class_value, polygon in polygons:

        updated_polygon = []
        
        previous_vertex = polygon[-1] #Previous vertex, initialized with last vertex
        previous_x_vertex = previous_vertex[0]
        previous_y_vertex = previous_vertex[1]
        previous_vertex = (previous_x_vertex, previous_y_vertex)
        
        previous_included = previous_x_vertex >= 0 and previous_y_vertex>= 0 and previous_x_vertex <= image_width and previous_y_vertex <= image_height #Tracker of whether previous vertex is included in cropped image or not.
        
        
        width_line_top = ((0, 0),(image_width, 0)) #width_line at top representation
        height_line_left = ((0, 0),(0, image_height)) #height_line at bottom representation
        width_line_bottom = ((0, image_height),(image_width, image_height)) #width_line at bottom representation
        height_line_right = ((image_width, 0),(image_width, image_height)) #height_line at right representation
        
        for vertex in polygon:
            
            x_vertex = vertex[0]
            y_vertex = vertex[1]
            cur_vertex = (x_vertex, y_vertex)
            
            previous_x_vertex = previous_vertex[0]
            previous_y_vertex = previous_vertex[1]
            previous_vertex = (previous_x_vertex, previous_y_vertex)
            
            included = x_vertex >= 0 and y_vertex >= 0 and x_vertex <= image_width and y_vertex <= image_height #Whether the current vertex is included in cropped image or not.
            
            #Intersection points for all lines
            vertex_on_height_left = find_intersection((previous_vertex, cur_vertex), height_line_left)
            vertex_on_height_right = find_intersection((previous_vertex, cur_vertex), height_line_right)
            vertex_on_width_top = find_intersection((previous_vertex, cur_vertex), width_line_top)
            vertex_on_width_bottom = find_intersection((previous_vertex, cur_vertex), width_line_bottom)

            
            if included and previous_included:
                
                normalized_vertex = (x_vertex / image_width, y_vertex / image_height)
                updated_polygon.append(normalized_vertex) 
                
                previous_included = True
            
            elif not included and not previous_included:
                previous_included = False
                
            
            elif included and not previous_included:
                
                
                
                #Append the correct vertex on cropped line to polygon
                if is_between(0, previous_x_vertex, x_vertex) and in_interval("left", vertex_on_height_left, 0, 0, image_width, image_height): #if the vertex on height_line_left
                    

                    
                    update_vertex_on_height = (vertex_on_height_left[0] - 0, vertex_on_height_left[1] - 0)
                    normalized_vertex_on_height = (update_vertex_on_height[0] / image_width, update_vertex_on_height[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_height) 
                
                elif is_between(0 + image_width, previous_x_vertex, x_vertex) and in_interval("right", vertex_on_height_right, 0, 0, image_width, image_height): #if the vertex on height_line_right
                    
                    
                    update_vertex_on_height = (vertex_on_height_right[0] - 0, vertex_on_height_right[1] - 0)
                    normalized_vertex_on_height = (update_vertex_on_height[0] / image_width, update_vertex_on_height[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_height) 
                

                elif is_between(0, previous_vertex[1], cur_vertex[1]) and in_interval("top", vertex_on_width_top, 0, 0, image_width, image_height): #if the vertex on width_line_top
                    
                    update_vertex_on_width = (vertex_on_width_top[0] - 0, vertex_on_width_top[1] - 0)
                    normalized_vertex_on_width = (update_vertex_on_width[0] / image_width, update_vertex_on_width[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_width) 

                elif is_between(0 + image_height, previous_vertex[1], cur_vertex[1]) and in_interval("bottom", vertex_on_width_bottom, 0, 0, image_width, image_height): #if the vertex on width_line_bottom
                    
                    update_vertex_on_width = (vertex_on_width_bottom[0] - 0, vertex_on_width_bottom[1] - 0)
                    normalized_vertex_on_width = (update_vertex_on_width[0] / image_width, update_vertex_on_width[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_width) 
                #Append the current vertex to polygon
                updated_vertex = (x_vertex - 0, y_vertex - 0)
                normalized_vertex = (updated_vertex[0] / image_width, updated_vertex[1] / image_height)
                updated_polygon.append(normalized_vertex) 
                
                previous_included = True
            
            elif not included and previous_included:
                
                if is_between(0, previous_x_vertex, x_vertex) and in_interval("left", vertex_on_height_left, 0, 0, image_width, image_height): #if the vertex on height_line_left
                    

                    
                    update_vertex_on_height = (vertex_on_height_left[0] - 0, vertex_on_height_left[1] - 0)
                    normalized_vertex_on_height = (update_vertex_on_height[0] / image_width, update_vertex_on_height[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_height) 
                
                elif is_between(0 + image_width, previous_x_vertex, x_vertex) and in_interval("right", vertex_on_height_right, 0, 0, image_width, image_height): #if the vertex on height_line_right
                    
                    
                    update_vertex_on_height = (vertex_on_height_right[0] - 0, vertex_on_height_right[1] - 0)
                    normalized_vertex_on_height = (update_vertex_on_height[0] / image_width, update_vertex_on_height[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_height) 
                

                elif is_between(0, previous_vertex[1], cur_vertex[1]) and in_interval("top", vertex_on_width_top, 0, 0, image_width, image_height): #if the vertex on width_line_top
                    
                    update_vertex_on_width = (vertex_on_width_top[0] - 0, vertex_on_width_top[1] - 0)
                    normalized_vertex_on_width = (update_vertex_on_width[0] / image_width, update_vertex_on_width[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_width) 

                elif is_between(0 + image_height, previous_vertex[1], cur_vertex[1]) and in_interval("bottom", vertex_on_width_bottom, 0, 0, image_width, image_height): #if the vertex on width_line_bottom
                    
                    update_vertex_on_width = (vertex_on_width_bottom[0] - 0, vertex_on_width_bottom[1] - 0)
                    normalized_vertex_on_width = (update_vertex_on_width[0] / image_width, update_vertex_on_width[1] / image_height)
                    updated_polygon.append(normalized_vertex_on_width) 
                
                previous_included = False

            previous_vertex = vertex
        if len(updated_polygon) > 2:
            updated_polygons.append((class_value, updated_polygon))
        else:
            continue
    return updated_polygons






if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs(output_images_directory, exist_ok=True)
    os.makedirs(output_labels_directory, exist_ok=True)
    counter = 0
    for filename in os.listdir(images_directory):
        
        
        image_path = os.path.join(images_directory, filename)
        label_path = os.path.join(labels_directory, filename)
        base_name, _ = os.path.splitext(label_path)
        label_path = base_name + ".txt"
        
        if os.path.isfile(label_path):
            polygons = extract_polygons(image_path, label_path)
            
            
            # Load the image
            img = cv2.imread(image_path)
            
            # Perform random cropping and update polygons
            rotated_polygons = extract_and_rotate_polygons(img.shape[1], img.shape[0], image_path, label_path, 37.786)
            updated_polygons = handle_edges(rotated_polygons, img.shape[1], img.shape[0])
            rotated_image = crop_to_center(img, rotate_image(img, 37.786))
            print(updated_polygons)

            # Generate output file paths
            output_image_path = os.path.join(output_images_directory, filename)
            base_name, file_type = os.path.splitext(output_image_path)
            output_image_path = base_name + "_rotated" + file_type
            
            output_label_path = os.path.join(output_labels_directory, filename)
            base_name, _ = os.path.splitext(output_label_path)
            output_label_path = base_name + "_rotated.txt"
            
            # Save cropped data
            save_rotated_data(rotated_image, updated_polygons, output_image_path, output_label_path)
            
        else:
            print(f"Label file missing for image: {image_path}")
        