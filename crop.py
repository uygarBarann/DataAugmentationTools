import cv2
import random
import numpy as np
import os
from functions import find_intersection, extract_polygons, is_between, in_interval

images_directory = "images"
labels_directory = "labels"
output_images_directory = "images_cropped"
output_labels_directory = "labels_cropped"

#Function that determines if the point is on the frame or not.

    



def random_crop(img, polygons, crop_width, crop_height):
    assert img.shape[0] >= crop_height
    assert img.shape[1] >= crop_width
    found = False #Whether there are at least one label inside the cropped
    while not found:
        
        x_crop = random.randint(0, img.shape[1] - crop_width)
        y_crop = random.randint(0, img.shape[0] - crop_height)
        print(x_crop,y_crop)
        
        cropped_image  = img[y_crop:y_crop+crop_height, x_crop:x_crop+crop_width]
        # Update polygon coordinates for the cropped region
        updated_polygons = []
        for class_value, polygon in polygons:

            updated_polygon = []
            
            previous_vertex = polygon[-1] #Previous vertex, initialized with last vertex
            previous_x_vertex = int(previous_vertex[0] * img.shape[1])
            previous_y_vertex = int(previous_vertex[1] * img.shape[0])
            denormalized_previous = (previous_x_vertex, previous_y_vertex)
            
            previous_included = previous_x_vertex >= x_crop and previous_y_vertex>= y_crop and previous_x_vertex <= x_crop + crop_width and previous_y_vertex <= y_crop + crop_height #Tracker of whether previous vertex is included in cropped image or not.
            
            
            width_line_top = ((x_crop, y_crop),(x_crop + crop_width, y_crop)) #width_line at top representation
            height_line_left = ((x_crop, y_crop),(x_crop, y_crop + crop_height)) #height_line at bottom representation
            width_line_bottom = ((x_crop, y_crop + crop_height),(x_crop + crop_width, y_crop + crop_height)) #width_line at bottom representation
            height_line_right = ((x_crop + crop_width, y_crop),(x_crop + crop_width, y_crop + crop_height)) #height_line at right representation

            corner_stack = [] #A stack that in case of a corner should added to the polygon
            #Flags
            left_top_included = False
            right_top_included = False
            left_bottom_included = False
            right_bottom_included = False
            from_inside = False
            
            for vertex in polygon:
                
                x_vertex = int(vertex[0] * img.shape[1]) #Anti normalize
                y_vertex = int(vertex[1] * img.shape[0])#Anti normalize
                denormalized_vertex = (x_vertex, y_vertex)
                
                previous_x_vertex = int(previous_vertex[0] * img.shape[1])
                previous_y_vertex = int(previous_vertex[1] * img.shape[0])
                denormalized_previous = (previous_x_vertex, previous_y_vertex)
                
                included = x_vertex >= x_crop and y_vertex >= y_crop and x_vertex <= x_crop + crop_width and y_vertex <= y_crop + crop_height #Whether the current vertex is included in cropped image or not.
                
                #Intersection points for all lines
                vertex_on_height_left = find_intersection((denormalized_previous, denormalized_vertex), height_line_left)
                vertex_on_height_right = find_intersection((denormalized_previous, denormalized_vertex), height_line_right)
                vertex_on_width_top = find_intersection((denormalized_previous, denormalized_vertex), width_line_top)
                vertex_on_width_bottom = find_intersection((denormalized_previous, denormalized_vertex), width_line_bottom)
                
                if included and previous_included:
                    updated_vertex = (x_vertex - x_crop, y_vertex - y_crop)
                    normalized_vertex = (updated_vertex[0] / crop_width, updated_vertex[1] / crop_height)
                    updated_polygon.append(normalized_vertex) 
                    
                    previous_included = True
                
                elif not included and not previous_included:
                    
                    #Handling the corner edge cases
                    if from_inside:    
                        if y_vertex < y_crop and is_between(x_crop, x_vertex, previous_x_vertex):
                            left_top_included = not left_top_included
                            if left_top_included:
                                corner_stack.append((x_crop, y_crop))
                            else:
                                corner_stack.pop()
                        
                        elif y_vertex < y_crop and is_between(x_crop + crop_width, x_vertex, previous_x_vertex):
                            right_top_included = not right_top_included
                            if right_top_included:
                                corner_stack.append((x_crop + crop_width, y_crop))
                            else:
                                corner_stack.pop()
                        
                        elif y_vertex > y_crop + crop_height and is_between(x_crop, x_vertex, previous_x_vertex):
                            left_bottom_included = not left_bottom_included
                            if left_bottom_included:
                                corner_stack.append((x_crop, y_crop + crop_height))
                            else:
                                corner_stack.pop()
                        
                        elif y_vertex > y_crop + crop_height and is_between(x_crop + crop_width, x_vertex, previous_x_vertex):
                            right_bottom_included = not right_bottom_included
                            if right_bottom_included:
                                corner_stack.append((x_crop + crop_width, y_crop + crop_height))
                            else:
                                corner_stack.pop()
                    
                    
                    
                    previous_included = False
                    
                
                elif included and not previous_included:
                    
                    #Append the previous corners
                    for corner in corner_stack:
                        updated_corner = (corner[0] - x_crop, corner[1] - y_crop)
                        normalized_corner = (updated_corner[0] / crop_width, updated_corner[1] / crop_height)
                        updated_polygon.append(normalized_corner)
                    #Delete the stack content
                    corner_stack = []
                    #Convert default settings the flags
                    left_top_included = False
                    right_top_included = False
                    left_bottom_included = False
                    right_bottom_included = False
                    from_inside = False
                    
                    #Append the correct vertex on cropped line to polygon
                    if is_between(x_crop, denormalized_previous[0], denormalized_vertex[0]) and in_interval("left", vertex_on_height_left, x_crop, y_crop, crop_width, crop_height): #if the vertex on height_line_left
                        
                        
                        update_vertex_on_height = (vertex_on_height_left[0] - x_crop, vertex_on_height_left[1] - y_crop)
                        normalized_vertex_on_height = (update_vertex_on_height[0] / crop_width, update_vertex_on_height[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_height) 
                    
                    elif is_between(x_crop + crop_width, denormalized_previous[0], denormalized_vertex[0]) and in_interval("right", vertex_on_height_right, x_crop, y_crop, crop_width, crop_height): #if the vertex on height_line_right
                        
                        
                        update_vertex_on_height = (vertex_on_height_right[0] - x_crop, vertex_on_height_right[1] - y_crop)
                        normalized_vertex_on_height = (update_vertex_on_height[0] / crop_width, update_vertex_on_height[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_height) 
                    
    
                    elif is_between(y_crop, denormalized_previous[1], denormalized_vertex[1]) and in_interval("top", vertex_on_width_top, x_crop, y_crop, crop_width, crop_height): #if the vertex on width_line_top
                        
                        update_vertex_on_width = (vertex_on_width_top[0] - x_crop, vertex_on_width_top[1] - y_crop)
                        normalized_vertex_on_width = (update_vertex_on_width[0] / crop_width, update_vertex_on_width[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_width) 

                    elif is_between(y_crop + crop_height, denormalized_previous[1], denormalized_vertex[1]) and in_interval("bottom", vertex_on_width_bottom, x_crop, y_crop, crop_width, crop_height): #if the vertex on width_line_bottom
                        
                        update_vertex_on_width = (vertex_on_width_bottom[0] - x_crop, vertex_on_width_bottom[1] - y_crop)
                        normalized_vertex_on_width = (update_vertex_on_width[0] / crop_width, update_vertex_on_width[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_width) 
                    #Append the current vertex to polygon
                    
                    updated_vertex = (x_vertex - x_crop, y_vertex - y_crop)
                    normalized_vertex = (updated_vertex[0] / crop_width, updated_vertex[1] / crop_height)
                    updated_polygon.append(normalized_vertex) 
                    
                    previous_included = True
                
                elif not included and previous_included:
                    
                    if is_between(x_crop, denormalized_previous[0], denormalized_vertex[0]) and in_interval("left", vertex_on_height_left, x_crop, y_crop, crop_width, crop_height): #if the vertex on height_line_left
                        
                        
                        update_vertex_on_height = (vertex_on_height_left[0] - x_crop, vertex_on_height_left[1] - y_crop)
                        normalized_vertex_on_height = (update_vertex_on_height[0] / crop_width, update_vertex_on_height[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_height) 
                    
                    elif is_between(x_crop + crop_width, denormalized_previous[0], denormalized_vertex[0]) and in_interval("right", vertex_on_height_right, x_crop, y_crop, crop_width, crop_height): #if the vertex on height_line_right
                        
                        
                        update_vertex_on_height = (vertex_on_height_right[0] - x_crop, vertex_on_height_right[1] - y_crop)
                        normalized_vertex_on_height = (update_vertex_on_height[0] / crop_width, update_vertex_on_height[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_height) 
                    
    
                    elif is_between(y_crop, denormalized_previous[1], denormalized_vertex[1]) and in_interval("top", vertex_on_width_top, x_crop, y_crop, crop_width, crop_height): #if the vertex on width_line_top
                        
                        update_vertex_on_width = (vertex_on_width_top[0] - x_crop, vertex_on_width_top[1] - y_crop)
                        normalized_vertex_on_width = (update_vertex_on_width[0] / crop_width, update_vertex_on_width[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_width) 

                    elif is_between(y_crop + crop_height, denormalized_previous[1], denormalized_vertex[1]) and in_interval("bottom", vertex_on_width_bottom, x_crop, y_crop, crop_width, crop_height): #if the vertex on width_line_bottom
                        
                        update_vertex_on_width = (vertex_on_width_bottom[0] - x_crop, vertex_on_width_bottom[1] - y_crop)
                        normalized_vertex_on_width = (update_vertex_on_width[0] / crop_width, update_vertex_on_width[1] / crop_height)
                        updated_polygon.append(normalized_vertex_on_width) 
                    
                    previous_included = False
                    from_inside = True
                previous_vertex = vertex
            if len(updated_polygon) > 2:
                found = True
                updated_polygons.append((class_value, updated_polygon))
            else:
                continue
    return cropped_image, updated_polygons


def save_cropped_data(cropped_image, updated_polygons, output_image_path, output_label_path):
    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_image)

    # Save the updated polygons as label file
    with open(output_label_path, 'w') as label_file:
        for class_value, polygon in updated_polygons:
            polygon_line = f"{class_value} " + " ".join([f"{coord:.6f}" for vertex in polygon for coord in vertex])
            label_file.write(polygon_line + "\n")





if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs(output_images_directory, exist_ok=True)
    os.makedirs(output_labels_directory, exist_ok=True)
    
    for filename in os.listdir(images_directory):
        image_path = os.path.join(images_directory, filename)
        label_path = os.path.join(labels_directory, filename)
        base_name, _ = os.path.splitext(label_path)
        label_path = base_name + ".txt"
        
        if os.path.isfile(label_path):
            polygons = extract_polygons(image_path, label_path)
            
            
            # Load the image
            img = cv2.imread(image_path)
            
            # Define crop dimensions
            crop_width = 350  # Modify as needed
            crop_height = 350  # Modify as needed
            
            # Perform random cropping and update polygons
            cropped_img, updated_polygons = random_crop(img, polygons, crop_width, crop_height)
            print(updated_polygons)

            # Generate output file paths
            output_image_path = os.path.join(output_images_directory, filename)
            base_name, file_type = os.path.splitext(output_image_path)
            output_image_path = base_name + "_cropped" + file_type
            
            output_label_path = os.path.join(output_labels_directory, filename)
            base_name, _ = os.path.splitext(output_label_path)
            output_label_path = base_name + "_cropped.txt"
            
            # Save cropped data
            save_cropped_data(cropped_img, updated_polygons, output_image_path, output_label_path)
            
        else:
            print(f"Label file missing for image: {image_path}")
        