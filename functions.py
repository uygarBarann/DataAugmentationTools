def in_interval(line_label, point, x_crop, y_crop, crop_width, crop_height):
    if line_label == "left" or line_label == "right":
        if point and y_crop <= point[1] <= y_crop + crop_height:  
            return True
        else:   
            return False
    elif line_label == "top" or line_label == "bottom":
        if point and x_crop <= point[0] <= x_crop + crop_width:  
            return True
        else:   
            return False





def find_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate slopes
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')

    # Calculate y-intercepts
    b1 = y1 - m1 * x1 if m1 != float('inf') else x1
    b2 = y3 - m2 * x3 if m2 != float('inf') else x3

    # Check if lines are parallel
    if m1 == m2:
        return None  # No intersection

    # Calculate intersection point
    if m1 == float('inf'):
        x_intersection = x1
        y_intersection = m2 * x_intersection + b2
    elif m2 == float('inf'):
        x_intersection = x3
        y_intersection = m1 * x_intersection + b1
    else:
        x_intersection = (b2 - b1) / (m1 - m2)
        y_intersection = m1 * x_intersection + b1

    return (x_intersection, y_intersection)


def extract_polygons(label_path):
    # Process the files using image_path and label_path
    polygons = []
    with open(label_path, 'r') as label_file:
        for line in label_file:
            line = line.strip()
            parts = line.split()


            # Parse the first number as 'class'
            class_value = int(parts[0])

            # Parse remaining parts into a list of tuples
            polygon = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]

            
            print(f"Class: {class_value}")
            print(f"Tuples: {polygon}")

            polygons.append((class_value, polygon))
    return polygons
    print("\n\n")

def is_between(value, num1, num2):
    lower_bound = min(num1, num2)
    upper_bound = max(num1, num2)
    return lower_bound <= value <= upper_bound