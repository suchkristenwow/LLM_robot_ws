import numpy as np 
import cv2 
import re 
from llm_robot_client.srv import getPixelFrom3D,get3DPointFromPixel, ProjectBackOfBbox
import rospy 
import tf.transformations as tft 
from shapely.geometry import Polygon
from shapely.geometry import Point as shapelyPoint  
import sensor_msgs.point_cloud2 as pc2 
from geometry_msgs.msg import Point 
from llm_utils import generate_with_openai 
from sensor_msgs.msg import CompressedImage 
from std_msgs.msg import Header 
import copy 
import pickle 
from rospy import ServiceException
import rosservice 
import os 
import concurrent.futures 
import matplotlib.pyplot as plt 

def draw_filled_box(image, bbox):
    """
    Draws a filled black box on the image for the given bounding box coordinates.
    
    Args:
        image (np.ndarray): The image data.
        bbox (tuple): The bounding box coordinates as (x1, y1, x2, y2).
    
    Returns:
        np.ndarray: The image with the filled box drawn.
    """
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox
    
    # Draw a filled black box on the image (color is black and thickness is -1 for filling)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    
    return image

def get_object_colors(description):
    prompt = """The user has provided this object description: 
        """ + description + """
        If the user has specified the color or possible colors of the object in the description, return them in a list like this: [green,blue]
        If the user has not mentioned any colors in the description, return []"""  

    print("prompt: ", prompt)
    response, _ = generate_with_openai(prompt)  # Assuming this returns a string
    
    if not isinstance(response, str):
        response = str(response)
        
    #print("response:", response)

    # Find the positions of the square brackets
    matches = list(re.finditer(r'[\[\]]', response))
    
    # Ensure we have both left and right brackets
    if len(matches) < 2:
        #print("No valid brackets found in the response")
        return []

    left_bracket = matches[0].start()
    right_bracket = matches[1].start()  # Change to matches[1] for the closing bracket
    
    # Extract the substring containing the colors
    sub_string = response[left_bracket + 1:right_bracket]  # +1 to exclude the brackets themselves
    #print("sub_string: ", sub_string)
    
    # Split the substring to extract colors
    if ',' in sub_string:
        colors = sub_string.split(',')
        # Strip extra spaces and quotes
        colors = [color.strip().replace("'", "").replace('"', "") for color in colors]
    else:
        colors = []
    
    #print("colors: ", colors)
    return colors

def contour_points_from_mask(binary_mask):
    """
    Find the largest contour in a binary mask and return an array of geometry_msgs/Point.

    Parameters:
        binary_mask (numpy array): 2D binary image mask (0 for background, 255 for object).

    Returns:
        largest_contour_points (list of geometry_msgs/Point): List of points representing the largest contour.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        rospy.logwarn("No contours found in the mask.")
        return []

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Convert the largest contour to a list of geometry_msgs/Point
    largest_contour_points = []
    for point in largest_contour:
        p = Point()
        p.x = float(point[0][0])  # X coordinate
        p.y = float(point[0][1])  # Y coordinate
        p.z = 0.0                 # Z coordinate (assuming 2D mask, so Z is 0)
        largest_contour_points.append(p)

    return largest_contour_points

def get_centroid_from_mask(mask):
    # Convert boolean mask to integer mask if needed
    mask = mask.astype(np.uint8)

    # Get the coordinates of all non-zero pixels
    y_indices, x_indices = np.nonzero(mask)

    # Check if the mask contains any non-zero values
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Mask contains no object")

    # Calculate the centroid by averaging the coordinates
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)

    # Return the centroid as (x, y) pixel coordinates
    return (centroid_x, centroid_y)

def extract_measurements(text):
    # Initialize the dictionary with None values
    measurements = {
        'area': None,
        'width': None,
        'height': None,
        'linear_length': None
    }

    # Define regex patterns to search for measurements in meters
    patterns = {
        'area': r'area\s*:\s*(\d+\.?\d*)',
        'width': r'width\s*:\s*(\d+\.?\d*)',
        'height': r'height\s*:\s*(\d+\.?\d*)',
        'linear_length': r'linear_length\s*:\s*(\d+\.?\d*)'
    }

    # Search the text for each key
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            measurements[key] = float(match.group(1))  # Convert matched value to float
    #print("measurements: ",measurements)
    return measurements

def get_obj_size(obj_type,obj_description): 
    """
    Initialize dictionary with size with keys: area, width, height, linear_length 
    """
    prompt = "The user has defined " + obj_type + " like this: " + "\n" + obj_description + "\n" + \
        """Can you return a dictionary with the keys area, width, height, linear_length? If you're not sure, 
            you can put None. For example: If the object was a car, and this was the description; a car is a 
            very large machine made of metal and glass that can seat between 2-6 people. It has 4 wheels, and 
            people use it to get places quickly. From this description, a good response would be something like 
            this: {'area':8.5,'width':2,'height':1.8,'linear_length':4.5}
            The measurements should be in meters."""

    #print("prompt: ",prompt)
    response,_ = generate_with_openai(prompt) 
    #print("response: ",response) 
    size_dict = extract_measurements(response) 
    return size_dict 
    
def is_ground_obj(obj_type,obj_description,image_path):
    """
    Given the object type and description, determine if this object is on the ground in this image
    """
    prompt = (
        "The user has defined " + obj_type + " like this: \n" + obj_description + "\n" +
        "Given this description and this image, would a " + obj_type +
        " be found on the ground in the kind of environment in this image?"
        )   
    #print("prompt: ",prompt)
    response,_ = generate_with_openai(prompt,image_path=image_path)
    #print("response: ",response)
    if 'yes' in response.lower():
        return True 
    else:
        return False 

def pointcloud2_msg_to_array(pointcloud_msg):
    """
    Convert a PointCloud2 message to a numpy array.
    """
    # Extract point data from PointCloud2 message
    points_list = []
    for point in pc2.read_points(pointcloud_msg, skip_nans=True):
        points_list.append([point[0], point[1], point[2]])  # x, y, z coordinates

    return np.array(points_list)

def get_ground_plane_pts(ground_plane_z,robot_tfs,robot_pose,ground_plane_pts,frame):
    """
    Get ground plane points that are in the field of view of the camera
    """ 
    x = robot_pose.pose.pose.position.x 
    y = robot_pose.pose.pose.position.y 
    z = robot_pose.pose.pose.position.z 
    rot_x = robot_pose.pose.pose.orientation.x 
    rot_y = robot_pose.pose.pose.orientation.y 
    rot_z = robot_pose.pose.pose.orientation.z 
    rot_w = robot_pose.pose.pose.orientation.w 
    euler = tft.euler_from_quaternion([rot_x, rot_y, rot_z, rot_w]) 

    if 'front' in frame: 
        frustrum_corners = Polygon(robot_tfs.get_front_cam_fov(frame,np.array([x,y,z,euler[0],euler[1],euler[2]])))  
    elif 'right' in frame: 
        frustrum_corners = Polygon(robot_tfs.get_right_cam_fov(frame,np.array([x,y,z,euler[0],euler[1],euler[2] + np.pi/2])))  
    elif 'left' in frame: 
        frustrum_corners = Polygon(robot_tfs.get_left_cam_fov(frame,np.array([x,y,z,euler[0],euler[1],euler[2] - np.pi/2])))   

    pc_points_arr = pointcloud2_msg_to_array(ground_plane_pts)
    
    pc_points = [shapelyPoint(x, y) for x, y, z in pc_points_arr]
    inside = np.array([frustrum_corners.contains(p) for p in pc_points])
    points_inside_frustrum = pc_points_arr[inside] 
    

    # If no points are inside, return an empty array
    if len(points_inside_frustrum) == 0:
        rospy.logerr("There are no points inside the frustrum!")
        return np.array([])

    # Get the median of the x-components of the points inside the frustum
    median_x = np.median(points_inside_frustrum[:, 0])

    # Filter the points to return only those with an x-component smaller than or equal to the median
    closer_points = points_inside_frustrum[points_inside_frustrum[:, 0] <= median_x]

    rospy.loginfo("There are {} closer_points".format(len(closer_points))) 

    ground_points = [x for x in closer_points if ground_plane_z - 0.1 <= x[2] <= ground_plane_z + 0.1]

    rospy.loginfo("There are {} ground points".format(len(ground_points)))

    return ground_points 

def inverse_projection(img_shape,pts,frame,service_name='/H03/project_detections/get_pixel_from_3d'):
    """
    Given a 3D camera coordinate, get the corresponding px coordinate 
    """
    print("Doing inverse projection ...")
    px_coords = []

    #Wait for the service to become available
    rospy.wait_for_service(service_name)

    for pt in pts: 
        try:
            # Create a proxy for the service
            get_pixel_from_3d = rospy.ServiceProxy(service_name, getPixelFrom3D)

            if 'front' in frame:
                frame_name = 'cam_front_link'
            elif 'right' in frame: 
                frame_name = 'cam_right_link'
            elif 'left' in frame: 
                frame_name = 'cam_left_link'
            elif 'back' in frame:   
                frame_name = 'cam_back_link' 

            #print("this is the 3D point im trying to project: ",pt)
            response = get_pixel_from_3d(pt[0],pt[1],pt[2],frame_name)
            if 0 <= response.u <= img_shape[1]:
                if 0 <= response.v <= img_shape[0]:
                    # Return the pixel coordinates from the response
                    px_coords.append((response.u, response.v)) 

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s",str(e))
            continue 

    return px_coords

def get_bounding_box_from_mask(mask):
    # Convert boolean mask to integer mask
    mask = mask.astype(np.uint8)

    # Find the indices where mask is non-zero (1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Get the bounding box by finding the min and max indices where the mask is non-zero
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Return the bounding box
    return (xmin, ymin, xmax + 1, ymax + 1)  # xmax and ymax are +1 because slicing is exclusive

def find_lowest_center(mask):
    # Find the indices of the mask where it is True
    true_indices = np.argwhere(mask)

    if len(true_indices) == 0:
        return None  # No mask found

    # Get the row with the largest Y value (lowest part of the image)
    max_y = np.max(true_indices[:, 0])

    # Get all X values for the max Y row
    x_coords = true_indices[true_indices[:, 0] == max_y][:, 1]

    # Find the center X coordinate
    center_x = np.median(x_coords).astype(int)


def pad_mask(img, mask, target_shape):
    """
    Pads a binary mask (2D binary array) with False values to fit the target shape.

    Args:
        mask (numpy.ndarray): The input binary mask (2D array).
        target_shape (tuple): The desired shape (height, width) to pad the mask to.

    Returns:
        numpy.ndarray: The padded binary mask with False values.
    """
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    # Get the current shape of the mask
    if isinstance(mask,dict): 
        mask_shape = mask['segmentation'].shape
    else:
        mask_shape = mask.shape

    # Calculate padding sizes
    pad_height = max(0, target_shape[0] - mask_shape[0])
    pad_width = max(0, target_shape[1] - mask_shape[1])

    # Calculate padding for each side (we'll pad equally on all sides)
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    # Apply padding using np.pad with constant values of False
    padded_mask = np.pad(mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=False)

    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(padded_mask) 
    plt.axis('off')
    #plt.show()
    print("writing: {}".format("/home/marble/padded_mask_fig.jpg"))
    plt.savefig("/home/marble/padded_mask_fig.jpg") 
    plt.close() 
    return padded_mask

def filter_ground_masks(custom_obj_masks,img,on_ground,ground_level,frame,service_name='/H03/project_detections/get_3d_point_from_pixel'): 
    print("filtering ground masks ...")
    from .object_detection_yolo_client import ObjectDetectionClient  

    #want to project the mask centroids and see if theyre on the ground or not
    if 'front' in frame: 
        frame_name = 'cam_front_link'
    elif 'left' in frame:
        frame_name = 'cam_left_link' 
    elif 'right' in frame: 
        frame_name = 'cam_right_link' 

    filtered_masks = []
    if not os.path.exists("/home/marble/filtered_annotated_imgs"):
        os.mkdir("/home/marble/filtered_annotated_imgs")

    for i,mask in enumerate(custom_obj_masks): 
        x,y = find_lowest_center(mask) 
        annotated_msg = ObjectDetectionClient.annotate_image(mask,img,'bgr8') 
        masked_img = annotated_msg.data 
        np_arr = np.frombuffer(masked_img,np.uint8) 
        image_np = cv2.imdecode(np_arr,cv2.IMREAD_COLOR) 
        # Draw the centroid point on the masked image
        cv2.circle(image_np, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at (x, y)
        
        # Save the annotated image using OpenCV
        output_path = "/home/marble/filtered_annotated_imgs/annotated_img_{}.png".format(i)
        print("writing {}".format(output_path))
        cv2.imwrite(output_path, image_np)   # Save the annotated image 

        get_3d_point = rospy.ServiceProxy(service_name, get3DPointFromPixel)        
        response = get_3d_point(int(x),int(y), frame_name)
        
        #while np.all([response.world_point.x,response.world_point.y,response.world_point.z] == 0): 
        tries = 0
        if response.world_point.x == 0.0 and response.world_point.y == 0.0 and response.world_point.z == 0.0:
            if not on_ground:
                print("I cant find intersection with ground ... going to append mask just in case!")
                filtered_masks.append(mask) 
        else: 
            if not (response.world_point.x == 0.0 and response.world_point.y == 0.0 and response.world_point.z == 0.0):
                print("response: ",response)
                if response.world_point.y  < ground_level + 0.1:
                    if on_ground:
                        print("appending mask to filtered mask")
                        filtered_masks.append(mask)
                else:
                    #the thing is not on the ground
                    if not on_ground: 
                        filtered_masks.append(mask)

    print("returning {} filtered masks".format(len(filtered_masks)))
    
    return filtered_masks 

def filter_masks(masks,image_data,obj_name,obj_description,image_frame,obj_colors): 
    print("entering filter masks!!!")

    '''
    if len(masks) > max_masks_for_consideration: 
        masks = filter_masks_by_size(masks,obj_size,image_frame)
    ''' 

    if len(obj_colors) > 0:
        masks = filter_masks_by_color(masks,obj_name,obj_description,image_data,obj_colors)

    return masks 

def filter_single_mask(mask, obj_color, obj_name, obj_description, image_data):
    """
    Helper function to process a single mask and return whether it has the correct color.
    """
    h, w, _ = image_data.shape
    if isinstance(mask, dict):
        mask_bb = get_bounding_box_from_mask(mask['segmentation'])
    else:
        mask_bb = get_bounding_box_from_mask(mask)

    x_min, y_min, x_max, y_max = mask_bb
    img = copy.deepcopy(image_data)
    img_with_box = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
    img_path = "/home/marble/img_with_box_{}.jpg".format(x_min)  # unique filename for each mask
    cv2.imwrite(img_path, img_with_box)

    if len(obj_color) == 1:
        prompt = "Is the object in this bounding box {}? Please include 'yes' or 'no' in your response for parsing.".format(obj_color[0])
    else:
        prompt = "Is the object in this bounding box any of the following colors {}? Please include 'yes' or 'no' in your response for parsing.".format(str(obj_color))

    response, _ = generate_with_openai(prompt, image_path=img_path)
    return 'yes' in response.lower()

def filter_masks_by_color(masks, obj_name, obj_description, image_data, obj_color):
    """
    Determine which masks have the right color, using parallelization.
    """
    print("filtering masks by color ...")
    correct_color_masks = []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each mask processing to the executor
        future_to_mask = {
            executor.submit(filter_single_mask, mask, obj_color, obj_name, obj_description, image_data): mask
            for mask in masks
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_mask):
            mask = future_to_mask[future]
            try:
                if future.result():
                    correct_color_masks.append(mask)
            except Exception as exc:
                print("An exception occurred during mask processing: {}".format(exc))

    return correct_color_masks

def filter_masks_by_size(masks, obj_size, image_frame, service_name='/H03/project_detections/get_3d_point_from_pixel'):
    print("filtering masks by size ...")

    rospy.wait_for_service(service_name)

    appropriately_sized_masks = [] 
    for mask in masks:
        if isinstance(mask, dict):
            mask_bb = get_bounding_box_from_mask(mask['segmentation'])
        else:
            mask_bb = get_bounding_box_from_mask(mask)

        x_min, y_min, x_max, y_max = mask_bb
        center_x = np.mean([x_min, x_max])
        center_y = np.mean([y_min, y_max])
        print("center_x: {}, center_y: {}".format(center_x, center_y))

        if 'front' in image_frame: 
            frame_name = 'cam_front_link'
        elif 'left' in image_frame:
            frame_name = 'cam_left_link' 
        elif 'right' in image_frame: 
            frame_name = 'cam_right_link' 

        projected_points = {}
        try:
            rospy.loginfo("Calling service: {} with center_x: {}, center_y: {}".format(service_name, center_x, center_y))

            # Call the service
            get_3d_point = rospy.ServiceProxy(service_name, get3DPointFromPixel)
            
            response = get_3d_point(int(center_x),int(center_y), frame_name)
            
            # Add logging for successful service call
            rospy.loginfo("Service call success, response: {}".format(response))
            
            center_pt = response.world_point
            projected_points['mask_center'] = center_pt
        except ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
            continue  # Skip this mask if there's an error
        except Exception as e:
            rospy.logerr("Unexpected error: {}".format(e))
            continue
        
        # Continue with the rest of the function as before...
        corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)] 
        world_corner_pts = []
        for corner in corners: 
            u = corner[0]; v = corner[1]
            try:
                get_3d_point = rospy.ServiceProxy(service_name, get3DPointFromPixel)
                response = get_3d_point(int(u),int(v), frame_name)
                #print("response:", response) 
                #while np.all([response.world_point.x,response.world_point.y,response.world_point.z] == 0):
                while response.world_point.x == 0.0 and response.world_point.y == 0.0 and response.world_point.z == 0.0:
                    print("All the world pt are 0 trying again ...")
                    response = get_3d_point(int(u),int(v), frame_name) 
                world_corner_pts.append(response.world_point) 
            except rospy.ServiceException as e:
                rospy.logerr("Couldn't project corner point. Service call failed: %s" % e)
                continue  # Skip if any of the corners fail to project

        projected_points['bounding_box_front_pts'] = world_corner_pts 
        # Try to get the back of the bounding box
        try:
            rospy.wait_for_service('/H03/project_detections/get_back_of_bbox_from_pixel')
            get_back_of_bbox = rospy.ServiceProxy('/H03/project_detections/get_back_of_bbox_from_pixel', ProjectBackOfBbox)
            response = get_back_of_bbox(center_x, center_y, frame_name) 
            projected_points['bounding_box_back_pts'] = response.back_corners 
        except rospy.ServiceException as e:
            rospy.logerr("Couldn't find the back of the bounding box. Service call failed: %s" % e) 

        if compare_size(projected_points, obj_size):
            appropriately_sized_masks.append(mask)
        else:
            rospy.loginfo("Excluded a mask because estimated size was not correct")
 
    return appropriately_sized_masks


def compare_size(projected_points,obj_size):
    """
    obj_size has keys: area,width,height,linear_length 
    projected_points: mask_center, bounding_box_front_pts, bounding_box_back_pts
    return bool
    """
    print("trying to compare size ...") 
    print("obj_size: ",obj_size) 
    if "bounding_box_front_pts" in projected_points.keys() and "bounding_box_back_pts" in projected_points.keys():
        #found the back of the bounding box 
        est_width,est_height,est_linear_length = calculate_bbox_dimensions(projected_points["bounding_box_front_pts"],projected_points["bounding_box_back_pts"])
        est_area = est_width * est_height  
        if obj_size["height"] is not None:
            if obj_size["height"]*0.5 <= est_height <= obj_size["height"]*1.5:
                if obj_size["area"] is not None: 
                    if obj_size["area"]*0.5 <= est_area <= obj_size["area"]*1.5:
                        return True 
                    
        if obj_size["area"]*0.5 <= est_area <= obj_size["area"]*1.5:
            if obj_size["linear_length"] is not None: 
                if obj_size["linear_length"]*0.5 <= obj_size["linear_length"] <= obj_size["linear_length"]*1.5:
                    return True 
            else:
                return True 
            
    elif "bounding_box_front_pts" in projected_points.keys():
        #couldnt find the back points of the bounding box 
        # top_left, top_right, bottom_left, bottom_right
        ftl,ftr,fbr,fbr = classify_corners(projected_points["bounding_box_front_pts"])
        est_width = euclidean_distance(ftl, ftr) 
        if obj_size["width"]*0.5 <= est_width <= obj_size["width"]*1.5:
            return True 
    else:   
        #include the mask to be conservative I guess 
        return True  
    
    return False 
        
def euclidean_distance(p1, p2):
    """Compute the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def classify_corners(corners):
    """
    Classifies the 4 corners of a face (front or back) of a 3D bounding box into
    top-left, top-right, bottom-left, and bottom-right based on their coordinates.

    corners: List of 4 tuples representing the 3D coordinates of the bounding box corners (face).
             The points should be passed without any specific order.

    Returns:
        (top_left, top_right, bottom_left, bottom_right): Classified corners
    """
    # Sort points by their y-coordinate to separate top and bottom
    tmp = []
    for corner in corners: 
        #these are point objects, want to convert to tuple 
        tmp.append((corner.x,corner.y))

    sorted_by_y = sorted(tmp, key=lambda p: p[1], reverse=True)  # Sort by y descending (top first)

    # From the top points, find the leftmost and rightmost points (based on x-coordinate)
    if sorted_by_y[0][0] < sorted_by_y[1][0]:
        top_left = sorted_by_y[0]
        top_right = sorted_by_y[1]
    else:
        top_left = sorted_by_y[1]
        top_right = sorted_by_y[0]

    # From the bottom points, find the leftmost and rightmost points (based on x-coordinate)
    if sorted_by_y[2][0] < sorted_by_y[3][0]:
        bottom_left = sorted_by_y[2]
        bottom_right = sorted_by_y[3]
    else:
        bottom_left = sorted_by_y[3]
        bottom_right = sorted_by_y[2]

    return top_left, top_right, bottom_left, bottom_right

def calculate_bbox_dimensions(front_corners,back_corners):
    """
    Calculate the width, height, and depth of a 3D bounding box given its 8 corners.

    corners: List of 8 tuples representing the 3D coordinates of the bounding box corners
             (front-top-left, front-top-right, front-bottom-left, front-bottom-right,
              back-top-left, back-top-right, back-bottom-left, back-bottom-right)
    
    Returns:
        width: Width of the bounding box (x-direction)
        height: Height of the bounding box (y-direction)
        depth: Depth (or length) of the bounding box (z-direction)
    """
    ftl,ftr,fbl,fbr = classify_corners(front_corners) 
    btl,btr,btl,bbr = classify_corners(back_corners)

    # Calculate width (distance between front-top-left and front-top-right)
    width = euclidean_distance(ftl, ftr)
    
    # Calculate height (distance between front-top-left and front-bottom-left)
    height = euclidean_distance(ftl, fbl)
    
    # Calculate depth (distance between front-top-left and back-top-left)
    depth = euclidean_distance(ftl, btl)
    
    return width, height, depth

def parse_mask_segmentation_responses(mask_responses,obj_type): 
    """
    Given the responses return an int or list of ints corresponding to the masks that include 
    the desired object 
    """ 
    print("parsing mask segmentation responses ...")
    mask_idx = []
    if obj_type == 'ground':
        prompt = """These are LLM generated responses evaluating different image masks of the ground.
        Given these responses, which mask seems like it best defines the ground mask of the image? Please give your answer as a number for example, 'mask 1'.

        """ + str(mask_responses) + """

        Keep in mind that there probably won't be a perfect ground mask, just indicate which mask seems to be the best given these responses. 
        """
        #print("Prompt: ",prompt)
        response,history = generate_with_openai(prompt)
        #print("Response: ",response) 
        # Use regular expression to find all integers in the text
        integers = re.findall(r'\b\d+\b',response) 
        # Convert the found integers from strings to integers
        integers = np.unique([int(num) for num in integers]) 
        #print("Integers: ",integers)
        if len(integers) > 1:
            prompt = """Sorry, I really need to pick just one mask. Keep in mind that the best mask is the one that covers the most ground area while excluding objects 
            on the ground. If it helps, it is preferential to exclude more objects sitting on the ground than it is to have complete ground coverage. Please try to 
            include the number of only one of the masks in your response, to help with parsing. Also, be sure that your response is a number - for example 'mask 2' """

            #print("prompt:",prompt)
            response,_  = generate_with_openai(prompt,conversation_history=history)
            #print("response: ",response) 
            integers = re.findall(r'\b\d+\b',response) 
            # Convert the found integers from strings to integers
            integers = np.unique([int(num) for num in integers]) 
            return integers 
        elif len(integers) == 0: 
            return [0] 
        else:
            return integers 
    else:     
        for i,response in enumerate(mask_responses):  
            prompt = response + "\n" + "Does this response indicate there is a " + obj_type + "included in this bounding box? "
            #print("Prompt: ",prompt)
            obj_bool_response,_ = generate_with_openai(prompt)
            #print("Response: ",obj_bool_response) 
            if 'yes' in obj_bool_response.lower():
                mask_idx.append(i) 
    return mask_idx  

from concurrent.futures import ThreadPoolExecutor, as_completed

def combine_images_with_labels(annotated_images, labels, output_path):
    """
    Combine a list of images side by side and add labels at the top.
    """
    # Ensure all images have the same height by resizing if necessary
    max_height = max(img.shape[0] for img in annotated_images)
    resized_images = [cv2.resize(img, (img.shape[1], max_height)) for img in annotated_images]

    # Combine images horizontally
    combined_image = np.hstack(resized_images)

    # Add labels to each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color for text
    thickness = 2
    line_type = cv2.LINE_AA

    label_height = 50  # Space at the top for labels
    labeled_image = np.zeros((max_height + label_height, combined_image.shape[1], 3), dtype=np.uint8)
    labeled_image[label_height:, :] = combined_image  # Put the combined image below the label area

    # Draw each label centered over its image
    for i, label in enumerate(labels):
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = int(i * annotated_images[i].shape[1] + (annotated_images[i].shape[1] - text_size[0]) / 2)
        text_y = int((label_height + text_size[1]) / 2)
        cv2.putText(labeled_image, label, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

    # Save the final labeled image
    cv2.imwrite(output_path, labeled_image)
    return labeled_image
   

def audition_masks(obj_type, masks, image, img_time, frame, image_format='mono8', object_description=None, multiple_objects=False):
    """
    Pick the mask(s) which contain the given objects, with parallel processing.
    """
    print("Auditioning masks....")
    from .object_detection_yolo_client import ObjectDetectionClient 

    history = None  # Shared history, initialized as None 

    def process_obj_mask(i, mask, prompt, image_path, history):
        #print("processing mask ... this is mask {} of {}".format(i,len(masks)))
        #print("prompt: ",prompt) 
        ObjectDetectionClient.draw_boundingBox_image(mask,image,image_format,path=image_path) 
        response, history = generate_with_openai(prompt, conversation_history=history, image_path=image_path)
        #print("response: ",response) 
        return response, history

    # Parallelizing mask processing for 'ground' type
    if obj_type == "ground": 
        prompt = """I am trying to find the image mask which covers the ground in this image. The masks are blue and semi-transparent. 
                        The best ground mask covers the ground and excludes the most objects on the ground. Keep in mind, there is no blue carpet. 
                        If you see something that you think is blue carpet, that's the mask. Please return the number of the best ground mask. In order to 
                        parse your response, please only return ONE mask number.""" 

        labels = []
        for i in range(len(masks)):
            labels.append("Mask " + str(i)) 
        annotated_images = []

        for i, mask in enumerate(masks): 
            img_copy = copy.deepcopy(image) 
            annotated_image_msg = ObjectDetectionClient.annotate_image(mask, img_copy, 'mono8', None)
            np_arr = np.frombuffer(annotated_image_msg.data,np.uint8) 
            image_np = cv2.imdecode(np_arr,cv2.IMREAD_COLOR) 
            annotated_images.append(image_np)

        print("writing {} ...".format("/home/marble/debug_imgs/"+frame+"_"+str(img_time)+"_merged_ground_img.jpg"))
        combine_images_with_labels(annotated_images, labels, "/home/marble/debug_imgs/"+frame+"_"+str(img_time)+"_merged_ground_img.jpg")  

        final_response,history = generate_with_openai(prompt,image_path="/home/marble/debug_imgs/"+frame+"_"+str(img_time)+"_merged_ground_img.jpg")

        print("final_response: ",final_response)

        if final_response is None:
            raise OSError 
        integers = re.findall(r'\b\d+\b',final_response) 
        #print("integers: ",integers)
        # Convert the found integers from strings to integers
        best_mask_id = np.unique([int(num) for num in integers])  

        #print("best_mask_id: ",best_mask_id)
        if len(integers) == 0:
            raise OSError 

        if not best_mask_id in range(len(masks)):
            raise OSError 

        if len(best_mask_id) > 1:
            best_masks = [masks[i] for i in best_mask_id]
            return pick_largest_k_masks(best_masks, 1)
        elif len(best_mask_id) == 1:
            best_mask_id = best_mask_id[0]
    else:
        if multiple_objects: 
            prompt_base = """I am trying to find bounding boxes in this image which contain """ + obj_type + """. The user has provided this description of """ + obj_type + ": " + object_description + """
                            I will give you a few masks and I want you to tell me which ones contain a mask of a """ + obj_type + """, if any. This is mask 0:"""
        else: 
            prompt_base = """I am trying to find the bounding box in this image which contains a """ + obj_type + """. The user has provided this description of """ + obj_type + ": " + object_description + """
                            I will give you a few masks and I want you to tell me which one contains a mask of a """ + obj_type + """, if any. This is mask 0:"""

        if not os.path.exists("/home/marble/bb_detx_images"):
            os.mkdir("/home/marble/bb_detx_images") 

        print("iterating over {} masks".format(len(masks))) 

        mask_responses = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, mask in enumerate(masks):
                cv2.imwrite("/home/marble/bb_detx_images/bb_orig_image.jpg",image)  
                #ObjectDetectionClient.draw_boundingBox_image(mask, image, image_format, "/home/marble/bb_detx_images/bb_image" +str(i)+".jpg")
                prompt = prompt_base if i == 0 else "Does the green bounding box contain an image of a "+ obj_type +" according to the user description? This is mask " + str(i) + ":" 
                image_path = "/home/marble/bb_detx_images/bb_image" +str(i)+".jpg"
                print("writing {}".format(image_path))
                futures.append(executor.submit(process_obj_mask, i, mask, prompt, image_path, history))

            for future in as_completed(futures):
                response, history = future.result()
                #print("response: ",response) 
                if response is not None: 
                    mask_index = futures.index(future)
                    print("Mask {} response: {}".format(mask_index, response.encode('utf-8')))
                    if not multiple_objects and "yes" in response.lower():
                        return masks[futures.index(future)]  # Return the first valid mask if not handling multiple objects 
                    mask_responses.append(response) 
        
        if multiple_objects:
            best_mask_id = [i for i,x in enumerate(mask_responses) if "yes" in x.lower()]
        else:
            print(mask_responses) 
            print("this is weird ... it should hav returned when it found one good response")
            raise OSError 

    if best_mask_id is not None:
        #print("best_mask_id: ",best_mask_id) 
        #print("there is {} masks".format(len(masks))) 
        if isinstance(best_mask_id,list): 
            return_masks = [masks[i] for i in best_mask_id] 
        else:
            #print("type(best_mask_id):",type(best_mask_id))
            return_masks = masks[best_mask_id]
        return return_masks 
    else:
        return pick_largest_k_masks(masks, 1)

def pick_largest_k_masks(masks, k):
    """
    Pick the largest k masks based on the number of non-zero pixels (mask area).
    
    Parameters:
        masks (list of numpy arrays): List of binary masks (2D numpy arrays)
        k (int): Number of largest masks to return
    
    Returns:
        largest_k_masks (list of numpy arrays): List of the largest k masks
    """
    # Create a list to store the masks and their areas
    mask_area_list = []

    for mask in masks:
        # Calculate the area of the mask (number of non-zero pixels)
        mask_area = np.sum(mask > 0)
        mask_area_list.append((mask, mask_area))

    # Sort the masks by area in descending order
    mask_area_list = sorted(mask_area_list, key=lambda x: x[1], reverse=True)
    print('mask_areas: ',[x[1] for x in mask_area_list]) 

    # Select the top k largest masks
    largest_k_masks = [mask for mask, _ in mask_area_list[:k]]

    return largest_k_masks 

def filter_bottom_masks(masks): 
    filtered_masks = []

    for mask in masks: 
        # Get the height of the mask
        height = mask.shape[0]

        # Calculate the cutoff for the top 1/8th of the image
        top_region = height // 4

        # Check if any pixel in the top 1/8th is non-zero
        if not np.any(mask[:top_region, :]):
            # If no pixels are non-zero in the top 1/8th, keep the mask
            filtered_masks.append(mask)

    return filtered_masks
 
def get_ground_plane_z(ground_plane_msg):
    ground_plane_pc = pointcloud2_msg_to_array(ground_plane_msg) 
    return np.median(ground_plane_pc[:,2]) #is this actually z? 

def sample_ground_pts(px_coords, n=10):
    # Ensure that n does not exceed the number of available pixel coordinates
    n = min(n, len(px_coords))
    
    # Randomly choose 'n' indices without replacement
    random_indices = np.random.choice(len(px_coords), size=n, replace=False)

    # Select the coordinates at the random indices (use list comprehension)
    random_coords = [px_coords[i] for i in random_indices]
    
    return random_coords


if __name__ == '__main__':
    rospy.init_node('debug_service_calls')
    with open("/home/marble/filter_masks.pickle","rb") as handle:
        masks = pickle.load(handle)
    
    obj_size = {'width': 0.254, 'height': None, 'linear_length': None, 'area': None}

    image_frame = "cam_front_link"

    get_back_of_bbox = rospy.ServiceProxy('/H03/project_detections/get_back_of_bbox_from_pixel', ProjectBackOfBbox)
    response = get_back_of_bbox(int(17),int(4),image_frame) 
    print("response: ",response) 