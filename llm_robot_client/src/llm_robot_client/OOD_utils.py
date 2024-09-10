import numpy as np 
import cv2 
import re 
from llm_robot_client.srv import getPixelFrom3D,get3DPointFromPixel 
import rospy 
import tf.transformations as tft 
from shapely.geometry import Polygon 
import sensor_msgs.point_cloud2 as pc2 
from geometry_msgs.msg import Point 
from llm_utils import generate_with_openai 

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

def get_mask_centroid(binary_mask):
    """
    Get the center (centroid) of a binary mask.

    Parameters:
        binary_mask (numpy array): 2D binary mask with 1 for object pixels and 0 for background.
    
    Returns:
        (cx, cy): The (x, y) coordinates of the centroid of the mask.
    """
    # Get the indices of non-zero (object) pixels
    indices = np.argwhere(binary_mask > 0)

    if len(indices) == 0:
        return None  # No object found in the mask

    # Calculate the centroid as the mean of the x and y coordinates
    cy, cx = np.mean(indices, axis=0)  # Note: y comes first in numpy indexing

    return int(cx), int(cy)

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
    
    response,_ = generate_with_openai(prompt) 
    size_dict = extract_measurements(response) 
    return size_dict 
    
def is_ground_obj(obj_type,obj_description,image):
    """
    Given the object type and description, determine if this object is on the ground in this image
    """
    prompt = (
        "The user has defined " + obj_type + " like this: \n" + obj_description + "\n" +
        "Given this description and this image, would a " + obj_type +
        " be found on the ground in the kind of environment in this image?"
        )   
    cv2.imwrite('tmp_ground_img.jpg',image) 
    response,_ = generate_with_openai(prompt,'tmp_ground_img.jpg')
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

def get_ground_plane_pts(robot_tfs,robot_pose,ground_plane_pts,frame):
    """
    Get ground plane points that are in the field of view of the camera
    """ 
    x = robot_pose.position.x 
    y = robot_pose.position.y 
    z = robot_pose.position.z 
    rot_x = robot_pose.quaternion.x 
    rot_y = robot_pose.quaternion.y 
    rot_z = robot_pose.quaternion.z 
    rot_w = robot_pose.quaternion.w 
    euler = tft.euler_from_quaternion([rot_x, rot_y, rot_z, rot_w]) 

    frustrum_corners = Polygon(robot_tfs.get_front_cam_fov(frame,np.array([x,y,z,euler[0],euler[1],euler[2]]))) 
    pc_points = pointcloud2_msg_to_array(ground_plane_pts)
    inside = np.array([frustrum_corners.contains(Point(p)) for p in pc_points])
    points_inside_frustrum = pc_points[inside] 

    return points_inside_frustrum 

def inverse_projection(pts,frame):
    """
    Given a 3D camera coordinate, get the corresponding px coordinate 
    """
    px_coords = []
    #Wait for the service to become available
    rospy.wait_for_service('get_pixel_from_3d')

    for pt in pts: 
        try:
            # Create a proxy for the service
            get_pixel_from_3d = rospy.ServiceProxy('get_pixel_from_3d', getPixelFrom3D)
            
            # Call the service with the given 3D coordinates and camera frame
            response = get_pixel_from_3d(pt.x, pt.y, pt.z, frame)
            
            # Return the pixel coordinates from the response
            px_coords.append((response.u, response.v)) 

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return None, None 

    return px_coords

def annotate_image(mask,image):
    """
    Draw the image with the mask on top 
    """
    m = mask['segmentation']
    color_mask = np.concatenate([255,0,0],[0.35])
    image[m] = color_mask 
    return image 

def get_ground_masks(ground_mask,all_masks): 
    """
    Return ground_masks,non_ground_masks 
    That is, the masks that are in contact with the ground mask or not 
    """
    results = []
    
    for obj_mask in all_masks:
        # Perform element-wise AND between ground_mask and obj_mask
        intersection = np.logical_and(ground_mask, obj_mask['segmentation'])
        
        # Check if any pixel in the intersection is non-zero
        if np.any(intersection):
            results.append(True)
        else:
            results.append(False)
    
    return results

def get_bounding_box_from_mask(binary_mask):
    # Find the non-zero points (object points)
    coords = np.column_stack(np.where(binary_mask > 0))

    if len(coords) == 0:
        return None  # No object found in the mask

    # Find the minimum and maximum x, y coordinates
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    # Return the bounding box (x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max

def filter_masks(masks,image_data,obj_size,image_frame,obj_color=None): 
    if obj_color is not None: 
        filtered_masks = filter_masks_by_color(masks,image_data,obj_color)
    else:
        filtered_masks = masks 
    filtered_masks = filter_masks_by_size(filtered_masks,obj_size,image_frame)
    return filtered_masks 

def filter_masks_by_color(masks,image_data,obj_color):
    """
    Determine which masks have the right color 
    """
    correct_color_masks = []
    for mask in masks: 
        mask_bb = get_bounding_box_from_mask(mask['segmentation'])
        x_min, y_min, x_max, y_max = mask_bb 
        # Draw the bounding box on the image
        img_with_box = cv2.rectangle(image_data, (x_min, y_min), (x_max, y_max), (0,255,0), thickness=2)
        cv2.imwrite("img_with_box.jpg",img_with_box) 
        prompt = "Is the object within this bounding box " + obj_color + "?"
        response,_ = generate_with_openai(prompt,image_path="img_with_box.jpg") 
        if 'yes' in response.lower():
            correct_color_masks.append(mask) 

    return correct_color_masks     

def filter_masks_by_size(masks,obj_size,image_frame):
    """
    obj_size is a dictionary with area, width, height, linear_length 
    Determine which masks correspond to an object with that size 
    """
    rospy.wait_for_service('get_3d_point_from_pixel')

    appropriately_sized_masks = [] 
    for mask in masks:
        mask_bb = get_bounding_box_from_mask(mask['segmentation'])
        x_min, y_min, x_max, y_max = mask_bb  
        center_x = np.mean([x_min,x_max]) 
        center_y = np.mean([y_min,y_max]) 
        projected_points = {} 
        try:
            get_3d_point = rospy.ServiceProxy('get_3d_point_from_pixel', get3DPointFromPixel)
            response = get_3d_point(center_x, center_y, image_frame)
            center_pt = response.world_point
            projected_points['mask_center'] = center_pt 
            # Create a proxy for the service
            corners = [(x_min,y_min),(x_min,y_max),(x_max,y_min),(x_max,y_max)] 
            world_corner_pts = []
            for corner in corners: 
                u = corner[0]; v = corner[1] 
                try:
                    get_3d_point = rospy.ServiceProxy('get_3d_point_from_pixel', get3DPointFromPixel)
                    response = get_3d_point(u, v, image_frame)
                    world_corner_pts.append(response.world_point) 
                except rospy.ServiceException as e:
                    rospy.logerr("Couldn't project corner point. Service call failed: %s" % e)
            projected_points['bounding_box_front_pts'] = world_corner_pts 
            try:
                rospy.wait_for_service('get_back_of_bbox_from_pixel')
                get_back_of_bbox = rospy.ServiceProxy('get_back_of_bbox_from_pixel', ProjectBackOfBbox)
                response = get_back_of_bbox(center_x, center_y, image_frame)
                projected_points['bounding_box_back_pts'] = response.back_corners 
            except rospy.ServiceException as e:
                rospy.logerr("Couldn't find the back of the bounding box. Service call failed: %s" % e) 
        except rospy.ServiceException as e:
            rospy.logerr("Couldn't project center point. Service call failed: %s" % e)
        if compare_size(projected_points,obj_size):
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
        elif obj_size["length"]*0.5 <= est_width <= obj_size["width"]*1.5:
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
    sorted_by_y = sorted(corners, key=lambda p: p[1], reverse=True)  # Sort by y descending (top first)

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
    mask_idx = []
    if obj_type == 'ground':
        prompt = """These are LLM generated responses evaluating different image masks of the ground.
        Given these responses, which mask seems like it best defines the ground mask of the image?"""
        response,_ = generate_with_openai(mask_responses)
        # Use regular expression to find all integers in the text
        integers = re.findall(r'\b\d+\b',response) 
        # Convert the found integers from strings to integers
        integers = [int(num) for num in integers] 
        return integers 

    else:     
        for i,response in enumerate(mask_responses):  
            prompt = response + "\n" + "Does this response indicate there is a " + obj_type + "included in this image mask?"
            obj_bool_response = generate_with_openai(prompt)
            if 'yes' in obj_bool_response.lower():
                mask_idx.append(i) 
    return mask_idx  

def audition_masks(obj_type,masks,image,multiple_objects=False):
    """
    pick the mask(s) which contain the given objects
    """
    best_mask_id = None  
    if obj_type == "ground": 
        prompt = """I am trying to find the image mask which covers the ground in this image. I will give you a few masks
                        and I want you to tell me which one is the best ground mask. This is mask 0:""" 
        mask_responses = []
        history = None 
        for i,mask in enumerate(masks): 
            img_w_mask = annotate_image(mask,image) 
            cv2.write_image("masked_img"+str(i)+".jpg",img_w_mask)
            if i > 0:
                prompt = "Does this mask contain more of the ground in this image? This is mask " + str(i) + ":" 
            response,history = generate_with_openai(prompt,conversational_history=history,image_path=img_w_mask)  
            mask_responses.append(response) 
        best_mask_id = parse_mask_segmentation_responses(mask_responses,obj_type) 
        if len(best_mask_id) > 1:
            best_masks = [masks[i] for i in best_mask_id] 
            return pick_largest_k_masks(best_masks,1) 
        elif len(best_mask_id) == 1:
            best_mask_id = best_mask_id[0]
    else:
        if multiple_objects: 
            prompt = """I am trying to find the image masks which contain """ + obj_type + """. I will give you a few masks and 
            I want you to tell me which ones contain a mask of a """ + obj_type + ", if any. This is mask 0: "
        else: 
            prompt = """I am trying to find the image mask which contains a """ +obj_type+ """. I will give you a few masks and 
            I want you to tell me which one contains a mask of a """ + obj_type + ", if any. This is mask 0: "

            mask_responses = []
            history = None 
            for i,mask in enumerate(masks): 
                img_w_mask = annotate_image(mask,image) 
                cv2.write_image("masked_img"+str(i)+".jpg",img_w_mask)
                if i > 0:
                    prompt = "Does this mask contain an image of a "+ obj_type +"? This is mask " + str(i) + ":" 
                response,history = generate_with_openai(prompt,conversational_history=history,image_path=img_w_mask)  
                if not multiple_objects:
                    if "yes" in response.lower():
                        return mask  
                mask_responses.append(response) 

            if multiple_objects: 
                best_mask_id = parse_mask_segmentation_responses(mask_responses,obj_type) 

    if not best_mask_id is None: 
        return masks[best_mask_id]
    else:
        return pick_largest_k_masks(masks,1)

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

    # Select the top k largest masks
    largest_k_masks = [mask for mask, _ in mask_area_list[:k]]

    return largest_k_masks