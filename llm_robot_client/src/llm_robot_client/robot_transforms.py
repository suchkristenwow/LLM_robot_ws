import math 
import numpy as np 
import matplotlib.pyplot as plt 
from geometry_msgs.msg import Point 
from ros_numpy import numpify 
from sensor_msgs.msg import CameraInfo 
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R 
from shapely.geometry import Polygon
from image_geometry import PinholeCameraModel 
import rospy 
import toml 

def compute_fov_y(camera_info):
    # Extract the height of the image
    height = camera_info.height

    # Extract fy from the K matrix
    fy = camera_info.K[4]

    # Compute the field of view in the Y direction
    fov_y = 2 * math.atan(height / (2 * fy))

    # Convert the field of view from radians to degrees
    fov_y_degrees = math.degrees(fov_y)

    return fov_y_degrees

def calculate_distances(points):
    """
    Calculate the distances between adjacent points in a 4x2 array representing a rectangle.

    Parameters:
    points (np.ndarray): A 4x2 array where each row represents a point (x, y).

    Returns:
    list: A list of four distances.
    """
    distances = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]  # Next point, with wrap-around
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance)
    return distances

def get_width_and_length(points):
    """
    Get the width (shorter dimension) and length (longer dimension) of a rectangle given its vertices.

    Parameters:
    points (np.ndarray): A 4x2 array where each row represents a point (x, y).

    Returns:
    tuple: A tuple containing (width, length).
    """
    distances = calculate_distances(points)
    print("distances original: ",distances)
    distances = [np.round(x,2) for x in distances] 
    print("distances: ",distances)
    unique_distances = list(set(distances))  # Get unique distances

    if len(unique_distances) != 2:
        print("unique_distances:",unique_distances)
        raise ValueError("The given points do not form a proper rectangle")

    width, length = sorted(unique_distances)

    return width, length

def find_bottom_left_point(points):
    """
    Find the bottom-left point from a 4x2 array of points.

    Parameters:
    points (np.ndarray): A 4x2 array where each row represents a point (x, y).

    Returns:
    np.ndarray: The bottom-left point (x, y).
    """
    # Ensure the input is a 4x2 array
    if points.shape != (4, 2):
        raise ValueError("Input array must be of shape (4, 2)")
    
    # Find the bottom-left point
    # Sort by y first (ascending), then by x (ascending)
    sorted_points = points[np.lexsort((points[:, 0], points[:, 1]))]
    bottom_left_point = sorted_points[0]
    
    return bottom_left_point

class CamProjector:
    def __init__(self, depth, cameraInfo_topic, camera_pose, robot_pose):
        #TO DO: subscribe to camera info topic instead of this 
        #self.camera_model = CamProjector.get_camera_model()
        # Initialize the node
        #rospy.init_node('camera_model_node')
        # Create a PinholeCameraModel object
        self.camera_model = PinholeCameraModel()
        # Subscribe to the CameraInfo topic
        self.camera_info_sub = rospy.Subscriber(cameraInfo_topic, CameraInfo, self.camera_info_callback)
        # Initialize a flag to indicate when the camera model is set up
        self.camera_model_initialized = False
        self.camera_pose = camera_pose
        self.robot_pose = robot_pose
        self.depth = depth

    def camera_info_callback(self, msg):
        # Update the camera model with the received CameraInfo message
        self.camera_model.fromCameraInfo(msg)
        
        # Set the initialized flag to True
        self.camera_model_initialized = True

    def get_camera_model(self):
        # Wait for the camera model to be initialized
        while not self.camera_model_initialized and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        return self.camera_model
    
    @staticmethod
    def pose_to_transformation_matrix(pose):
        tf_matrix = np.zeros((4,4))
        r = R.from_euler("XYZ", pose[3:], degrees=False)
        tf_matrix[:3,:3] = r.as_matrix()
        tf_matrix[0,3] = pose[0]
        tf_matrix[1,3] = pose[1]
        tf_matrix[3,3] = pose[2]
        tf_matrix[3,3] = 1
        return tf_matrix

    def project_pixel(self, pixel):
        ray = np.asarray(self.camera_model.projectPixelTo3dRay(pixel))
        # Convert to Point
        point = ray * self.depth
        return point
    
    def convert_optical_to_nav(self, cam_point):
        cam_nav_frame_point = Point()
        cam_nav_frame_point.x = cam_point[2]
        cam_nav_frame_point.y = -1.0 *cam_point[0]
        cam_nav_frame_point.z = -1.0 * cam_point[1]
        return cam_nav_frame_point

    def apply_cam_transformation(self, point):
        #print("this is camera pose: ",self.camera_pose)
        cam_tf = self.pose_to_transformation_matrix(self.camera_pose)
        robot_tf = self.pose_to_transformation_matrix(self.robot_pose)
        # First apply cam_tf then robot_tf
        full_tf = np.dot(robot_tf, cam_tf)
        point_np = np.append(numpify(point),1)
        new_point = np.dot(full_tf, point_np)
        return new_point

    def project(self,pixel):
        print("pixel: ",pixel) 
        # Project To A Point in Camera frame
        cam_point = self.project_pixel(pixel)
        # Camera Point To Cam Nav Frame
        cam_point_frame = self.convert_optical_to_nav(cam_point)
        # Transfrom point
        new_point = self.apply_cam_transformation(cam_point_frame)
        return new_point[:2]

class robotTransforms: 
    def __init__(self,config_path,frustrum_length=25):
        '''
        fig, ax = plt.subplots(1,len(poses),figsize=(24,8)) #this is for the BEV animation thing 
        self.fig = fig; self.ax = ax  
        for sub_ax in self.ax:
            sub_ax.set_xlim((-6,6)) 
            sub_ax.set_ylim((-6,6)) 
            sub_ax.set_aspect('equal')  
        self.fig.tight_layout()  
        '''
        
        with open(config_path, "r") as f:
            self.settings = toml.load(f)

        self.left_down_camera_tf = self.settings["sensor_transforms"]["left_down_camera_tf"]
        self.right_down_camera_tf = self.settings["sensor_transforms"]["right_down_camera_tf"]

        self.front_camera_tf = self.settings["sensor_transforms"]["front_camera_tf"]
        self.right_camera_tf = self.settings["sensor_transforms"]["right_camera_tf"]
        self.left_camera_tf = self.settings["sensor_transforms"]["left_camera_tf"]

        self.left_front_tire_tf = self.settings["tire_transforms"]["left_front_tire_tf"]
        self.right_front_tire_tf = self.settings["tire_transforms"]["right_front_tire_tf"]
        self.left_back_tire_tf = self.settings["tire_transforms"]["left_back_tire_tf"]
        self.right_back_tire_tf = self.settings["tire_transforms"]["right_back_tire_tf"]

        self.tire_length = self.settings["tire_transforms"]["tire_length"]
        self.tire_width = self.settings["tire_transforms"]["tire_width"]

        self.planter_tf = self.settings["sensor_transforms"]["planter_tf"]

        self.fov = np.deg2rad(self.settings["robot"]["front_camera_fov_deg"])
        
        self.img_width = self.settings["robot"]["down_cam_img_width"]
        self.img_height = self.settings["robot"]["down_cam_img_height"]

        self.maxD = frustrum_length  
    
    def get_robot_transformation(self,robot_pose):
        robot_length = self.husky_dim[0]; robot_width = self.husky_dim[1] 
        x = robot_pose[0]; y = robot_pose[1]; yaw = robot_pose[5] 
        
        # Calculate the bottom-left corner of the rectangle considering the yaw angle
        offset = 0.05
        corner_x = x - robot_length * np.cos(yaw) + (robot_width / 2) * np.sin(yaw) + offset * np.cos(yaw)
        corner_y = y - robot_length * np.sin(yaw) - (robot_width / 2) * np.cos(yaw) + offset * np.sin(yaw)

        # Create the rectangle patch
        robot_rect = patches.Rectangle(
            (corner_x, corner_y), robot_length, robot_width,
            angle=np.degrees(yaw), edgecolor='black', facecolor='yellow', alpha=0.5
        )

        # Extract the corner points
        xmin, ymin = robot_rect.get_xy()
        width = robot_rect.get_width()
        height = robot_rect.get_height() 
        # Define the corners of the rectangle
        xmax = xmin + width
        ymax = ymin + height

        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        rectangle_polygon = Polygon(corners) 
        return corners 
        #return rectangle_polygon 
    
    def plot_robot(self,random_poses): 
        for i,robot_pose in enumerate(random_poses):
            robot_length = self.husky_dim[0]; robot_width = self.husky_dim[1] 
            x = robot_pose[0]; y = robot_pose[1]; yaw = robot_pose[5] 
            
            # Calculate the bottom-left corner of the rectangle considering the yaw angle
            offset = 0.05
            corner_x = x - robot_length * np.cos(yaw) + (robot_width / 2) * np.sin(yaw) + offset * np.cos(yaw)
            corner_y = y - robot_length * np.sin(yaw) - (robot_width / 2) * np.cos(yaw) + offset * np.sin(yaw)

            # Create the rectangle patch
            robot_rect = patches.Rectangle(
                (corner_x, corner_y), robot_length, robot_width,
                angle=np.degrees(yaw), edgecolor='black', facecolor='yellow', alpha=0.5
            )

            # Add the rectangle to the plot
            self.ax[i].add_patch(robot_rect)

            # Add an arrow to indicate the heading
            arrow_length = 0.5 * robot_length
            arrow_dx = arrow_length * np.cos(yaw)
            arrow_dy = arrow_length * np.sin(yaw)
            self.ax[i].arrow(x, y, arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

    def plot_down_cams(self,random_poses): 
        depth = self.down_cam_height 
        for i,robot_pose in enumerate(random_poses):
            #left 
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_down_camera_tf)) 
            self.ax[i].scatter(camera_tf_mat[0,3],camera_tf_mat[1,3],color="blue",label="left down camera")
            #depth, cameraInfo_topic, camera_pose, robot_pose 
            cam_projector = CamProjector(depth, self.left_down_cam_info_topic, camera_pose=self.left_down_camera_tf, robot_pose=robot_pose) 
            corners = [(0,0),(self.img_width,0),(self.img_width,self.img_height),(0,self.img_height)]
            points = np.array([cam_projector.project(c) for c in corners]); points = np.reshape(points,(4,2)) 
            self.ax[i].fill(points[:,0],points[:,1],color="blue",alpha=0.25)
            #right 
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_down_camera_tf)) 
            self.ax[i].scatter(camera_tf_mat[0,3],camera_tf_mat[1,3],color="red",label="right down camera")
            #depth, cameraInfo_topic, camera_pose, robot_pose
            cam_projector = CamProjector(depth, self.right_down_cam_info_topic, camera_pose=self.right_down_camera_tf, robot_pose=robot_pose) 
            corners = [(0,0),(self.img_width,0),(self.img_width,self.img_height),(0,self.img_height)]
            points = np.array([cam_projector.project(c) for c in corners]); points = np.reshape(points,(4,2)) 
            self.ax[i].fill(points[:,0],points[:,1],color="red",alpha=0.25) 

    def get_down_cam_fov(self,cam_type,robot_pose):
        depth = self.down_cam_height 
        if "left" in cam_type:
            #camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_down_camera_tf)) 
            #depth, cameraInfo_topic, camera_pose, robot_pose 
            cam_projector = CamProjector(depth, self.left_down_cam_info_topic, camera_pose=self.left_down_camera_tf, robot_pose=robot_pose) 
            corners = [(0,0),(self.img_width,0),(self.img_width,self.img_height),(0,self.img_height)]
            points = np.array([cam_projector.project(c) for c in corners]); points = np.reshape(points,(4,2)) 
            fov_polygon = Polygon(points)  
        else:
            #depth, cameraInfo_topic, camera_pose, robot_pose 
            cam_projector = CamProjector(depth, self.right_down_cam_info_topic, camera_pose=self.right_down_camera_tf, robot_pose=robot_pose) 
            corners = [(0,0),(self.img_width,0),(self.img_width,self.img_height),(0,self.img_height)]
            points = np.array([cam_projector.project(c) for c in corners]); points = np.reshape(points,(4,2)) 
            fov_polygon = Polygon(points)  

        return points 
        #return fov_polygon

    def get_front_cam_fov(self,cam_type,robot_pose): 
        if "left" in cam_type:
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_camera_tf))  
            cam_pose = [camera_tf_mat[0,3],camera_tf_mat[1,3],camera_tf_mat[3,3]] 
            cam_pose[2] = robot_pose[5] + np.pi/2 
        elif "right" in cam_type: 
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_camera_tf))  
            cam_pose = [camera_tf_mat[0,3],camera_tf_mat[1,3],camera_tf_mat[3,3]] 
            cam_pose[2] = robot_pose[5] - np.pi/2  
        elif "front" in cam_type:
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.front_camera_tf))  
            cam_pose = [camera_tf_mat[0,3],camera_tf_mat[1,3],camera_tf_mat[3,3]] 
            cam_pose[2] = robot_pose[5] 

        min_d = 0.1; max_d = self.maxD 
        min_theta = cam_pose[2] - self.fov/2
        max_theta = cam_pose[2] + self.fov/2
        x1 = cam_pose[0] + min_d*np.cos(min_theta)
        y1 = cam_pose[1] + min_d*np.sin(min_theta)
        x0 = cam_pose[0] + min_d*np.cos(max_theta)
        y0 = cam_pose[1] + min_d*np.sin(max_theta)
        x2 = cam_pose[0] + max_d*np.cos(min_theta)
        y2 = cam_pose[1] + max_d*np.sin(min_theta)
        x3 = cam_pose[0] + max_d*np.cos(max_theta)
        y3 = cam_pose[1] + max_d*np.sin(max_theta)

        points = [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] 
        return points 
        #return Polygon(points)

    def plot_cam_frustrums(self,random_poses): 
        self.plot_down_cams(random_poses)
        for i,robot_pose in enumerate(random_poses): 
            print("robot_pose: ",robot_pose) 
            #left cam 
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_camera_tf))  
            cam_pose = [camera_tf_mat[0,3],camera_tf_mat[1,3],camera_tf_mat[3,3]] 
            cam_pose[2] = robot_pose[5] + np.pi/2 
            self.ax[i].scatter(cam_pose[0],cam_pose[1],marker="*",color="b")
            min_d = 0.1; max_d = self.maxD 
            min_theta = cam_pose[2] - self.fov/2
            max_theta = cam_pose[2] + self.fov/2
            x1 = cam_pose[0] + min_d*np.cos(min_theta)
            y1 = cam_pose[1] + min_d*np.sin(min_theta)
            x0 = cam_pose[0] + min_d*np.cos(max_theta)
            y0 = cam_pose[1] + min_d*np.sin(max_theta)
            x2 = cam_pose[0] + max_d*np.cos(min_theta)
            y2 = cam_pose[1] + max_d*np.sin(min_theta)
            x3 = cam_pose[0] + max_d*np.cos(max_theta)
            y3 = cam_pose[1] + max_d*np.sin(max_theta)
            self.ax[i].plot([x0,x1],[y0,y1],'b')
            self.ax[i].plot([x1,x2],[y1,y2],'b')
            self.ax[i].plot([x2,x3],[y2,y3],'b')
            self.ax[i].plot([x3,x0],[y3,y0],'b')   
            #right cam 
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_camera_tf))  
            cam_pose = [camera_tf_mat[0,3],camera_tf_mat[1,3],camera_tf_mat[3,3]] 
            cam_pose[2] = robot_pose[5] - np.pi/2 
            self.ax[i].scatter(cam_pose[0],cam_pose[1],marker="*",color="r")
            min_d = 0.1; max_d = self.maxD 
            min_theta = cam_pose[2] - np.deg2rad(self.fov/2)
            max_theta = cam_pose[2] + np.deg2rad(self.fov/2)
            x1 = cam_pose[0] + min_d*np.cos(min_theta)
            y1 = cam_pose[1] + min_d*np.sin(min_theta)
            x0 = cam_pose[0] + min_d*np.cos(max_theta)
            y0 = cam_pose[1] + min_d*np.sin(max_theta)
            x2 = cam_pose[0] + max_d*np.cos(min_theta)
            y2 = cam_pose[1] + max_d*np.sin(min_theta)
            x3 = cam_pose[0] + max_d*np.cos(max_theta)
            y3 = cam_pose[1] + max_d*np.sin(max_theta)
            self.ax[i].plot([x0,x1],[y0,y1],'r')
            self.ax[i].plot([x1,x2],[y1,y2],'r')
            self.ax[i].plot([x2,x3],[y2,y3],'r')
            self.ax[i].plot([x3,x0],[y3,y0],'r')   
            #front cam 
            camera_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.front_camera_tf))  
            cam_pose = [camera_tf_mat[0,3],camera_tf_mat[1,3],camera_tf_mat[3,3]] 
            cam_pose[2] = robot_pose[5]
            self.ax[i].scatter(cam_pose[0],cam_pose[1],marker="*",color="purple")
            min_d = 0.1; max_d = self.maxD 
            min_theta = cam_pose[2] - np.deg2rad(self.fov/2)
            max_theta = cam_pose[2] + np.deg2rad(self.fov/2)
            x1 = cam_pose[0] + min_d*np.cos(min_theta)
            y1 = cam_pose[1] + min_d*np.sin(min_theta)
            x0 = cam_pose[0] + min_d*np.cos(max_theta)
            y0 = cam_pose[1] + min_d*np.sin(max_theta)
            x2 = cam_pose[0] + max_d*np.cos(min_theta)
            y2 = cam_pose[1] + max_d*np.sin(min_theta)
            x3 = cam_pose[0] + max_d*np.cos(max_theta)
            y3 = cam_pose[1] + max_d*np.sin(max_theta)
            self.ax[i].plot([x0,x1],[y0,y1],'purple')
            self.ax[i].plot([x1,x2],[y1,y2],'purple')
            self.ax[i].plot([x2,x3],[y2,y3],'purple')
            self.ax[i].plot([x3,x0],[y3,y0],'purple')   

    def tire_plot_helper_function(self,x,y,angle): 
        # Define the corners of the rectangle in the local frame (centered at the origin)
        rectangle_corners_local = np.array([
            [-self.tire_length / 2, -self.tire_width / 2],  # Bottom-left
            [self.tire_length / 2, -self.tire_width / 2],   # Bottom-right
            [self.tire_length / 2, self.tire_width / 2],    # Top-right
            [-self.tire_length / 2, self.tire_width / 2]    # Top-left
        ])

        # Rotation matrix for the given angle
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Rotate the corners to align with the global coordinate system
        rectangle_corners_global = np.dot(rectangle_corners_local, rotation_matrix.T)

        # Translate the rectangle to its center point
        rectangle_corners_global += np.array([x, y])

        # Close the loop for the rectangle
        rectangle_corners_global = np.vstack([rectangle_corners_global, rectangle_corners_global[0]])  

        return rectangle_corners_global 
    
    def get_tire_corners(self,robot_pose,tire_type): 
        if "front" in tire_type: 
            if 'right' in tire_type:
                tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_front_tire_tf))  
            else: 
                tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_front_tire_tf))  
        else:
            if 'right' in tire_type:
                tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_back_tire_tf))  
            else:
                tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_back_tire_tf))  
        
        #self.ax[i].scatter(tire_tf_mat[0,3],tire_tf_mat[1,3],color='orange',alpha=0.5)
        tire_corners = self.tire_plot_helper_function(tire_tf_mat[0,3],tire_tf_mat[1,3],robot_pose[-1]) 
        return tire_corners 

    def plot_tires(self,random_poses):
        for i, robot_pose in enumerate(random_poses): 
            #tire center 
            tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_back_tire_tf))  
            #self.ax[i].scatter(tire_tf_mat[0,3],tire_tf_mat[1,3],color='orange',alpha=0.5)
            tire_corners = self.tire_plot_helper_function(tire_tf_mat[0,3],tire_tf_mat[1,3],robot_pose[-1]) 
            self.ax[i].fill(tire_corners[:,0],tire_corners[:,1],color='orange',alpha=0.5) 
            tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.left_front_tire_tf))   
            #self.ax[i].scatter(tire_tf_mat[0,3],tire_tf_mat[1,3],color='green',alpha=0.5) 
            tire_corners = self.tire_plot_helper_function(tire_tf_mat[0,3],tire_tf_mat[1,3],robot_pose[-1]) 
            self.ax[i].fill(tire_corners[:,0],tire_corners[:,1],color='green',alpha=0.5) 
            tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_back_tire_tf))   
            #self.ax[i].scatter(tire_tf_mat[0,3],tire_tf_mat[1,3],color='c',alpha=0.5) 
            tire_corners = self.tire_plot_helper_function(tire_tf_mat[0,3],tire_tf_mat[1,3],robot_pose[-1]) 
            self.ax[i].fill(tire_corners[:,0],tire_corners[:,1],color='c',alpha=0.5) 
            tire_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.right_front_tire_tf))   
            #self.ax[i].scatter(tire_tf_mat[0,3],tire_tf_mat[1,3],color='m',alpha=0.5) 
            tire_corners = self.tire_plot_helper_function(tire_tf_mat[0,3],tire_tf_mat[1,3],robot_pose[-1]) 
            self.ax[i].fill(tire_corners[:,0],tire_corners[:,1],color='m',alpha=0.5) 

    def get_robot_pose_from_planter(self, planter_position):
        if not isinstance(planter_position,np.ndarray): 
            tmp = np.zeros((2,))
            tmp[0] = planter_position.x; tmp[1] = planter_position.y 
            planter_position = tmp 

        # Convert the planter position to a homogeneous coordinate
        planter_coord_homogeneous = np.array([planter_position[0], planter_position[1], 0, 1])

        # Compute the transformation matrix from the robot to the planter
        robot_to_planter_tf = CamProjector.pose_to_transformation_matrix(self.planter_tf)

        # Invert the transformation matrix
        planter_to_robot_tf = np.linalg.inv(robot_to_planter_tf)

        # Apply the inverse transformation to get the robot position
        robot_coord_homogeneous = planter_to_robot_tf.dot(planter_coord_homogeneous)

        # Extract the robot position from the homogeneous coordinates
        robot_position = robot_coord_homogeneous[:3]  # Extract the x, y, z positions
        robot_orientation = R.from_matrix(planter_to_robot_tf[:3, :3]).as_euler('XYZ')  # Extract the orientation

        # Combine position and orientation into a single pose
        robot_pose = np.concatenate((robot_position, robot_orientation))

        return robot_pose

    def get_planter_position(self,robot_pose): 
        planter_tf_mat = CamProjector.pose_to_transformation_matrix(robot_pose).dot(CamProjector.pose_to_transformation_matrix(self.planter_tf))   
        planter_coord = np.array([planter_tf_mat[0,3],planter_tf_mat[1,3]]) 
        return planter_coord 
