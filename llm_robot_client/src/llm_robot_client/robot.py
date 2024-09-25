import rospy
import requests
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Point, PoseStamped
from std_msgs.msg import Header, String, Bool, ColorRGBA
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from llm_robot_client.msg import ProjectedDetectionArray
from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import numpy as np
import threading
from collections import deque
import message_filters
import copy
#from urllib.parse import urlparse
from urlparse import urlparse
import math
from std_srvs.srv import Trigger
from planner_msgs.srv import (
    pci_initialization,
    pci_global,
    planner_go_to_waypoint,
    pci_search,
    pci_searchRequest, 
    pci_searchResponse
)

class CircularBuffer(deque):
    def __init__(self, size=0):
        #super().__init__(maxlen=size)
        super(CircularBuffer, self).__init__(maxlen=size)
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            self.append(item)
    
    def get(self):
        with self.lock:
            if len(self) == 0:
                return None
            return self.pop()


class PlannerController:
    def __init__(self):
        # Service clients
        self.planner_client_start_planner = rospy.ServiceProxy(
            "planner_control_interface/std_srvs/automatic_planning", Trigger
        )
        self.planner_client_stop_planner = rospy.ServiceProxy(
            "planner_control_interface/std_srvs/stop", Trigger
        )
        self.planner_client_homing = rospy.ServiceProxy(
            "planner_control_interface/std_srvs/homing_trigger", Trigger
        )
        self.planner_client_init_motion = rospy.ServiceProxy(
            "pci_initialization_trigger", pci_initialization
        )
        self.planner_client_plan_to_waypoint = rospy.ServiceProxy(
            "planner_control_interface/std_srvs/go_to_waypoint", Trigger
        )
        self.planner_client_global_planner = rospy.ServiceProxy(
            "pci_global", pci_global
        )
        self.planner_client_plan_to_waypoint_with_pose = rospy.ServiceProxy(
            "planner_control_interface/std_srvs/planner_go_to_waypoint",
            planner_go_to_waypoint,
        )
        self.planner_client_search = rospy.ServiceProxy(
            "pci_search", pci_search
        )

        # Estop For Controller
        # Stop Robot Publisher
        stop_topic = rospy.get_param("~stop_topic", "estop")
        self.stop_pub = rospy.Publisher(stop_topic, Bool, queue_size=1)
        self.robot_stopped = True
        self.planner_stoped = True

    def start_motion(self):
        rospy.logdebug("Starting Motion")
        self.stop_pub.publish(False)
        self.robot_stopped = False

    def stop_motion(self):
        rospy.logdebug("Stopping Motion")
        self.stop_pub.publish(True)
        self.robot_stopped = True

    def start_planner(self):
        try:
            response = self.planner_client_start_planner()
            self.planner_stoped = False
            rospy.loginfo("Start Planner Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def stop_planner(self):
        try:
            response = self.planner_client_stop_planner()
            self.planner_stoped = True
            rospy.loginfo("Stop Planner Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def homing(self):
        try:
            response = self.planner_client_homing()
            rospy.loginfo("Homing Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def init_motion(self):
        try:
            response = self.planner_client_init_motion()
            rospy.loginfo("Init Motion Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def plan_to_waypoint(self):
        try:
            response = self.planner_client_plan_to_waypoint()
            rospy.loginfo("Plan to Waypoint Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            
    def search(self):
        try:
            # Create a request object for the pci_search service
            planner_search_req = pci_searchRequest()
            planner_search_req.header.stamp = rospy.Time.now()
            planner_search_req.header.frame_id = "map"  # Make sure this is the correct frame ID for your application
            planner_search_req.use_current_state = True
            planner_search_req.not_exe_path = False
            planner_search_req.bound_mode = pci_searchRequest.kNoBound  # Assuming kExtendedBound is a constant defined in the service definition

            # Call the service with the request object
            response = self.planner_client_search(planner_search_req)
            rospy.loginfo("Search Response: %s", response.success)
            if response.success:
                rospy.loginfo("Path found with %d poses", len(response.path))
            else:
                rospy.logwarn("Search failed to find a path")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)


    def plan_to_waypoint_with_pose(self, pose):
        try:
            # Stop motion and stop planner
            self.stop_planner()
            # check collision should be true
            rospy.loginfo("Planning to Waypoint With Pose at: %s", pose)
            response = self.planner_client_plan_to_waypoint_with_pose(True, pose)
            # self.start_motion()
            rospy.loginfo("Plan to Waypoint With Pose Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def global_planner(self, id):
        try:
            response = self.planner_client_global_planner(id=id)
            rospy.loginfo("Global Planner Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)


class Robot:
    def __init__(self):
        self.is_ready = False
        # Keep Track of Current Waypoint
        self.current_waypoint = None
        self.current_odom_msg = None
        self.waypoint_status = None
        self.current_path = None
        self.current_global_graph_points = []
        self.current_local_graph_points = []
        self.current_frontier_points = []
        self.filtered_graph_points = None
        self.all_objects = None
        self.current_object_detections = None
        self.current_object_detections_last_update = None
        self.frame_width = None
        self.last_movement_time = None
        self.waypoint_plan_status = None
        self.planner_mode = None
        self.object_interrupt = False
        self.enable_movement_time = False

        self.object_detector_status = False
        self.reached_end_goal = False
        
   
        
        # Locks
        self.odom_lock = threading.Lock()
        self.path_lock = threading.Lock()
        self.all_objects_lock = threading.Lock()
        self.current_objects_lock = threading.Lock()
        self.frontier_points_lock = threading.Lock()
        self.global_graph_points_lock = threading.Lock()
        self.local_graph_points_lock = threading.Lock()
        self.waypoint_lock = threading.Lock()
        self.is_ready_lock = threading.Lock()
        self.end_goal_lock = threading.Lock()
        self.planner_mode_lock = threading.Lock()
        self.object_status_lock = threading.Lock()
        #self.waypoint_status_lock = threading.Lock()

        self.use_glip = rospy.get_param("~use_glip", False)

        self.eps_dbscan = rospy.get_param("~eps_dbscan", 0.5)
        self.min_samples = rospy.get_param("~min_samples", 1)
        self.visualize_graph_points = rospy.get_param("~visualize_graph_points", True)
        self.enable_naive_exploration = rospy.get_param("~enable_naive_exploration", False)
        self.is_replanning = True
        
        # Only used during navie exploration with vision
        #self.target_object_name = rospy.get_param("~target_object_name", "fire extinguisher")
        
        self.visited_objects = []
        self.navigating_to_object = False

        # Planner Controller
        self.planner_controller = PlannerController()

        # Check if robot has moved forward for initialization
        # Start planner when called
        is_ready_topic = rospy.get_param("~ready_topic", "ready")
        self.is_ready_sub = rospy.Subscriber(
            is_ready_topic, Bool, self.is_ready_callback, queue_size=1
        )
        rospy.loginfo("[llm_robot] Subscribed to Ready: %s", is_ready_topic)

        # Odom Functions
        odom_topic = rospy.get_param("~odom_topic", "odom")
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self.odom_callback, queue_size=1
        )
        rospy.loginfo("[llm_robot] Subscribed to Odom: %s", odom_topic)

        self.current_odom = None

        # Path Functions
        path_topic = rospy.get_param("~path_topic", "path")
        self.path_sub = rospy.Subscriber(
            path_topic, Path, self.path_callback, queue_size=1
        )
        rospy.loginfo("[llm_robot] Subscribed to Path: %s", path_topic)

        # Frontier Points (Local)
        local_graph_points_topic = rospy.get_param(
            "~local_graph_points_topic", "local_points"
        )
        self.local_graph_points_sub = rospy.Subscriber(
            local_graph_points_topic,
            PoseArray,
            self.local_graph_points_callback,
            queue_size=1,
        )
        rospy.loginfo(
            "[llm_robot] Subscribed to Local Graph Points: %s",
            local_graph_points_topic,
        )

        # Global Graph
        global_graph_points_topic = rospy.get_param(
            "~global_graph_points_topic", "global_points"
        )
        self.global_graph_points_sub = rospy.Subscriber(
            global_graph_points_topic,
            PoseArray,
            self.global_graph_points_callback,
            queue_size=1,
        )
        rospy.loginfo(
            "[llm_robot] Subscribed to Global Graph Points: %s",
            global_graph_points_topic,
        )

        frontier_points_topic = rospy.get_param(
            "~frontier_points_topic", "frontier_points"
        )
        self.frontier_points_sub = rospy.Subscriber(
            frontier_points_topic,
            PoseArray,
            self.frontier_points_callback,
            queue_size=1,
        )

        # All Objects
        all_objects_topic = rospy.get_param("~all_objects_topic", "all_objects_topic")
        self.all_objects_sub = rospy.Subscriber(
            all_objects_topic,
            ProjectedDetectionArray,
            self.all_objects_callback,
            queue_size=1,
        )
        rospy.loginfo(
            "[llm_robot] Subscribed to Projected Objects: %s",
            all_objects_topic,
        )

        # Current Object Detections
        current_object_detections_topic = rospy.get_param(
            "~current_objects_topic", "current_objects_topic"
        )
        self.current_object_detections_sub = rospy.Subscriber(
            current_object_detections_topic,
            ProjectedDetectionArray,
            self.current_object_detecions_callback,
            queue_size=1,
        )
        rospy.loginfo(
            "[llm_robot] Subscribed to Current Object Detections: %s",
            current_object_detections_topic,
        )

        # Object Detection Status
        self.object_detector_status_topic = rospy.get_param(
            "~object_detector_status_topic", "object_detector_status"
        )
        self.object_detector_status_sub = rospy.Subscriber(
            self.object_detector_status_topic,
            Bool,
            self.object_detector_status_callback,
            queue_size=1,
        )

        # Get End Goal State From Unreal
        self.end_goal_topic = rospy.get_param(
            "~end_goal_topic", "/unreal/game_terminated"
        )

        self.end_goal_sub = rospy.Subscriber(
            self.end_goal_topic,
            Bool,
            self.end_goal_callback,
            queue_size=1,
        )

        #  Planning Status Flag
        self.planning_mode_topic = rospy.get_param(
            "~planning_mode_topic", "/planning_status"
        )
        self.planning_status_sub = rospy.Subscriber(
            self.planning_mode_topic,
            String,
            self.planning_mode_callback,
            queue_size=1,
        )

        self.waypoint_plan_status_topic = rospy.get_param(
            "~waypoint_plan_status_topic", "/waypoint_plan_status"
        )
        rospy.loginfo(
            "[llm_robot] Subscribed to waypoint plan status topic: %s",
            self.waypoint_plan_status_topic,
        )
        '''
        self.waypoint_plan_status_sub = rospy.Subscriber(
            self.waypoint_plan_status_topic,
            WaypointPlanStatus,
            self.waypoint_plan_status_callback,
            queue_size=1,
        )
        '''
        # Object Label Publisher
        object_list_topic = rospy.get_param("~object_list_topic", "object_list")
        self.object_list_pub = rospy.Publisher(
            object_list_topic, String, queue_size=1, latch=True
        )

        # Filtered Graph Points Publisher (Visualization)
        filtered_graph_points_viz_topic = rospy.get_param(
            "~filtered_graph_points_viz_topic", "filtered_graph_points"
        )
        self.filtered_graph_points_viz_pub = rospy.Publisher(
            filtered_graph_points_viz_topic, Marker, queue_size=1, latch=True
        )

        waypoint_marker_topic = self.waypoint_plan_status_topic + "/viz"
        rospy.loginfo(
            "[llm_robot] waypoint marker topic: %s", waypoint_marker_topic
        )
        self.waypoint_marker_pub = rospy.Publisher(
            waypoint_marker_topic, Marker, queue_size=1
        )
        
        self.toggle_cameras_sub = rospy.Subscriber(
            "toggle_cameras", Bool, self.toggle_cameras_callback, queue_size=1
        )
        
        self.toggle_cameras_pub = rospy.Publisher( "toggle_cameras", Bool, queue_size=1)

        self.host_url = rospy.get_param("~host_url", "http://localhost:5000")

        # CV Bridge
        self.bridge = CvBridge()

        self.subscribe_to_cameras()

        self.waypoint_threshold = rospy.get_param("~waypoint_threshold", 0.5)

        # Open the file in read mode
        with open('/home/marble/LLM_robot_ws/src/llm_robot_client/target_objects.txt', 'r') as file:
            # Read all lines and store them in a list
            lines = [line.strip() for line in file]  

        self.target_object_names = lines 

    def initialize_object_detection_server(self):
        object_list = self.target_object_names
        formatted_object_labels = ",".join(object_list)
        self.object_list_pub.publish(formatted_object_labels)
        
    def run(self):
        if self.is_ready:
            if self.enable_naive_exploration == True:
                self.target_object_controller_run()
            else:   
                self.waypoint_controller_run()
                
    def target_object_controller_run(self):
        rospy.loginfo_throttle(1, "Running Target Object Controller")
        if not self.navigating_to_object:
            target_object = self.check_target_object()
            if target_object is not None:
                self.planner_controller.stop_motion()
                self.planner_controller.stop_planner()
                self.set_target_object_waypoint(target_object)
                self.navigating_to_object = True
            else:
                if self.planner_controller.robot_stopped:
                    self.planner_controller.start_motion()
                    self.planner_controller.start_planner()
                self.explore_controller_run()
        else:
            time_since_last_move = rospy.get_time() - self.last_movement_time
            if time_since_last_move < 10:
                if self.check_waypoint_ready():
                    self.waypoint_controller_run()
                else:
                    self.planner_controller.stop_motion()
            else:
                self.navigating_to_object = False
                rospy.loginfo("Target object navigation failed, replanning")
        
        
    def check_target_object(self):
        """
        Checks if the target object has been detected and returns its position
        if it hasn't been visited before and is within a certain distance threshold.

        Args:
            distance_threshold (float): The maximum allowable distance to consider the object as 'found'.

        Returns:
            dict: The position of the detected target object if it hasn't been visited before, otherwise None.
        """
        rospy.loginfo_throttle(1, "All detected objects: %s", self.all_objects)
        rospy.loginfo_throttle(1, "Vistied objects: %s", self.visited_objects)
        for index, obj in self.all_objects.items():
            if obj["name"] in self.target_object_names:
                position = obj["position"]
                if not self.target_has_been_visited(position):
                    rospy.loginfo("Target object found at position: %s", position)
                    self.visited_objects.append(position)  # Add the object position to the visited list
                    return position
        return None
    
    
    
    def target_has_been_visited(self, position):
        """
        Checks if an object has been visited before based on the waypoint threhsold.

        Args:
            position (dict): The position to check with keys 'x', 'y', 'z'.

        Returns:
            bool: True if the position has been visited before, False otherwise.
        """
        for visited_object in self.visited_objects:
            if self.calculate_distance(position, visited_object) < self.waypoint_threshold:
                return True
        return False
    
    def calculate_distance(self, pos1, pos2):
        """
        Calculates the Euclidean distance between two positions.

        Args:
            pos1 (dict): The first position with keys 'x', 'y', 'z'.
            pos2 (dict): The second position with keys 'x', 'y', 'z'.

        Returns:
            float: The Euclidean distance between the two positions.
        """
        return math.sqrt((pos1["x"] - pos2["x"]) ** 2 + (pos1["y"] - pos2["y"]) ** 2 + (pos1["z"] - pos2["z"]) ** 2)
        
        
                        
    def waypoint_controller_run(self):
        """
        Executes the waypoint controller logic.

        This method checks the waypoint status and performs the necessary actions based on the status.
        If the waypoint status indicates that the robot is ready, arrived, and the motion is planned,
        it stops the motion and starts the planner to get new points.
        If the waypoint status indicates that the robot is ready but not arrived yet, and the motion is stopped,
        it starts the motion.
        Used for llm_gudaiance

        Returns:
            None
        """
        rospy.loginfo_throttle(1, "Running Waypoint Controller")
        self.check_waypoint_status()
        with self.waypoint_status_lock:
            if self.waypoint_status is not None:
                if self.waypoint_status["Ready"] == True and self.waypoint_status["Arrived"] == True and self.waypoint_status["Planned"] == True:
                    if not self.planner_controller.robot_stopped:
                        self.planner_controller.stop_motion()
                        self.enable_arrvial_time = False
                        # Get new points
                        self.planner_controller.stop_planner()
                        self.planner_controller.start_planner()
                        if self.enable_naive_exploration:
                            rospy.loginfo("Target object navigation successful, Disabling target waypoint controller!")
                            self.navigating_to_object = False
                elif self.waypoint_status["Ready"] == True and self.waypoint_status["Arrived"] == False:
                    if self.planner_controller.robot_stopped and not self.object_interrupt:
                        self.planner_controller.start_motion()
                        
                        
    def explore_controller_run(self):
        """
        Runs the explore controller.

        For naive exploration. This makes sure the robot is moving and if it's not
        the planner is toggled to generate a new path
        Returns:
            None
        """
        rospy.loginfo_throttle(1, "Running Explore Controller")
        if self.last_movement_time is not None and self.is_ready:
            time_since_last_move = rospy.get_time() - self.last_movement_time
            rospy.loginfo_throttle(1, "Last Movement Time: %s", time_since_last_move)
            if  time_since_last_move > 15 and not self.is_replanning:
                self.planner_controller.stop_motion()
                self.planner_controller.stop_planner()
                self.planner_controller.start_planner()
                self.planner_controller.start_motion()
                self.is_replanning = True
            elif time_since_last_move <= 1 and self.is_replanning:
                self.is_replanning = False
                

    def subscribe_to_cameras(self):
        self.image_queue = CircularBuffer(size=2)

        cam_front_topic = rospy.get_param("~cam_front_topic", "image")
        cam_left_topic = rospy.get_param("~cam_left_topic", "image")
        cam_right_topic = rospy.get_param("~cam_right_topic", "image")
        rospy.loginfo("cam_front_topic: %s", cam_front_topic)
        rospy.loginfo("cam_left_topic: %s", cam_left_topic)
        rospy.loginfo("cam_right_topic: %s", cam_right_topic)

        self.image_front_sub = message_filters.Subscriber(
            cam_front_topic, CompressedImage
        )
        self.image_left_sub = message_filters.Subscriber(
            cam_left_topic, CompressedImage
        )
        self.image_right_sub = message_filters.Subscriber(
            cam_right_topic, CompressedImage
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_front_sub, self.image_left_sub, self.image_right_sub],
            queue_size=10,
            slop=0.1,
        )

        self.ts.registerCallback(self.image_callback)
        self.cam_frame_ids = {"right": None, "front": None, "left": None}
        
    def unsubscribe_from_cameras(self):
        # Check if the subscribers exist and then unregister them.
        if self.image_front_sub is not None:
            self.image_front_sub.sub.unregister()  # Unregister the underlying rospy.Subscriber
            self.image_front_sub = None

        if self.image_left_sub is not None:
            self.image_left_sub.sub.unregister()  # Unregister the underlying rospy.Subscriber
            self.image_left_sub = None

        if self.image_right_sub is not None:
            self.image_right_sub.sub.unregister()  # Unregister the underlying rospy.Subscriber
            self.image_right_sub = None
            
        self.image_queue = None
        self.ts = None
        rospy.loginfo("Unsubscribed from camera topics.")
        
    def toggle_cameras_callback(self, msg):
        if msg.data == True:
            self.unsubscribe_from_cameras()
            self.subscribe_to_cameras()

    def image_callback(self, front, left, right):
        """
        Callback function for processing image messages.
        Called from time synchronizer.

        Args:
            msg: The image message received.
            front: The front image message.
            left: The left image message.
            right: The right image message.
        """
        cv_image_front = self.bridge.compressed_imgmsg_to_cv2(
            front, desired_encoding="bgr8"
        )
        cv_image_left = self.bridge.compressed_imgmsg_to_cv2(
            left, desired_encoding="bgr8"
        )
        cv_image_right = self.bridge.compressed_imgmsg_to_cv2(
            right, desired_encoding="bgr8"
        )
        concatenated_image = self.concatenate_images(
            cv_image_front, cv_image_left, cv_image_right
        )

        # Check if frame_ids have been gathered
        if self.cam_frame_ids["front"] is None:
            self.cam_frame_ids["front"] = front.header.frame_id
        if self.cam_frame_ids["left"] is None:
            self.cam_frame_ids["left"] = left.header.frame_id
        if self.cam_frame_ids["right"] is None:
            self.cam_frame_ids["right"] = right.header.frame_id

        if self.frame_width is None:
            self.frame_width = concatenated_image.shape[1] / 3

        # rospy.loginfo("[llm_robot_client] Storing Image")
        self.image_queue.put({"time": front.header.stamp, "image": concatenated_image})

    def concatenate_images(self, front, left, right):
        """
        Concatenates three images horizontally and publishes the resulting image.

        Args:
            front (numpy.ndarray): The front image.
            left (numpy.ndarray): The left image.
            right (numpy.ndarray): The right image.

        Returns:
            numpy.ndarray: The concatenated image.
        """
        threshold = 15
        left = copy.deepcopy(left)
        right = copy.deepcopy(right)
        front = copy.deepcopy(front)
        left_image = self.crop_image(left, threshold)
        front_image = self.crop_image(front, threshold)
        right_image = self.crop_image(right, threshold)
        single_image = np.concatenate((left_image, front_image, right_image), axis=1)
        return single_image

    def crop_image(self, image, threshold):
        """
        Crop the input image by setting the left and right columns to zero (black).

        Args:
            image (numpy.ndarray): The input image to be cropped.
            threshold (int): The number of columns to be set to zero on each side.

        Returns:
            numpy.ndarray: The cropped image.
        """
        image[:, :threshold] = 0  # Left
        image[:, -threshold:] = 0  # Right
        return image

    def query_near(self):
        """
        Returns the current status of being at the waypoint.
        """
        return jsonify({"at_waypoint": self.at_waypoint})

    def aggregate_and_filter_points(self):
        """
        Aggregates points from both local and global graph points and filters them using the L3 norm,
        utilizing NumPy for efficient computation. Returns a list of points with modified sequence ID
        and the most recent timestamp applied to all points.

        Args:
            l3_norm_threshold (float): Threshold for the L3 norm. Points below this value are kept.

        Returns:
            List[Dict]: A list of dictionaries for each point below the L3 norm threshold,
                        with modified sequence ID and a uniform, most recent timestamp.
        """
        all_points, timestamps, seq_ids = self.aggregate_points()

        if all_points == []:
            rospy.loginfo(
                "[llm_robot_client] No points to aggregate and filter"
            )
            return

        clustered_points = self.format_points(all_points, seq_ids, timestamps)
        rospy.loginfo("[llm_robot_client] Points being sent. Numer: %d", len(clustered_points))
        self.filtered_graph_points = clustered_points
        if self.visualize_graph_points:
            self.visualize_filtered_graph_points()

    def aggregate_points(self):
        """
        Aggregates the local and global graph points into a single list.

        Returns:
            all_points (list): A list of all aggregated points, where each point is represented as [x, y, z].
            timestamps (list): A list of timestamps corresponding to each point.
            seq_ids (list): A list of sequence IDs corresponding to each point, where 'L' denotes local points and 'G' denotes global points.
        """
        all_points = []
        timestamps = []
        seq_ids = []
        rospy.loginfo("[llm_robot_client] Aggregating points")

        # Process local points
        '''
        rospy.loginfo(
            f"[llm_robot_client] Local graph point length : {len(self.current_local_graph_points)}"
        )
        '''
        
        if (
            self.current_local_graph_points != []
            and self.current_local_graph_points is not None
        ):
            local_timestamp = self.current_local_graph_points[
                next(iter(self.current_local_graph_points))
            ]["time"]
            for index, point in self.current_local_graph_points.items():
                all_points.append(
                    [
                        point["position"]["x"],
                        point["position"]["y"],
                        point["position"]["z"],
                    ]
                )
                timestamps.append(local_timestamp)
                #seq_ids.append(f"L{point['seq']}")  # Prefix sequence ID with 'L'
                seq_ids.append('L'+str(point['seq'])) 
        '''
        rospy.loginfo(
            f"[llm_robot_client] Global graph points length: {len(self.current_global_graph_points)}"
        )
        '''
        # Process global points
        if (
            self.current_global_graph_points != []
            and self.current_global_graph_points is not None
        ):
            global_timestamp = self.current_global_graph_points[
                next(iter(self.current_global_graph_points))
            ]["time"]
            for index, point in self.current_global_graph_points.items():
                all_points.append(
                    [
                        point["position"]["x"],
                        point["position"]["y"],
                        point["position"]["z"],
                    ]
                )
                timestamps.append(global_timestamp)
                #seq_ids.append(f"G{point['seq']}")  # Prefix sequence ID with 'G'
                seq_ids.append('G'+str(point['seq']))  

        return all_points, timestamps, seq_ids
    
    def format_points(self, all_points, seq_ids, timestamps):
        most_recent_timestamp = max(timestamps)
        formatted_points = []
        for index, point in enumerate(all_points):
            point = {
                "seq": seq_ids[index],
                "time": most_recent_timestamp,
                "position": {
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                },
            }
            formatted_points.append(point)
        return formatted_points

    def filter_points(self, all_points, seq_ids, timestamps):
        print("LENGTH OF ALL POINTS:")
        print(len(all_points))
        # Convert all_points to NumPy array for clustering
        points_array = np.array(all_points)
        # roslaunch llm_robot_client llm_robot.launch object_detection_server_url:=http://0.0.0.0:5005 host_url:=http://0.0.0.0:5000 eps_dbscan:=5.0 min_samples:=5 median_filter_rate:=2 distance_threshold:=0.5 waypoint_threshold:=0.35 enable_bagging:=True
        # Perform DBSCAN clustering
        print("Clustering with values eps: ", self.eps_dbscan, " min_samples: ", self.min_samples)
        dbscan = DBSCAN(
            eps=self.eps_dbscan, min_samples=self.min_samples
        )  # Adjust parameters as necessary
        clusters = dbscan.fit_predict(points_array)
        print("Cluster Length: ", len(clusters))

        # Determine the most recent timestamp
        most_recent_timestamp = max(timestamps)

        # Compile clustered points
        clustered_points = []
        print("Cluster Set", set(clusters))
        for cluster_id in set(clusters):
            print("Cluster ID: ", cluster_id)
            if cluster_id != -1:  # Ignore noise points
                # Find points in the current cluster
                cluster_points = points_array[clusters == cluster_id]

                # Compute centroid of the cluster
                centroid = np.mean(cluster_points, axis=0)

                # Find the closest original point to the centroid as the representative
                closest_point_index = np.argmin(
                    np.linalg.norm(cluster_points - centroid, axis=1)
                )
                original_index = np.where(clusters == cluster_id)[0][
                    closest_point_index
                ]

                point = {
                    "seq": seq_ids[original_index],
                    "time": most_recent_timestamp,
                    "position": {
                        "x": cluster_points[closest_point_index, 0],
                        "y": cluster_points[closest_point_index, 1],
                        "z": cluster_points[closest_point_index, 2],
                    },
                }
                clustered_points.append(point)

        return clustered_points

    def visualize_filtered_graph_points(self, graph_lifetime=0):
        """
        Creates and returns a Marker message to visualize points in RViz.

        Args:
            points (List[Dict]): List of points with 'x', 'y', and 'z' keys.
            world_frame_id (str): The frame ID in which the points are visualized.
            graph_lifetime (int): Duration (in seconds) for which the points should be displayed.

        Returns:
            visualization_msgs/Marker: The configured Marker message for RViz visualization.
        """
        # Create a Marker message
        vertex_marker = Marker()
        vertex_marker.header = Header(stamp=rospy.Time.now(), frame_id="world")
        vertex_marker.ns = "vertices"
        vertex_marker.action = Marker.ADD
        vertex_marker.type = Marker.SPHERE_LIST
        vertex_marker.scale.x = 0.3
        vertex_marker.scale.y = 0.3
        vertex_marker.scale.z = 0.3

        vertex_marker.color = ColorRGBA(1.0, 0.5, 0.0, 1.0)  # RGBA for orange

        vertex_marker.lifetime = rospy.Duration(graph_lifetime)

        # Add points to the Marker message
        for point in self.filtered_graph_points:
            p = Point(
                x=point["position"]["x"],
                y=point["position"]["y"],
                z=point["position"]["z"],
            )
            vertex_marker.points.append(p)

        self.filtered_graph_points_viz_pub.publish(vertex_marker)

    # Method to receive and process waypoints
    def receive_waypoint(self):
        try:
            waypoint = request.get_json()
            #rospy.loginfo(f"[llm_robot_client] Received waypoint: {waypoint}")
            # Update the waypoint attribute
            self.waypoint = waypoint
            return jsonify({"message": "Waypoint received successfully"}), 200
        except Exception as e:
            #print(f"Error processing waypoint: {e}")
            return jsonify({"error": str(e)}), 500

    def odom_callback(self, msg):
        # rospy.loginfo("[llm_robot_client] Received odom")
        with self.odom_lock:
            if self.current_odom_msg is not None:
                current_position = self.current_odom_msg.pose.pose.position
                current_orientation = self.current_odom_msg.pose.pose.orientation
                new_position = msg.pose.pose.position
                new_orientation = msg.pose.pose.orientation
                if (
                    new_position.x != current_position.x
                    or new_position.y != current_position.y
                    or new_position.z != current_position.z
                    or new_orientation.x != current_orientation.x
                    or new_orientation.y != current_orientation.y
                    or new_orientation.z != current_orientation.z
                    or new_orientation.w != current_orientation.w
                ):
                    # Update last movement time if position has changed
                    self.last_movement_time = msg.header.stamp.to_sec()

            timestamp_in_seconds = msg.header.stamp.to_sec()
            self.current_odom = {
                "time": timestamp_in_seconds,
                "position": {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "z": msg.pose.pose.position.z,
                },
                "orientation": {
                    "x": msg.pose.pose.orientation.x,
                    "y": msg.pose.pose.orientation.y,
                    "z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w,
                },
            }
            self.current_odom_msg = copy.deepcopy(msg)

    def path_callback(self, msg):
        """
        Callback function for processing path messages.

        Args:
            msg (Path): The path message containing a list of poses.

        Returns:
            None
        """
        # rospy.loginfo("[llm_robot_client] Received path")
        with self.path_lock:
            self.current_path = {}
            for index, pose_stamped in enumerate(msg.poses):
                # Extracting the timestamp from each PoseStamped in the path
                timestamp_in_seconds = pose_stamped.header.stamp.to_sec()

                self.current_path[index] = {
                    "seq": pose_stamped.header.seq,  # Add the header sequence ID
                    "time": timestamp_in_seconds,  # Add the converted timestamp
                    "position": {
                        "x": pose_stamped.pose.position.x,
                        "y": pose_stamped.pose.position.y,
                        "z": pose_stamped.pose.position.z,
                    },
                    "orientation": {
                        "x": pose_stamped.pose.orientation.x,
                        "y": pose_stamped.pose.orientation.y,
                        "z": pose_stamped.pose.orientation.z,
                        "w": pose_stamped.pose.orientation.w,
                    },
                }

    def local_graph_points_callback(self, msg):
        """
        Callback function for handling frontier points messages.

        Args:
            msg (FrontierPoints): The message containing frontier points.

        Returns:
            None
        """
        with self.local_graph_points_lock:
            rospy.loginfo("[llm_robot_client] Received local graph points: %d", len(msg.poses))
            self.current_local_graph_points = {}
            for index, pose_stamped in enumerate(msg.poses):
                # Extracting the timestamp from each PoseStamped in the path
                timestamp_in_seconds = msg.header.stamp.to_sec()

                self.current_local_graph_points[index] = {
                    "seq": msg.header.seq,  # Add the header sequence ID
                    "time": timestamp_in_seconds,  # Add the converted timestamp
                    "position": {
                        "x": pose_stamped.position.x,
                        "y": pose_stamped.position.y,
                        "z": pose_stamped.position.z,
                    },
                }

    def global_graph_points_callback(self, msg):
        """
        Callback function for handling global points messages.

        Args:
            msg (FrontierPoints): The message containing global graph points.

        Returns:
            None
        """
        with self.global_graph_points_lock:
            rospy.loginfo("[llm_robot_client] Received global graph points: %d", len(msg.poses))
            self.current_global_graph_points = {}
            for index, pose_stamped in enumerate(msg.poses):
                # Extracting the timestamp from each PoseStamped in the path
                timestamp_in_seconds = msg.header.stamp.to_sec()

                self.current_global_graph_points[index] = {
                    "seq": msg.header.seq,  # Add the header sequence ID
                    "time": timestamp_in_seconds,  # Add the converted timestamp
                    "position": {
                        "x": pose_stamped.position.x,
                        "y": pose_stamped.position.y,
                        "z": pose_stamped.position.z,
                    },
                }

    def frontier_points_callback(self, msg):
        """
        Callback function for handling global points messages.

        Args:
            msg (FrontierPoints): The message containing global graph points.

        Returns:
            None
        """
        with self.frontier_points_lock:
            rospy.loginfo("[llm_robot_client] Received frontier points: %d", len(msg.poses))
            self.current_frontier_points = {}
            for index, pose_stamped in enumerate(msg.poses):
                # Extracting the timestamp from each PoseStamped in the path
                timestamp_in_seconds = msg.header.stamp.to_sec()

                self.current_frontier_points[index] = {
                    "seq": msg.header.seq,  # Add the header sequence ID
                    "time": timestamp_in_seconds,  # Add the converted timestamp
                    "position": {
                        "x": pose_stamped.position.x,
                        "y": pose_stamped.position.y,
                        "z": pose_stamped.position.z,
                    },
                }

    def all_objects_callback(self, msg):
        """
        Callback function for handling projected objects messages.

        Args:
            msg: The message containing projected objects information.

        Returns:
            None
        """
        with self.all_objects_lock:
            rospy.logdebug("[llm_robot_client] Received All Objects objects")
            self.all_objects = {}
            for index, detection in enumerate(msg.detections):
                timestamp_in_seconds = detection.header.stamp.to_sec()
                self.all_objects[index] = {
                    "name": detection.name,
                    "seq": detection.header.seq,
                    "time": timestamp_in_seconds,
                    "confidence": detection.confidence,
                    "position": {
                        "x": detection.position.x,
                        "y": detection.position.y,
                        "z": detection.position.z,
                    },
                }

    def current_object_detecions_callback(self, msg):
        """
        Callback function for handling projected objects messages.

        Args:
            msg: The message containing projected objects information.

        Returns:
            None
        """
        with self.current_objects_lock:
            self.current_object_detections = {}
            for index, detection in enumerate(msg.detections):
                timestamp_in_seconds = detection.header.stamp.to_sec()
                self.current_object_detections[index] = {
                    "name": detection.name,
                    "seq": detection.header.seq,
                    "time": timestamp_in_seconds,
                    "confidence": detection.confidence,
                    "position": {
                        "x": detection.position.x,
                        "y": detection.position.y,
                        "z": detection.position.z,
                    },
                }

    def object_detector_status_callback(self, msg):
        """
        Determines if the Object server is running and has detected labels.

        Args:
            msg: The message containing the status data.

        Returns:
            None
        """
        with self.object_status_lock:
            self.object_detector_status = msg.data

    def is_ready_callback(self, msg):
        """
        Determines if the robot has a map and is ready to go.

        Args:
            msg: A boolean value indicating the robot's readiness.

        Returns:
            None
        """
        with self.is_ready_lock:
            self.is_ready = msg.data
            if self.is_ready == True:
                self.planner_controller.start_planner()
                if self.enable_naive_exploration == True:
                    self.planner_controller.start_motion()
                    self.initialize_object_detection_server()


    def end_goal_callback(self, msg):
        """
        Callback function for the end goal message from Unreal.

        Args:
            msg: The end goal message.

        Returns:
            None
        """
        with self.end_goal_lock:
            self.reached_end_goal = msg.data

    def planning_mode_callback(self, msg):
        """
        Callback function for the planning status message from the planner.

        Args:
            msg: The planning status message.

        Returns:
            None
        """
        with self.planner_mode_lock:
            self.planner_mode = msg.data

    def waypoint_plan_status_callback(self, msg):
        """
        Callback function for the can plan message from the planner.

        Args:
            msg: The can plan message.

        Returns:
            None
        """
        rospy.loginfo("[llm_robot_client] Received waypoint plan status")
        rospy.loginfo("[llm_robot_client] Waypoint Plan Status: %s", msg)
        self.waypoint_plan_status = msg
        

    def check_waypoint_status(self):
        """
        Checks if the robot is at the current waypoint.

        Returns:
            None
        """
        rospy.loginfo_throttle(1, "[llm_robot_client] Checking waypoint status")
        if self.current_waypoint is None:
            return
        if self.current_odom_msg is None:
            return
        if self.waypoint_plan_status is None:
            return

        current_position = self.current_odom_msg.pose.pose.position
        current_waypoint = self.current_waypoint.pose.position
        current_end_of_path = self.waypoint_plan_status.path_end_pose.position
        # Current waypoint from planner. Make sure it matches what explore thinks is the waypont
        current_planned_waypoint = self.waypoint_plan_status.waypoint.pose.position
        rospy.loginfo_throttle(1,
            "[llm_robot_client] current_position: %s", current_position
        )
        rospy.loginfo_throttle(1,
            "[llm_robot_client] current_waypoint: %s", current_waypoint
        )
        rospy.loginfo_throttle(1,
            "[llm_robot_client] current_planned_waypoint: %s",
            current_planned_waypoint,
        )
        rospy.loginfo_throttle(1,
            "[llm_robot_client] current_end_of_path: %s", current_end_of_path
        )
        distance_to_waypoint = math.sqrt(
            (current_position.x - current_end_of_path.x) ** 2
            + (current_position.y - current_end_of_path.y) ** 2
            + (current_position.z - current_end_of_path.z) ** 2
        )

        rospy.loginfo_throttle(1,
            "[llm_robot_client] distance_to_waypoint: %s", distance_to_waypoint
        )

        rospy.loginfo_throttle(1,
            "[llm_robot_client] current_waypoint: %s", current_waypoint
        )
        rospy.loginfo_throttle(1,
            "[llm_robot_client] current_planned_waypoint: %s",
            current_planned_waypoint,
        )
        
        waypoint_ready = self.check_waypoint_ready()

        if not waypoint_ready:
            rospy.loginfo_throttle(1, "[llm_robot_client] Waypoint not ready")
            self.waypoint_status = {
                "Distance": float("inf"),
                "Arrived": False,
                "Ready": False,
                "Planned": False,
            }

        elif waypoint_ready and not self.waypoint_plan_status.success:
            rospy.loginfo_throttle(1,
                "[llm_robot_client] Planner was unable to find planning point close to the waypoint"
            )
            self.waypoint_status = {
                "Distance": float("inf"),
                "Arrived": False,
                "Ready": True,
                "Planned": False,
            }

        else:
            if distance_to_waypoint < self.waypoint_threshold:
                self.waypoint_status = {
                    "Distance": distance_to_waypoint,
                    "Arrived": True,
                    "Ready": True,
                    "Planned": True,
                }
            else:
                self.waypoint_status = {
                    "Distance": distance_to_waypoint,
                    "Arrived": False,
                    "Ready": True,
                    "Planned": True,
                }

    def check_planned_waypoint(self):
        """
        Checks if the returned path from the planner matches the desired waypoint

        Returns:
            bool: True if the current waypoint can be planned, False otherwise.
        """
        tolerance = self.waypoint_threshold
        current_requested_waypoint = self.waypoint_plan_status.waypoint.pose.position
        current_planned_waypoint = self.waypoint_plan_status.path_end_pose.position
        # Calculate the difference in x and y positions
        x_diff = abs(current_requested_waypoint.x - current_planned_waypoint.x)
        y_diff = abs(current_requested_waypoint.y - current_planned_waypoint.y)

        rospy.loginfo("X diff: %f", x_diff)
        rospy.loginfo("Y diff: %f", y_diff)
        rospy.loginfo("Tolerance: %f", tolerance)

        # Check if the differences are within the tolerance
        if x_diff <= tolerance and y_diff <= tolerance:
            rospy.loginfo("Waypoint matches planned waypoint")
            return True
        else:
            rospy.loginfo("Waypoint does not match planned waypoint")
            return False

    def check_waypoint_ready(self):
        """
        Checks if the plan for the sent waypoint has arrived

        Returns:
            bool: True if the sent waypoint matches the planned waypoint in the status message, False otherwise.
        """

        x_diff = abs(
            self.current_waypoint.pose.position.x
            - self.waypoint_plan_status.waypoint.pose.position.x
        )
        y_diff = abs(
            self.current_waypoint.pose.position.y
            - self.waypoint_plan_status.waypoint.pose.position.y
        )
        # Tolernace to account for rounding errors
        if x_diff <= 0.1 and y_diff <= 0.1:
            return True
        else:
            return False

    def receive_object_labels(self):
        rospy.loginfo("[llm_robot_client] Received Object labels")
        try:
            # Extract GLIP label data from the request
            object_labels = request.get_json()
            rospy.loginfo("[llm_robot_client] Received Object labels %s", object_labels)
            if self.use_glip:
                self.object_labels = " . ".join(object_labels) + " ."
            else:
                self.object_labels = ",".join(object_labels)
            self.object_list_pub.publish(self.object_labels)
            rospy.loginfo("Published Object labels: %s", self.object_labels)
            # Return a success response
            return jsonify({"message": "Object labels received successfully"}), 200
        except Exception as e:
            # Handle any exceptions
            #print(f"Error processing Object labels: {e}")
            print("Error processing object labels: ",str(e))
            return jsonify({"error": str(e)}), 500

    def set_waypoint_threshold(self):
        """
        Sets the waypoint threshold for the robot.

        This method receives a JSON object containing the threshold value.
        It sets the waypoint threshold for the robot.

        Returns:
            A JSON response indicating the success or failure of setting the waypoint threshold.
        """
        try:
            # Extract the threshold value from the request
            threshold = request.get_json()
            self.waypoint_threshold = float(threshold["threshold"])
            return jsonify({"message": "Waypoint threshold set successfully"}), 200
        except Exception as e:
            #print(f"Error processing waypoint threshold: {e}")
            return jsonify({"error": str(e)}), 500

    def send_image(self):
        if len(self.image_queue) > 0:
            image = copy.deepcopy(self.image_queue.get())
            if image is not None:
                _, buffer = cv2.imencode(".jpg", image["image"])
            return buffer.tobytes(), 200, {"Content-Type": "image/jpeg"}
        return "Images not available", 404

    def send_odom(self):
        with self.odom_lock:
            if self.current_odom is not None:
                return jsonify(self.current_odom)
        return "Odom not available", 404

    def send_path(self):
        with self.path_lock:
            if self.current_path is not None:
                return jsonify(self.current_path)
        return "Path not available", 404

    def send_graph_points(self):
        with self.global_graph_points_lock:
            with self.local_graph_points_lock:
                self.aggregate_and_filter_points()
                if self.filtered_graph_points != []:
                    rospy.loginfo("[llm_robot_client] Sending filtered graph points")
                    return jsonify(self.filtered_graph_points)
        rospy.loginfo("[llm_robot_client] Filtered graph points not available")
        return "Graph points not available", 404

    def send_frontier_points(self):
        with self.frontier_points_lock:
            if self.current_frontier_points != []:
                rospy.loginfo("[llm_robot_client] Sending frontier points")
                return jsonify(self.current_frontier_points)
            rospy.loginfo("[llm_robot_client] Frontier points not available")
        return "Frontier points not available", 404

    def send_current_object_detections(self):
        with self.current_objects_lock:
            if self.current_object_detections is not None:
                return jsonify(self.current_object_detections)
        return "Projection artifacts not available", 404

    def send_all_object_detections(self):
        with self.all_objects_lock:
            if self.all_objects is not None:
                return jsonify(self.all_objects)
            return "All objects not available", 404

    def send_object_detection_status(self):
        with self.object_status_lock:
            if self.object_detector_status is not None:
                return jsonify({"server_status": self.object_detector_status})
            return "Object detector status not available", 404

    def send_waypoint_status(self):
        with self.waypoint_status_lock:
            if self.waypoint_status is not None:
                # Removed check since explore checks in in wait for waypoint
                # if self.check_waypoint_ready():
                return jsonify(self.waypoint_status)
        return "Waypoint status not available", 404

    def send_last_movement_time(self):
        if self.last_movement_time is not None:
            current_time = rospy.get_time()  # Get current time in seconds
            # Only update last movement time if we are actively going to a waypoint
            if self.enable_movement_time == True:
                time_since_last_movement = current_time - self.last_movement_time
            else:
                time_since_last_movement = -1.0
            return jsonify({"Last Movement Time": time_since_last_movement})
        return "Last movement time not available", 404

    def send_reached_end_goal(self):
        if self.reached_end_goal is not None:
            return jsonify({"End Goal Status": self.reached_end_goal})
        return "Reached end goal not available", 404

    def send_can_plan_status(self):
        if self.waypoint_plan_status is not None:
            if self.check_waypoint_ready():
                '''
                rospy.loginfo(
                    f"[llm_robot_client] Sending can plan status: {self.waypoint_plan_status.success}"
                )
                '''
                return jsonify({"Can Plan Status": self.waypoint_plan_status.success})
        return "Can plan status not available", 404

    def send_planner_mode(self):
        if self.planner_mode is not None:
            return jsonify({"Planner Mode": self.planner_mode})
        return "Planner status not available", 404

    def send_is_ready(self):
        return jsonify({"Ready": self.is_ready}), 200

    def set_waypoint(self):
        """
        Sets the waypoint for the robot.

        This method receives a JSON object containing the x, y, and z coordinates of the waypoint.
        It creates a PoseStamped message with the received coordinates and publishes it to the waypoint_pub topic.
        Additionally, it publishes a "guiCMD" message to the task_pub to notify the planner to follow a waypoint.

        Returns:
            A JSON response indicating the success or failure of setting the waypoint.
        """
        rospy.loginfo("Setting Waypoint")
        try:
            # Extract waypoint data from the request
            waypoint = request.get_json()
            waypoint_msg = PoseStamped()
            waypoint_msg.header = Header()
            waypoint_msg.header.stamp = rospy.Time.now()
            waypoint_msg.header.frame_id = "world"
            waypoint_msg.pose.position.x = waypoint["x"]
            waypoint_msg.pose.position.y = waypoint["y"]
            waypoint_msg.pose.position.z = waypoint["z"]

            # Initialize the quaternion
            waypoint_msg.pose.orientation.x = 0.0
            waypoint_msg.pose.orientation.y = 0.0
            waypoint_msg.pose.orientation.z = 0.0
            waypoint_msg.pose.orientation.w = 1.0
            #Stop planner
            self.planner_controller.stop_planner()
            self.planner_controller.plan_to_waypoint_with_pose(waypoint_msg)
            #self.planner_controller.search()
            self.planner_controller.start_planner()
            # Flag for arrival time
            self.enable_movement_time = True
            self.current_waypoint = copy.deepcopy(waypoint_msg)

            waypoint_marker = Marker()
            waypoint_marker.header = waypoint_msg.header
            waypoint_marker.type = Marker.SPHERE  # Or any other shape
            waypoint_marker.action = Marker.ADD
            waypoint_marker.pose.position = waypoint_msg.pose.position
            waypoint_marker.scale.x = 0.2  # Specify the size of the marker
            waypoint_marker.scale.y = 0.2
            waypoint_marker.scale.z = 0.2
            waypoint_marker.color.a = 1.0  # Don't forget to set the alpha!
            waypoint_marker.color.r = 1.0  # Red color
            waypoint_marker.color.g = 0.0
            waypoint_marker.color.b = 0.0
            self.waypoint_marker_pub.publish(waypoint_marker)

            # If we were previously interrupted for an object explore has now sent a new waypoint
            self.object_interrupt = False
            rospy.loginfo("Published Waypoint: %s", waypoint_msg)

            return jsonify({"message": "Waypoint received successfully"}), 200
        except Exception as e:
            #rospy.logerr(f"Error processing waypoint: {e}")
            print("Error processing waypoint: ",str(e))
            return jsonify({"error": str(e)}), 500
        
        
    def set_target_object_waypoint(self, target_object):
        """
        Sets the waypoint for the robot.

        This method receives a JSON object containing the x, y, and z coordinates of the waypoint.
        It creates a PoseStamped message with the received coordinates and publishes it to the waypoint_pub topic.
        Additionally, it publishes a "guiCMD" message to the task_pub to notify the planner to follow a waypoint.

        Returns:
            A JSON response indicating the success or failure of setting the waypoint.
        """
        rospy.loginfo("Setting Target Object Waypoint")
        
        # Create Waypoint Message
        waypoint_msg = PoseStamped()
        waypoint_msg.header = Header()
        waypoint_msg.header.stamp = rospy.Time.now()
        waypoint_msg.header.frame_id = "world"
        waypoint_msg.pose.position.x = target_object["x"]
        waypoint_msg.pose.position.y = target_object["y"]
        waypoint_msg.pose.position.z = target_object["z"]
        
        rospy.loginfo("Publishing target object at: %s", waypoint_msg)

        # Initialize the quaternion
        waypoint_msg.pose.orientation.x = 0.0
        waypoint_msg.pose.orientation.y = 0.0
        waypoint_msg.pose.orientation.z = 0.0
        waypoint_msg.pose.orientation.w = 1.0
        #Stop planner
        self.planner_controller.stop_planner()
        self.planner_controller.plan_to_waypoint_with_pose(waypoint_msg)
        #self.planner_controller.search()
        self.planner_controller.start_planner()
        # Flag for arrival time
        self.enable_movement_time = True
        self.current_waypoint = copy.deepcopy(waypoint_msg)

        waypoint_marker = Marker()
        waypoint_marker.header = waypoint_msg.header
        waypoint_marker.type = Marker.SPHERE  # Or any other shape
        waypoint_marker.action = Marker.ADD
        waypoint_marker.pose.position = waypoint_msg.pose.position
        waypoint_marker.scale.x = 0.2  # Specify the size of the marker
        waypoint_marker.scale.y = 0.2
        waypoint_marker.scale.z = 0.2
        waypoint_marker.color.a = 1.0  # Don't forget to set the alpha!
        waypoint_marker.color.r = 1.0  # Red color
        waypoint_marker.color.g = 0.0
        waypoint_marker.color.b = 0.0
        self.waypoint_marker_pub.publish(waypoint_marker)
        rospy.loginfo("Published Waypoint: %s", waypoint_msg)
        
        
    def set_object_interrupt(self):
        self.object_interrupt = True
        self.stop_robot()
        return jsonify({"message": "Object interrupt set successfully"}), 200

    def start_robot(self):
        """
        Starts the robot.

        This method publishes a command to start the robot and logs an info message.

        Returns:
            A JSON response with a success message and HTTP status code 200.
        """
        self.planner_controller.start_motion()
        rospy.loginfo("Started Robot")
        return jsonify({"message": "Robot Start Command Sent"}), 200

    def stop_robot(self):
        """
        Stops the robot by publishing a stop command and logs the action.

        Returns:
            A JSON response with a success message and HTTP status code 200.
        """
        self.planner_controller.stop_motion()
        rospy.loginfo("Stopped Robot")
        return jsonify({"message": "Robot Stop Command Sent"}), 200

    def start_planner(self):
        """
        Starts the robot.

        This method publishes a command to start the robot planner (in automatic mode)

        Returns:
            A JSON response with a success message and HTTP status code 200.
        """
        self.planner_controller.start_planner()
        rospy.loginfo("Started Planner")
        return jsonify({"message": "Start Planner Command Sent Sent"}), 200

    def stop_planner(self):
        """
        Stops the robot by publishing a stop command and logs the action.

        Returns:
            A JSON response with a success message and HTTP status code 200.
        """
        self.planner_controller.stop_planner()
        rospy.loginfo("Stopped Planner")
        return jsonify({"message": "Planner Stop Command Sent"}), 200
    
    def trigger_planner_graph_update(self):
        """
        Triggers an update of the planner graph.

        This method starts the planner, waits for the current local and global graph points to be updated,
        and then stops the planner. It returns a JSON response indicating that the graph update was triggered.

        :return: A JSON response indicating that the graph update was triggered.
        :rtype: dict
        """
        self.current_local_graph_points = None
        self.current_global_graph_points = None
        self.planner_controller.start_planner()
        while self.current_local_graph_points is None or self.current_global_graph_points is None:
            rospy.sleep(0.1)
            rospy.loginfo_throttle(1, "Waiting for graph points")
        self.planner_controller.stop_planner()
        return jsonify({"message": "Graph update triggered"}), 200
    
    def toggle_cameras(self):
        """
        Toggles the camera feed.

        This method toggles the camera feed by publishing a command to the camera feed topic.

        Returns:
            A JSON response with a success message and HTTP status code 200.
        """
        toggle_cam_msg = Bool()
        toggle_cam_msg.data = True  
        self.camera_toggle_pub.publish(toggle_cam_msg)
        return jsonify({"message": "Camera Toggle Command Sent"}), 200
    
    def set_target_object(self):
        """
        Sets the target object name based on the JSON object received in the request.

        Parameters:
            None

        Returns:
            None
        """
        try:
            # Get JSON data from request
            data = request.get_json()

            # Ensure the data contains a list of objects
            if "target_objects" not in data:
                return jsonify({"success": False, "message": "Missing target_objects key"}), 400

            # Update the target_object_names list
            self.target_object_names = data["target_objects"]

            # Provide feedback to client
            return jsonify({"success": True, "message": "Target objects updated", "target_objects": self.target_object_names}), 200
        except Exception as e:
            # Handle any errors
            return jsonify({"success": False, "message": str(e)}), 500
