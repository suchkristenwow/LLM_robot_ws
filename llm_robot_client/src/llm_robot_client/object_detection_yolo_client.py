#!/usr/bin/env python2

import rospy
import requests
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import Header, String, Bool
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import PointCloud2 
from llm_robot_client.msg import ObjectDetection, ObjectDetectionArray
import copy
from flask import Flask, request, jsonify
import cv2
import numpy as np
import threading
from collections import deque
import message_filters
import re   

from OOD_utils import * 
from llm_utils import generate_with_openai  
from robot_transforms import robotTransforms 
import subprocess 
import json 
import base64 

"""
class CircularBuffer(deque):
    def __init__(self, size=0):
        super().__init__(maxlen=size)

    def put(self, item):
        self.append(item)
"""

def process_image(image_data):
    #rospy.loginfo("Processing image... this is the image type: %s",type(image_data))
    # Check if image data is a byte array and encode to base64 string
    """
    if isinstance(image_data, (bytes, bytearray)):
        return base64.b64encode(image_data).decode('utf-8')
    return image_data
    """
    _, buffer = cv2.imencode('.jpg', image_data)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image 

def clean_detection_for_json(detection):
    # Convert unsupported types into serializable ones
    if detection.contour_pts == []:
        detection.contour_pts = None  # or an empty list if needed

    if not detection.image.data:
        detection.image.data = None

    # Ensure format is a string (in case it's empty or not set)
    if not detection.image.format:
        detection.image.format = ""

    return detection

class CircularBuffer(deque):
    def __init__(self, size=0):
        # In Python 2, you have to use the old-style super() or call the parent class directly
        deque.__init__(self, maxlen=size)

    def put(self, item):
        self.append(item)

class ObjectDetectionClient:
    def __init__(self):
        # Initialize ROS Node
        odometry_topic = rospy.get_param("~robot_pose_topic") 
        self.current_odom_msg = None 
        ground_plane_topic = rospy.get_param("~ground_plane_topic")
        self.current_ground_plane_msg = None 
        config_path = rospy.get_param("~config_path","/home/marble/LLMGuidedSeeding/configs/example_config.toml")
        self.robot_tfs = robotTransforms(config_path) 
        cam_front_topic = rospy.get_param("~cam_front_topic", "image")
        cam_left_topic = rospy.get_param("~cam_left_topic", "image")
        cam_right_topic = rospy.get_param("~cam_right_topic", "image")
        rospy.loginfo("cam_front_topic: %s", cam_front_topic)
        rospy.loginfo("cam_left_topic: %s", cam_left_topic)
        rospy.loginfo("cam_right_topic: %s", cam_right_topic)
        # object_list_topic = rospy.get_param("~object_list_topic", "object_list")
        self.crop_threshold = rospy.get_param("~crop_threshold", 15)
        self.publish_concate_image_bool = rospy.get_param(
            "~publish_concate_image", True
        )
        self.publish_detections_image_bool = rospy.get_param(
            "~publish_detections_image", True
        )
        self.object_detection_server_url = rospy.get_param(
            "~object_detection_server_url", "http://0.0.0.0.edu:5000"
        )
        self.segment_anything_server_url = rospy.get_param(
            "~segment_anything_server_url", "http://0.0.0.0.edu:5000"
        )
        
        self.object_detector_status_topic = rospy.get_param(
            "~object_detector_status_topic", "object_detector/server_status"
        )

        self.image_front_sub = message_filters.Subscriber(
            cam_front_topic, CompressedImage
        )
        self.image_left_sub = message_filters.Subscriber(
            cam_left_topic, CompressedImage
        )
        self.image_right_sub = message_filters.Subscriber(
            cam_right_topic, CompressedImage
        )

        self.odom_sub = rospy.Subscriber(
            odometry_topic,Odometry,self.odom_callback 
        )

        self.ground_plane_sub = rospy.Subscriber(
            ground_plane_topic,PointCloud2,self.ground_plane_callback 
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_front_sub, self.image_left_sub, self.image_right_sub],
            queue_size=10,
            slop=0.5,
        )
        self.ts.registerCallback(self.image_callback)
        self.image_queue = CircularBuffer(size=2)

        # Object List Callback
        """
        self.object_list_subscriber = rospy.Subscriber(
            object_list_topic, String, self.receive_object_labels
        )
        """
        
        self.concate_image_publisher = rospy.Publisher(
            "object_detector/concate/image/compressed",
            CompressedImage,
            latch=True,
            queue_size=10,
        )
        
        object_detection_topic = rospy.get_param("~object_detection_2D_topic", "object_detection")

        self.detected_image_publisher = rospy.Publisher(
            "object_detector/detection/image/compressed",
            CompressedImage,
            latch=True,
            queue_size=10,
        )
        
        rospy.loginfo("object_detection_topic: %s", object_detection_topic)

        self.detections_publisher = rospy.Publisher(
            object_detection_topic,
            ObjectDetectionArray,
            queue_size=10,
            latch=True,
        )
        
        object_detector_status_topic = rospy.get_param("~object_detector_status_topic", "object_detector/server_status")
        
        self.server_status_publisher = rospy.Publisher(
            object_detector_status_topic,
            Bool,
            queue_size=10,
            latch=True,
        )

        # CV Bridge
        self.bridge = CvBridge()

        #TO DO: Load in Objects 
        self.object_list = ['fire extinguisher']

        #TO DO: Load in OOD Objects 
        self.ood_objects = {}
        with open("/home/marble/LLMGuidedSeeding/prompts/custom_objects/tape.txt",'r') as f:
            self.ood_objects['tape'] = f.read() 

        self.cam_frame_ids = {'right': None, 'front': None, 'left': None}
        
        self.frame_width = None
        
        self.init_detection_thread()

        rospy.loginfo("[object_detection_client] loaded and ready")
        
        self.server_running = False
        
        request = {"action": "initialize"}
        rospy.loginfo("initializing segment anything server ...") 
        response = requests.post(
                    self.segment_anything_server_url + "/initialize",
                    json=request, 
                    timeout=10
                ) 

        if response.status_code != 200:
            rospy.logwarn("Segment Anything Server initialization failed with response: %s",response.text)

        rospy.loginfo("Done initializing Segment Anything server") 


    def annotate_image(self, mask, image, format):
        """
        Draw the image with the mask on top.
        """
        rospy.loginfo("Annotating Image!")
        # Ensure the mask is properly extracted
        if isinstance(mask,dict): 
            m = mask['segmentation']
        else:
            m = mask 
        # Ensure that `m` is a valid mask, typically a boolean or integer array
        if m is not None:
            rospy.loginfo("[annotate_image] m.shape: {}".format(m.shape)) 
            rospy.loginfo("[annotate_image] sum: %d",np.sum(m))
            # Make sure the mask and the image have compatible shapes
            if m.shape != image.shape[:2]:
                raise ValueError("Mask shape {} does not match image shape {}".format(m.shape, image.shape[:2]))
            
            # Create an RGBA color mask (red with 35% transparency)
            color_mask = np.array([255, 0, 0, 0.35])  # RGBA color mask
            
            # Add the color mask to the image where the mask is True
            image[m] = color_mask[:3]  # Apply RGB values to the image
        else:
            raise OSError("m is NONE type")
        
        # Create a CompressedImage message
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.format = format
        msg.data = np.array(cv2.imencode('.jpg', image)[1]).tobytes()
        rospy.loginfo("Returning annotated image! Writing to: {}".format("/home/marble/annotated_img.jpg"))
        cv2.imwrite("/home/marble/annotated_img.jpg",image)
        return msg

    def init_detection_thread(self):
        self.detection_lock = threading.Lock()
        self.detection_thread = threading.Thread(target=self.run_detection_thread)
        self.detection_thread.start()

    def run_detection_thread(self):
        while not rospy.is_shutdown():
            # Acquire the lock
            acquired = self.detection_lock.acquire(False)  # Non-blocking acquisition
            
            try: 
                if acquired:  # Proceed only if the lock was acquired
                    if len(self.image_queue) > 0 and self.object_list is not None:
                        image_data = copy.deepcopy(self.image_queue.pop())
                        self.detect_objects(image_data)
                    else:
                        #rospy.logwarn("Cannot detect objects: %s",self.object_list)
                        if self.object_list is None:
                            rospy.loginfo("Object List is None!")
                        if len(self.image_queue) == 0:
                            rospy.loginfo("Image queue is empty!")
                else:
                    rospy.logwarn("Failed to acquire lock")
                    
            except Exception as e: 
                error_message = "Error in detection thread: {}".format(str(e))
                rospy.logerr(error_message)
            
            finally:
                # Ensure the lock is only released if it was acquired
                if acquired and self.detection_lock.locked():
                    self.detection_lock.release()
            
            rospy.sleep(0.05)
    
    def odom_callback(self,msg):
        self.current_odom_msg = msg 

    def ground_plane_callback(self,msg):
        self.current_ground_plane_msg = msg 

    def image_callback(self, front, left, right):
        #rospy.loginfo("Inside image callback ....") 
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
        
        if cv_image_front is None or cv_image_left is None or cv_image_right is None:
            rospy.logwarn("One or more images are invalid")
            return

        images = {
            'front': cv_image_front,
            'left': cv_image_left,
            'right': cv_image_right
        }
        
        # Check if frame_ids have been gathered
        if self.cam_frame_ids['front'] is None:
            self.cam_frame_ids['front'] = front.header.frame_id
        if self.cam_frame_ids['left'] is None:
            self.cam_frame_ids['left'] = left.header.frame_id
        if self.cam_frame_ids['right'] is None:
            self.cam_frame_ids['right'] = right.header.frame_id
        
        self.image_queue.put({'time': front.header.stamp, 'image': images})

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
        if self.publish_concate_image:
            self.publish_concate_image(single_image)
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
    
    """
    def receive_object_labels(self, msg):
        #rospy.loginfo("Object list received")
        self.object_list = msg.data
    """ 

    def encode_image(self, image):
        """
        Encodes the input image to JPEG format.

        Args:
            image (numpy.ndarray): The input image to be encoded.

        Returns:
            bytes: The encoded image.
        """

        success, encoded_image = cv2.imencode(".jpg", image)

        if not success:
            rospy.loginfo("Error encoding image")
            return None
        return success,encoded_image
    
    def detect_objects(self, image_data):
        image_list = ['front', 'left', 'right']
        rospy.loginfo("detecting objects...")
        for frame in image_list:
            rospy.loginfo("processing frame: %s", frame)
            try:
                success,encoded_image = self.encode_image(image_data['image'][frame])
                # Check if encoding succeeded
                if not success:
                    rospy.logwarn("Failed to encode image for frame: {}".format(frame))
                    continue
                else:
                    rospy.loginfo("Successfully encoded image for frame: {}".format(frame))
                #cv2.imwrite("/home/marble/debug_imgs/ros_image.jpg", image_data['image'][frame])
                # Prepare the headers for the HTTP request
                headers = {"Content-Type": "image/jpeg"}
                rospy.loginfo("Posting request to yolo world server ...") 
                response = requests.post(
                    self.object_detection_server_url + "/process",
                    data=encoded_image.tobytes(),
                    headers=headers,
                    params={"caption": self.object_list},
                )
                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the response
                    #rospy.loginfo("Response JSON: %s", response.json())
                    self.process_detection_results(image_data, response.json(), frame)
                else:
                    rospy.logwarn(
                        "Error in yolo world server response: {}".format(response.status_code)
                    )
            except requests.exceptions.RequestException as e:
                rospy.logwarn("Error querying yolo world server: {}".format(e))
    
    def segment_ground_mask(self,image,frame,masks,max_masks_for_consideration):
        #5. Get ground and non-ground masks 
        ground_plane_z = get_ground_plane_z(self.current_ground_plane_msg) 
        pts_inside_frustrum = get_ground_plane_pts(self.robot_tfs,self.current_odom_msg,self.current_ground_plane_msg,frame) 
        ground_plane_px_coords = inverse_projection(pts_inside_frustrum,frame) 
        grounded_px_coords = sample_ground_pts(ground_plane_z,ground_plane_px_coords)
        encoded_image = process_image(image) 
        request = {
            'action': 'predict_masks',
            'px_coords': grounded_px_coords,  
            'image': encoded_image 
        }
        rospy.loginfo("Calling Segment Anything server ... calling predict masks")
        response_data = requests.post(self.segment_anything_server_url + "/predict_masks",json=request)
        potential_ground_masks = response_data['masks']

        if len(potential_ground_masks) > max_masks_for_consideration:
            potential_ground_masks = pick_largest_k_masks(potential_ground_masks,max_masks_for_consideration) 

        largest_ground_mask = audition_masks('ground',potential_ground_masks,image) 

        ground_masks,non_ground_masks = get_ground_masks(largest_ground_mask,masks) 
        return ground_masks,non_ground_masks 

    def detect_OOD_objects(self, image_data, max_masks_for_consideration=10):
        image_list = ['front', 'left', 'right']
        rospy.loginfo("detecting OOD objects...")
        for frame in image_list:
            encoded_image = process_image(image_data['image'][frame]) 
            request = {
                        'action': 'generate_masks',
                        'image': encoded_image 
                    }
            rospy.loginfo("Calling Segment Anything server ... calling generate masks")
            response_data = requests.post(self.segment_anything_server_url + "/generate_masks",json=request) 
            masks = response_data['masks']
            ground_masks,non_ground_masks = self.segment_ground_mask(image_data['image'][frame],frame,masks,max_masks_for_consideration)
            #cv2.imwrite("tmp_frame.jpg",image_data['image'][frame])
            try:
                for custom_obj in self.ood_objects.keys:
                    #1. Ask ChatGPT if foo is in the frame 
                    prompt = "The user has defined a"+custom_obj+"like this: \n" + self.ood_objects[custom_obj]
                    response,history = generate_with_openai(prompt,image_path="tmp_frame.jpg")
                    #2. If so, are there multiple foo 
                    if not 'yes' in response.lower():
                        continue 
                    prompt = "Are there more than one " + custom_obj + " in this image?" 
                    response,history = generate_with_openai(prompt,conversational_history=history,image_path="tmp_frame.jpg") 
                    multi_obj = False 
                    if 'yes' in response.lower:
                        multi_obj = True 
                    #3. Would foo on the ground in this image? 
                    on_ground = is_ground_obj(custom_obj,self.ood_objects[custom_obj],image_data['image'][frame])
                    # The following steps are to get the obj mask(s) in this frame 
                    #4. Extract all the masks
                    if on_ground: 
                        possible_masks = ground_masks 
                    else:
                        possible_masks = non_ground_masks
                    if len(possible_masks) > max_masks_for_consideration:
                        #Filter by color + size
                        obj_size = get_obj_size(custom_obj,self.ood_objects[custom_obj]) 
                        prompt = "Given this object description, is there a particular color of the object we should look for?\n" + "Object description: " + self.ood_objects[custom_obj] 
                        response,_ = generate_with_openai(prompt) 
                        if 'no' in response.lower():
                            possible_masks = filter_masks(possible_masks,image_data['image'][frame],obj_size,frame) 
                        else:
                            colors = extract_colors(response)
                            obj_size = get_obj_size(custom_obj,self.ood_objects[custom_obj]) 
                            possible_masks = filter_masks(possible_masks,image_data['image'][frame],obj_size,frame,color=colors)  
                        if len(possible_masks) > max_masks_for_consideration: 
                            #idk what to do if this happens :/ 
                            raise OSError 
                    custom_obj_masks = audition_masks(custom_obj,possible_masks,image_data['image'][frame],multiple_objects=multi_obj) 
                    #6. Publish the detection given the object mask 
                    self.process_custom_detection_results(self,image_data,custom_obj_masks,frame)

            except requests.exceptions.RequestException as e:
                rospy.loginfo("Error detecting custom objects: {}".format(str(e)))

    def get_contour_pts(self, image, detection):
        rospy.loginfo("getting contour points...")
        encoded_image = process_image(image)
        request = {
            'action': 'generate_masks',
            'image': encoded_image,
            'detection': detection
        }

        try:
            # Make the request to the Segment Anything server
            response = requests.post(self.segment_anything_server_url + "/get_contour_pts", json=request)

            # Check for a successful response
            if response.status_code != 200:
                rospy.logwarn("Failed to get contour points: {}".format(response.text))
                return None, []

            # Get the JSON data from the response
            response_data = response.json()

            # Extract mask and contour points from the JSON response
            mask = np.array(response_data['mask'])
            #rospy.loginfo("This is mask shape rn: {}".format(str(mask.shape)))
            mask = np.reshape(mask, tuple(map(int, image.shape[:2])))  
            contour_pts = response_data['contour_pts']

            #rospy.loginfo("Got Contour points! type(mask): {}, mask.shape: {}".format(str(type(mask)), str(mask.shape)))
            return mask, contour_pts

        except requests.exceptions.RequestException as e:
            rospy.logwarn("Error querying the segment anything server: {}".format(e))
            return None, []

    def process_detection_results(self, image_data, response_data, frame):
        rospy.loginfo("Processing detection results...")

        # Extracting data from the response
        x_coords = response_data["x_coords"]
        y_coords = response_data["y_coords"]
        scores = response_data["scores"]
        labels = response_data["labels"]

        detection_array = ObjectDetectionArray()
        detection_array.header = Header()
        detection_array.header.stamp = image_data['time']

        for i in range(len(labels)):
            detection = ObjectDetection()
            detection.header = detection_array.header
            detection.name = labels[i].encode('ascii','ignore')
            detection.confidence = scores[i]
            rospy.loginfo("x: {}, y:{}".format(int((x_coords[i][0] + x_coords[i][1]) / 2), int((y_coords[i][0] + y_coords[i][1]) / 2)))
            detection.x = int((x_coords[i][0] + x_coords[i][1]) / 2)
            detection.y = int((y_coords[i][0] + y_coords[i][1]) / 2)
            detection.left = int(x_coords[i][0])
            detection.lower = int(y_coords[i][1])
            detection.right = int(x_coords[i][1])
            detection.upper = int(y_coords[i][0])

            # Debugging log
            rospy.loginfo("Fetching contour points for detection: x={}, y={}".format(detection.x, detection.y))
            
            # Get mask and contour points
            obj_mask, contour_pts = self.get_contour_pts(image_data['image'][frame], (detection.x, detection.y))
            
            # Ensure that contour_pts and obj_mask are valid for indexing
            if obj_mask is None or contour_pts is None:
                rospy.logwarn("Failed to retrieve valid mask or contour points.")
                continue

            # Check if contour_pts contains floats or invalid types
            for pt in contour_pts:
                if not isinstance(pt[0], int) or not isinstance(pt[1], int):
                    rospy.logwarn("Invalid contour point with non-integer values: {}".format(pt))

            detection.contour_pts = [Point(x=pt[0], y=pt[1], z=0.0) for pt in contour_pts]

            # Annotate the image
            rospy.loginfo("Annotating image with detection.")
            try:
                detection.image = self.annotate_image(obj_mask, image_data['image'][frame], "bgr8")
            except Exception as e:
                rospy.logwarn("Error during image annotation: {}".format(e))

            detection.header.frame_id = self.cam_frame_ids[frame]
            if detection.header.frame_id is None:
                rospy.logerr('Frame ID is None')
                continue

            # Clean detection to ensure JSON serializability
            detection = clean_detection_for_json(detection)

            rospy.loginfo("Appending detection of type: {} to array!".format(type(detection)))
            detection_array.detections.append(detection)

        rospy.loginfo("len(detection_array): %d", len(detection_array.detections))
        self.detections_publisher.publish(detection_array)

    def get_frame_id(self, x):
        if  x < self.frame_width:
            rospy.logdebug('left')
            return 'left'  # Frame ID of the left camera
        elif x < self.frame_width * 2:
            rospy.logdebug('front')
            return 'front'  # Frame ID of the front camera
        else:
            rospy.logdebug('right')
            return 'right'
        
    def get_center_x(self, x):
        
        if x < self.frame_width:
            return x   # Coordinate is in the left image
        elif x < self.frame_width * 2:
            return x - self.frame_width  # Coordinate is in the front image
        else:
            return x - self.frame_width*2  # Coordinate is in the right image

    def publish_concate_image(self, image):
        """
        Publishes the concatenated image.

        Args:
            image (numpy.ndarray): The concatenated image.
        """
        image_msg = self.bridge.cv2_to_compressed_imgmsg(image)
        self.concate_image_publisher.publish(image_msg)

    def publish_detections_image(self, image, detections):
        """
        Draws bounding boxes and labels on the image.

        Args:
        - image (numpy.ndarray): The image on which to draw, in OpenCV format.
        - detections (list of ObjectDetection): The list of detections. Each detection
        should have attributes 'left', 'lower', 'right', 'upper', 'name'.

        Returns:
        - An image with detections drawn.
        """
        for detection in detections.detections:
            # Coordinates of the bounding box
            x_min = detection.left
            y_min = detection.upper
            x_max = detection.right
            y_max = detection.lower

            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Prepare the label text
            #label_text = f"{detection.name}: {detection.confidence:.2f}"
            label_text = str(detection.name) + ": " + str(np.round(detection.confidence,2)) 

            # Compute text size to create a background for it
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image,
                (x_min, y_min - text_height - 5),
                (x_min + text_width, y_min),
                (0, 255, 0),
                -1,
            )

            # Put the label text on the image
            cv2.putText(
                image,
                label_text,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        self.detected_image_publisher.publish(
            self.bridge.cv2_to_compressed_imgmsg(image)
        )


