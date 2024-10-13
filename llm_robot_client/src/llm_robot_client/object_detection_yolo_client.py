#!/usr/bin/env python2

import rospy
import requests
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, PointCloud2, Image 
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import Header, String, Bool, Float32 
from nav_msgs.msg import Odometry 
from llm_robot_client.msg import ObjectDetection, ObjectDetectionArray
import copy
from flask import Flask, request, jsonify
import cv2
import numpy as np
import threading
from collections import deque
import message_filters
import re   
import random 
from OOD_utils import * 
from llm_utils import generate_with_openai  
from robot_transforms import robotTransforms 
import subprocess 
import json 
import base64 
import matplotlib.pyplot as plt 
from datetime import datetime
import time 

"""
class CircularBuffer(deque):
    def __init__(self, size=0):
        super().__init__(maxlen=size)

    def put(self, item):
        self.append(item)
"""

def show_anns(anns,ax):
    print("entered show anns...")
    if len(anns) == 0:
        return
    
    if len(anns) > 1:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    else:
        sorted_anns = anns 

    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

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
    print("cleaning detection ...")
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

        self.ground_level = None 
        self.ground_level_z_sub = rospy.Subscriber(
            "/H03/ground_level_z",Float32,self.ground_level_callback 
        )
        
        rospy.loginfo("Subscribing to " + ground_plane_topic) 
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

        self.debug_frustrum_pts_publisher = rospy.Publisher(
            "debug_frustrum_points",
            PointCloud2,
            queue_size = 10, 
            latch = True 
        )

        self.debug_inverse_projection = rospy.Publisher(
            "debug_inverse_projection", 
            Image, 
            queue_size=10, 
            latch = True 
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

    @staticmethod
    def crop_image_to_bounding_box(mask, orig_image, path):
        print("Cropping image to bounding box...")

        image = copy.deepcopy(orig_image)  # Make a deep copy of the original image

        # Ensure the mask is properly extracted
        if isinstance(mask, dict):
            m = mask['segmentation']
        else:
            m = mask

        # Ensure mask is a binary image of type uint8
        m = m.astype(np.uint8)  # Convert mask to uint8 if it's not already

        # Check if mask is not None and has non-zero size
        if m is not None and m.size > 0:
            # Use OpenCV to find contours
            _, contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print("Number of contours found:", len(contours))

            # If there are contours found, calculate the bounding box for the largest one
            if contours:
                # Find the bounding box for the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)  # Get the bounding box

                # Crop the image to the bounding box
                cropped_img = image[y:y+h, x:x+w]  # Crop the region of the image inside the bounding box

                # Save the cropped image
                print("Saving cropped image at:", path)
                cv2.imwrite(path, cropped_img)
            else:
                raise OSError("No contours found in the mask.")
        else:
            raise OSError("Mask is None or empty.") 

    @staticmethod
    def draw_boundingBox_image(mask, orig_image, format, path):
        print("drawing bound box on image ... ")

        image = copy.deepcopy(orig_image)  # Make a deep copy of the original image

        # Ensure the mask is properly extracted
        if isinstance(mask, dict):
            m = mask['segmentation']
        else:
            m = mask

        # Ensure mask is a binary image of type uint8
        m = m.astype(np.uint8)  # Convert mask to uint8 if it's not already

        # Check if mask is not None and has non-zero size
        if m is not None and m.size > 0:
            # Use OpenCV to find contours (for OpenCV 4.x, it returns 2 values)
            _, contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print("Number of contours found:", len(contours))

            # If there are contours found, calculate the bounding box for the largest one
            if contours:
                # Optionally, you can loop through all contours if you want bounding boxes for all objects
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)  # Get bounding box for each contour

                    # Draw the bounding box on the image
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)  # Convert to uint8 if needed

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

                print("Drew the bounding boxes.")
            else:
                raise OSError("No contours found in the mask.")
        else:
            raise OSError("Mask is None or empty.")

        print("Saving the image with bounding box at:", path)
        cv2.imwrite(path, image)  # Save the modified image
        
    @staticmethod 
    def annotate_image(mask, image, format, path=None):
        """
        Draw the image with the mask on top using transparency.
        """
        # Ensure the mask is properly extracted
        if isinstance(mask, dict):
            m = mask['segmentation']
        else:
            m = mask
       
        # Ensure that `m` is a valid mask, typically a boolean or integer array
        if m is not None:
            # Make sure the mask and the image have compatible shapes
            if m.shape != image.shape[:2]:
                raise ValueError("Mask shape {} does not match image shape {}".format(m.shape, image.shape[:2]))

            # Convert image to float for blending (to handle transparency)
            image_float = image.astype(float)

            # Create an RGBA color mask (red with 35% transparency)
            color_mask = np.array([255, 0, 0, 0.35])  # RGBA color mask

            # Split the alpha channel from the color mask
            alpha = color_mask[3]

            # Create a color mask for RGB channels (ignoring the alpha)
            color_mask_rgb = color_mask[:3]

            # Blend the mask with the image based on the alpha value
            for c in range(3):  # Apply the blend for R, G, B channels
                image_float[:, :, c] = np.where(m, 
                                                (1 - alpha) * image_float[:, :, c] + alpha * color_mask_rgb[c],
                                                image_float[:, :, c])

            # Convert back to uint8
            image = np.clip(image_float, 0, 255).astype(np.uint8)
        else:
            raise OSError("m is NONE type") 
        
        # Create a CompressedImage message
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.format = format
        msg.data = np.array(cv2.imencode('.jpg', image)[1]).tobytes()

        # Save the annotated image if a path is provided
        if path is not None:
            print("annotating image ... writing image to {}".format(path))
            cv2.imwrite(path, image)

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
                    print("getting len(self.image_queue)...")
                    if len(self.image_queue) > 0 and self.object_list is not None:
                        image_data = copy.deepcopy(self.image_queue.pop())
                        self.detect_objects(image_data)
                        if len(self.ood_objects.keys()) > 0:
                            self.detect_OOD_objects(image_data) 
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

    def ground_level_callback(self,msg):
        #print("entered ground_level_callback ... ")
        #print("msg.data:{}".format(msg.data))
        self.ground_level = msg.data 
                
    def odom_callback(self,msg):
        self.current_odom_msg = msg 

    def ground_plane_callback(self,msg):
        #rospy.loginfo("Received Ground Plane message!")
        '''
        if isinstance(msg, PointCloud2):
            rospy.loginfo("Received valid PointCloud2 message")
        else:
            rospy.logerr("Received an invalid message type: %s", type(msg))
        ''' 
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
            #rospy.loginfo("processing frame: %s", frame)
            try:
                success,encoded_image = self.encode_image(image_data['image'][frame])
                # Check if encoding succeeded
                if not success:
                    rospy.logwarn("Failed to encode image for frame: {}".format(frame))
                    continue
                '''
                else:
                    rospy.loginfo("Successfully encoded image for frame: {}".format(frame))
                '''
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
                rospy.loginfo("Received response from object detection server")
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
    
    def publish_annotated_image(self, img, px_coords):
        for idx, (u, v) in enumerate(px_coords):
            # Ensure u and v are integers
            u = int(u)
            v = int(v)
            
            # Draw a circle at each point
            cv2.circle(img, (u,v), 5, (0, 255, 0), -1)  # Green circle with radius 5

        # Convert the OpenCV image to a ROS Image message
        annotated_image_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        rospy.loginfo("Debug inverse projection image publishing ...")
        self.debug_inverse_projection.publish(annotated_image_msg)

    def segment_ground_mask(self,image,frame,max_masks_for_consideration):
        rospy.loginfo("Segmenting ground mask ....")
        #5. Get ground and non-ground masks 
        """
        ground_plane_z = get_ground_plane_z(self.current_ground_plane_msg) 

        pts_inside_frustrum = get_ground_plane_pts(ground_plane_z,self.robot_tfs,self.current_odom_msg,self.current_ground_plane_msg,frame) 

        ### DEBUG ### 
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "H03/base_link"  # Set the appropriate frame

        # Create the PointCloud2 message
        debug_msg = pc2.create_cloud_xyz32(header, pts_inside_frustrum)

        self.debug_frustrum_pts_publisher.publish(debug_msg) 
        ### END DEBUG ### 

        ground_plane_px_coords = inverse_projection(image.shape,pts_inside_frustrum,frame) 

        print("Got ground plane px coords!")
        grounded_px_coords = sample_ground_pts(ground_plane_px_coords)
        if len(grounded_px_coords) == 0:
            raise OSError 
        
        self.publish_annotated_image(image,grounded_px_coords) 
        """ 
        #Get the ground mask 
        encoded_image = process_image(image) 
        request = {
            'action': 'generate_masks',
            'image': encoded_image 
        }

        
        t0 = time.time() 
        rospy.loginfo("Calling Segment Anything server ... calling generate masks: {}".format(t0)) #1728340876.82 
        response_data = requests.post(self.segment_anything_server_url + "/generate_masks",json=request)
        response = response_data.json() 
        print("received response from segment anything server. That took {} secs".format(np.round(time.time()-t0,2)))

        all_masks = response['masks']; all_masks = [np.array(x) for x in all_masks] 

    def detect_OOD_objects(self, image_data, max_masks_for_consideration=5):
        image_list = ['front', 'left', 'right']
        rospy.loginfo("detecting OOD objects...")
        for frame in image_list:
            print("writing {}".format(('/home/marble/tmp_frame.jpg'))) 
            cv2.imwrite('/home/marble/tmp_frame.jpg',image_data['image'][frame]) 

            try:
                for custom_obj in self.ood_objects.keys():
                    #print("custom_obj: ",custom_obj)
                    #1. Ask ChatGPT if foo is in the frame 
                    '''
                    #im commenting this out bc I want to speed things up 
                    prompt = "The user has defined a "+custom_obj+" like this: \n" + self.ood_objects[custom_obj] + ". Is there " + custom_obj + " in this image?"
                    print("prompt: ",prompt)
                    response,history = generate_with_openai(prompt,image_path='/home/marble/tmp_frame.jpg')
                    print("response: ",response) 
                    #2. If so, are there multiple foo 
                    if not 'yes' in response.lower():
                        continue 
                    prompt = "Are there more than one " + custom_obj + " in this image?" 
                    print("prompt: ",prompt)
                    response,history = generate_with_openai(prompt,conversation_history=history,image_path='/home/marble/tmp_frame.jpg') 
                    print("response: ",response)
                    multi_obj = False 
                    if 'yes' in response.lower():
                        multi_obj = True 
                    ''' 
                    
                    multi_obj = True 
                    
                    #3. Would foo on the ground in this image? 
                    #on_ground = is_ground_obj(custom_obj,self.ood_objects[custom_obj],'/home/marble/tmp_frame.jpg') 
                    on_ground = True 

                    encoded_image = process_image(image_data['image'][frame])
                    request = {
                        'action': 'get_ground_mask',
                        'frame':frame, 
                        'time':time.time(),
                        'image': encoded_image 
                    }

                    try:
                        print("Trying to get possible ground masks!")
                        t0 = time.time() 
                        # Make the request to the Segment Anything server
                        rospy.loginfo("Posting request to segment anything server: {}".format('get_ground_mask'))
                        response = requests.post(self.segment_anything_server_url + "/get_ground_mask", json=request)
                        print("recieved response from segment anything server ... that took {} secs".format(np.round(time.time() - t0,2)))
                        # Check for a successful response
                        if response.status_code != 200:
                            rospy.logwarn("Failed to get ground mask: {}".format(response.text))
                            return None, []

                        # Get the JSON data from the response
                        response_data = response.json()

                        # Extract mask and contour points from the JSON response
                        
                        h,w,c = image_data['image'][frame].shape 
                        masks = []
                        for list_mask in response_data['masks']:
                            new_mask = np.array(list_mask); new_mask = np.reshape(new_mask,(h,w))
                            masks.append(new_mask) 

                        #(obj_type, masks, image, img_time, frame,
                        timestamp = datetime.now()

                        # Convert to human-readable format
                        human_readable =  timestamp.strftime("%Y-%m-%d_%H-%M-%S") 

                        #largest_ground_mask = audition_masks('ground',masks,image_data['image'][frame],human_readable,frame)  
                        largest_ground_mask = None 
                        max_area = 0
                        for mask in masks:
                            if np.sum(mask) > max_area:
                                largest_ground_mask = mask 
                                max_area = np.sum(mask) 

                    except requests.exceptions.RequestException as e:
                        rospy.logerr("Error querying the segment anything server: {}".format(e))


                    ground_bb = get_bounding_box_from_mask(largest_ground_mask) 

                    frame_copy = copy.deepcopy(image_data['image'][frame])
                    timestamp = datetime.now()

                    # Convert to human-readable format
                    human_readable =  timestamp.strftime("%Y-%m-%d_%H-%M-%S") 
                    self.crop_image_to_bounding_box(largest_ground_mask, frame_copy, "/home/marble/debug_imgs/" + frame +"_cropped_ground_bounding_box_"+human_readable+".jpg")
                    cropped_img = cv2.imread("/home/marble/debug_imgs/" + frame +"_cropped_ground_bounding_box_"+human_readable+".jpg") 

                    #self.draw_boundingBox_image(largest_ground_mask, image_data['image'][frame], 'bgr8', "/home/marble/ground_bounding_box.jpg") 

                    encoded_cropped_img = process_image(cropped_img) 

                    request = {
                        'action': 'get_possible_object_masks',
                        'cropped_img' : encoded_cropped_img,
                        'frame':frame, 
                        'largest_ground_mask_boundingBox':ground_bb,
                        'grounded':on_ground
                    }

                    try:
                        print("Trying to get possible object masks ...")
                        # Make the request to the Segment Anything server
                        t0 = time.time() 
                        rospy.loginfo("posting request to segment anything server: {}".format('get_possible_object_masks')) 
                        response = requests.post(self.segment_anything_server_url + "/get_possible_object_masks", json=request)
                        print("that took {} seconds.".format(np.round(time.time() - t0,2)))  

                        # Check for a successful response
                        if response.status_code != 200:
                            rospy.logwarn("Failed to get object masks: {}".format(response.text))
                            return None, []

                        # Get the JSON data from the response
                        response_data = response.json()

                        #print("type(response_data['masks'][0]): ",type(response_data['masks'][0]))
                        # Extract mask and contour points from the JSON response
                        possible_masks = [np.array(mask) for mask in response_data['masks']]
                        #pad the mask with False to make it fit the size of the full image 
                        '''
                        possible_masks = []
                        print("image.shape: {}".format(image_data['image'][frame].shape))
                        for mask in raw_masks:
                            print("mask.shape: {}".format(mask.shape))
                            if mask.shape[0] != image_data['image'][frame].shape[0] or mask.shape[1] != image_data['image'][frame].shape[1]:
                                print("padding mask ...")
                                possible_masks.append(pad_mask(mask, image_data['image'][frame].shape)) 
                            else:
                                possible_masks.append(mask) 
                        print("there are {} possible_masks ...".format(len(possible_masks))) 
                        ''' 
                    except requests.exceptions.RequestException as e:
                        rospy.logerr("Error querying the segment anything sergver: {}".format(e))

                    #masks,image_data,obj_size,image_frame 
                    ''' 
                    if multi_obj:
                        print("incrementing max masks for consideration ...")
                        max_masks_for_consideration = max_masks_for_consideration * 5
                    ''' 

                    #possible_masks = filter_masks(possible_masks,image_data['image'][frame],custom_obj,self.ood_objects[custom_obj],obj_size,frame,obj_colors,max_masks_for_consideration) 
                    # Get current timestamp
                    timestamp = datetime.now()

                    # Convert to human-readable format
                    human_readable =  timestamp.strftime("%Y-%m-%d_%H-%M-%S") 

                    custom_obj_masks = audition_masks(custom_obj,possible_masks,cropped_img,human_readable,frame,object_description = self.ood_objects[custom_obj], multiple_objects=multi_obj) 

                    while self.ground_level is None:
                        print("waiting to receive ground level message ... ")
                        rospy.sleep(1.0)

                    custom_obj_masks = filter_ground_masks(custom_obj_masks,cropped_img,on_ground,self.ground_level,frame)

                    #6. Publish the detection given the object mask 
                    self.process_custom_detection_results(self,cropped_img,image_data,custom_obj_masks,frame)

            except requests.exceptions.RequestException as e:
                rospy.loginfo("Error detecting custom objects: {}".format(str(e)))

    def get_contour_pts(self, image, detection_center, detection_bounding_box):
        #rospy.loginfo("getting contour points...")
        encoded_image = process_image(image)
        request = {
            'action': 'generate_masks',
            'image': encoded_image,
            'detection': detection_center,
            'bounding_box': detection_bounding_box 
        }

        try:
            t0 = time.time() 
            rospy.loginfo("posting request to segment anything server!")
            # Make the request to the Segment Anything server
            response = requests.post(self.segment_anything_server_url + "/get_contour_pts", json=request)

            # Check for a successful response
            if response.status_code != 200:
                rospy.logwarn("Failed to get contour points: {}".format(response.text))
                return None, []

            rospy.loginfo("Received response from the server ... that took {} seconds".format(np.round(time.time()-t0,2)))
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

    def process_custom_detection_results(self,obj_name,cropped_img,image_data,masks,frame): 
        #rospy.loginfo("Processing detection results...")

        # Extracting data from the response
        '''
        x_coords = response_data["x_coords"]
        y_coords = response_data["y_coords"]
        scores = response_data["scores"]
        labels = response_data["labels"]
        ''' 

        detection_array = ObjectDetectionArray()
        detection_array.header = Header()
        detection_array.header.stamp = image_data['time']
        
        print("there are {} custom obj masks".format(len(masks)))
        for i in range(len(masks)):
            obj_mask = masks[i]
            detection = ObjectDetection()
            detection.header = detection_array.header
            detection.name = obj_name 
            detection.confidence = float(1.0) 
            #rospy.loginfo("x: {}, y:{}".format(int((x_coords[i][0] + x_coords[i][1]) / 2), int((y_coords[i][0] + y_coords[i][1]) / 2)))
            x,y = get_centroid_from_mask(masks[i])
            detection.x = int(x)
            detection.y = int(y) 
            xmin, ymin, xmax, ymax  = get_bounding_box_from_mask(masks[i])
            detection.left = int(xmin)
            detection.lower = int(ymax)
            detection.right = int(xmax)
            detection.upper = int(ymin)

            mask_uint8 = (obj_mask * 255).astype(np.uint8)

            # Find contours in the mask
            _, contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter_px_coords = largest_contour.reshape(-1, 2)

            # Convert NumPy array to a Python list of tuples with native int type
            contour_pts = [(int(pt[0]), int(pt[1])) for pt in perimeter_px_coords] 

            # Check if contour_pts contains floats or invalid types
            for pt in contour_pts:
                if not isinstance(pt[0], int) or not isinstance(pt[1], int):
                    rospy.logwarn("Invalid contour point with non-integer values: {}".format(pt))

            detection.contour_pts = [Point(x=pt[0], y=pt[1], z=0.0) for pt in contour_pts]

            # Annotate the image
            #rospy.loginfo("Annotating image with detection.")
            if not os.path.exists("/home/marble/debug_tape_detx"):
                os.mkdir("/home/marble/debug_tape_detx") 

            try:
                detection.image = self.annotate_image(obj_mask, cropped_img, "bgr8", path="/home/marble/debug_tape_detx/img_"+str(random.randint(1, 1000))+".jpg") 
            except Exception as e:  
                rospy.logwarn("Error during image annotation: {}".format(e))

            debug_msg = CompressedImage()  
            debug_msg.header = detection_array.header 
            debug_msg.format = "mono8"   
            debug_msg.data = detection.image.data  

            print("publishing custom detected image!!!")
            self.detected_image_publisher.publish(debug_msg)
            
            detection.header.frame_id = self.cam_frame_ids[frame]
            if detection.header.frame_id is None:
                rospy.logerr('Frame ID is None')
                continue

            # Clean detection to ensure JSON serializability
            detection = clean_detection_for_json(detection)

            rospy.loginfo("Appending detection to array!")
            detection_array.detections.append(detection)

        rospy.loginfo("yay we are publishing the custom detections!")

        self.detections_publisher.publish(detection_array)

    def process_detection_results(self, image_data, response_data, frame):
        #rospy.loginfo("Processing detection results...")

        # Extracting data from the response
        x_coords = response_data["x_coords"]
        y_coords = response_data["y_coords"]
        scores = response_data["scores"]
        labels = response_data["labels"]

        detection_array = ObjectDetectionArray()
        detection_array.header = Header()
        detection_array.header.stamp = image_data['time']
        print("there are {} labels".format(len(labels)))
        for i in range(len(labels)):
            detection = ObjectDetection()
            detection.header = detection_array.header
            detection.name = labels[i].encode('ascii','ignore')
            detection.confidence = scores[i]
            #rospy.loginfo("x: {}, y:{}".format(int((x_coords[i][0] + x_coords[i][1]) / 2), int((y_coords[i][0] + y_coords[i][1]) / 2)))
            detection.x = int((x_coords[i][0] + x_coords[i][1]) / 2)
            detection.y = int((y_coords[i][0] + y_coords[i][1]) / 2)
            detection.left = int(x_coords[i][0])
            detection.lower = int(y_coords[i][1])
            detection.right = int(x_coords[i][1])
            detection.upper = int(y_coords[i][0])

            # Debugging log
            #rospy.loginfo("Fetching contour points for detection: x={}, y={}".format(detection.x, detection.y))
            
            # Get mask and contour points
            obj_mask, contour_pts = self.get_contour_pts(image_data['image'][frame], (detection.y, detection.x), (detection.left, detection.lower, detection.right, detection.upper))
            
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
            #rospy.loginfo("Annotating image with detection.")
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

            #rospy.loginfo("Appending detection to array!")
            detection_array.detections.append(detection)

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
        for detection in detections.detections:
            # Ensure the coordinates are integers
            x_min = int(detection.left)
            y_min = int(detection.upper)
            x_max = int(detection.right)
            y_max = int(detection.lower)

            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Prepare the label text
            label_text = str(detection.name) + ": " + str(np.round(detection.confidence, 2))

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
