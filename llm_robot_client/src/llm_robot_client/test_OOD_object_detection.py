import rospy 
import cv2 
import base64 
from llm_utils import generate_with_openai   
from OOD_utils import * 
import requests 
import json 

# Send a request and print the response
def send_request(self, request):
    if request['action'] == 'initialize':
        rospy.loginfo("Initting the segment anything server...")
        self.segment_anything_proc.stdin.write(json.dumps(request) + "\n")
        self.segment_anything_proc.stdin.flush()  
    else: 
        try:
            rospy.loginfo("Sending request to segment anything server ... this is the action: %s",request['action'])
            # Send request to the Segment Anything server
            self.segment_anything_proc.stdin.write(json.dumps(request) + "\n")
            self.segment_anything_proc.stdin.flush()

            # Read the response
            response = self.segment_anything_proc.stdout.readline().strip()

            # If there's no response (which might be the case for init), return a default success message
            if not response:
                return {"status": "success", "message": "No response, assuming success for action: {}".format(request.get('action', 'unknown'))}
            
            # Try to parse the response as JSON
            response = json.loads(response)
            return response
        
        # Handle JSON decoding errors
        except ValueError as e:  # Use ValueError in Python 2
            rospy.loginfo("[object_detection_client] Error decoding response: %s", str(e))
            return {"status": "error", "message": "Invalid JSON response"}

        # Handle any other exception that might occur
        except Exception as e:
            rospy.loginfo("Error sending request: %s", str(e))
            return {"status": "error", "message": str(e)}
            
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

def segment_ground_mask(image,frame,masks,max_masks_for_consideration):
    #5. Get ground and non-ground masks 
    ground_plane_z = get_ground_plane_z(current_ground_plane_msg) 
    pts_inside_frustrum = get_ground_plane_pts(robot_tfs,current_odom_msg,current_ground_plane_msg,frame) 
    ground_plane_px_coords = inverse_projection(pts_inside_frustrum,frame) 
    grounded_px_coords = sample_ground_pts(ground_plane_z,ground_plane_px_coords)
    request = {
        'action': 'predict_masks',
        'px_coords': grounded_px_coords,  
        'image': process_image(image) 
    }

    potential_ground_masks,_ = send_request(request) 
    if len(potential_ground_masks) > max_masks_for_consideration:
        potential_ground_masks = pick_largest_k_masks(potential_ground_masks,max_masks_for_consideration) 

    largest_ground_mask = audition_masks('ground',potential_ground_masks,image) 

    ground_masks,non_ground_masks = get_ground_masks(largest_ground_mask,masks) 
    return ground_masks,non_ground_masks 

def detect_OOD_objects(self, image_data, max_masks_for_consideration=10):
    rospy.loginfo("detecting OOD objects...")
    request = {
                'action': 'generate_masks',
                'image': process_image(image_data) 
            }
    masks = send_request(request) 
    ground_masks,non_ground_masks = segment_ground_mask(image_data,frame,masks,max_masks_for_consideration)
    #cv2.imwrite("tmp_frame.jpg",image_data['image'][frame])
    try:
        for custom_obj in ood_objects.keys:
            #1. Ask ChatGPT if foo is in the frame 
            prompt = "The user has defined a"+custom_obj+"like this: \n" + ood_objects[custom_obj]
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
            on_ground = is_ground_obj(custom_obj,ood_objects[custom_obj],image_data['image'][frame])
            # The following steps are to get the obj mask(s) in this frame 
            #4. Extract all the masks
            if on_ground: 
                possible_masks = ground_masks 
            else:
                possible_masks = non_ground_masks
            if len(possible_masks) > max_masks_for_consideration:
                #Filter by color + size
                obj_size = get_obj_size(custom_obj,ood_objects[custom_obj]) 
                prompt = "Given this object description, is there a particular color of the object we should look for?\n" + "Object description: " + ood_objects[custom_obj] 
                response,_ = generate_with_openai(prompt) 
                if 'no' in response.lower():
                    possible_masks = filter_masks(possible_masks,image_data['image'][frame],obj_size,frame) 
                else:
                    colors = extract_colors(response)
                    obj_size = get_obj_size(custom_obj,ood_objects[custom_obj]) 
                    possible_masks = filter_masks(possible_masks,image_data['image'][frame],obj_size,frame,color=colors)  
                if len(possible_masks) > max_masks_for_consideration: 
                    #idk what to do if this happens :/ 
                    raise OSError 
            custom_obj_masks = audition_masks(custom_obj,possible_masks,image_data['image'][frame],multiple_objects=multi_obj) 
            #6. Publish the detection given the object mask 
            process_custom_detection_results(self,image_data,custom_obj_masks,frame)

    except requests.exceptions.RequestException as e:
        rospy.loginfo("Error detecting custom objects: {}".format(e))

if __name__ == '__main__': 
    img = cv2.imread("/home/marble/ex_image.jpg") 
