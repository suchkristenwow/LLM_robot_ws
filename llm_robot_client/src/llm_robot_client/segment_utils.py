import json
import sys 
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from geometry_msgs.msg import Point 
import numpy as np 
import cv2 
import rospy 

def handle_request(request):
    print("Handling request...")
    global server
    if request['action'] == 'initialize':
        server = SegmentAnythingServer()
        return {"status": "Class initialized"}

    elif request['action'] == 'get_contour_pts':
        print("trying to get contour points ...") 
        detection = request.get('detection',1) 
        contour_pts,largest_contour = server.get_contour_points(detection) 
        return {"status": "Done", "contour_pts": contour_pts, "largest_contour":largest_contour}

    elif request['action'] == 'generate_masks':
        print("generating masks ...")
        image_data = request.get('image',1)
        masks = server.generate_masks(image_data) 
        return {"status":"Done","masks":masks} 

    elif request['action'] == 'predict_masks': 
        print("predicting masks ...")
        px_coords = request.get("px_coords",1)
        image_data = request.get("image",1)
        masks = server.predict_masks(px_coords,image_data) 
        return {'status':'Done','masks':masks} 
    
class SegmentAnythingServer:
    def __init__(self): 
        sam_checkpoint = "/home/marble/LLM_robot_ws/src/llm_robot_client/src/llm_robot_client/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.seg_anything_predictor = SamPredictor(sam) 
        
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def get_contour_pts(self,detection): 
        '''
        Input is a ObjectDetection msg. Want to find the contour px coordinates using seg anything 
        '''
        self.seg_anything_predictor.set_image(detection.image) 
        input_point = (detection.x,detection.y)  
        input_label = 1
        masks, scores, _ = self.seg_anything_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        ) 
        best_mask = masks[np.argmax(scores)] 
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        # Extract the perimeter points (x, y coordinates)
        perimeter_px_coords = largest_contour.reshape(-1, 2)    
        perimeter_points = [] 
        for pt in perimeter_px_coords:
            pt_i = Point() 
            pt_i.x = pt[0]; pt_i.y = pt[1]; 
            perimeter_points.append(pt_i) 

        return largest_contour,perimeter_points 

    def generate_masks(self,image):
        return self.mask_generator.generate(image) 
    
    def predict_masks(self,px_coords,image):
        self.seg_anything_predictor.set_image(image) 
        masks, scores, _ = self.seg_anything_predictor.predict(
            point_coords=px_coords,
            point_labels=np.ones((len(px_coords),)),
            multimask_output=True,
        ) 
        return masks,scores 

if __name__ == '__main__':
    server = None
    while True: 
        input_line = sys.stdin.readline().strip()
        rospy.loginfo("this is the contents of the request: ",input_line)
        if not input_line:
            break 
        try:
            request = json.loads(input_line)
            response = handle_request(request)
            print(json.dumps(response))
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": "Invalid JSON input"}))
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
        
        sys.stdout.flush()