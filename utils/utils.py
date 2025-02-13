import pickle
from ultralytics import YOLO
import numpy as np
def save_annotaion_pkl(frames , model_path, pkl_path):
    """
    Save the annotations in a pickle file.
    """

    # load model
    model = YOLO(model_path)

    # store annotations in a dictionary where keys are frame numbers 
    # and values are lists of dictionaries containing bounding box and keypoint coordinates for each object in the frame
    annotations = {}

    # iterate over frames and track objects in each frame
    for i, frame in enumerate(frames):
        
        annotations[i] = [] 
        results = model.track(frame)

        # for each object in the frame, add its bounding box and keypoint coordinates to the annotations dictionary
        for result in results:
            for box, keypoint in zip(result.boxes, result.keypoints):
                
                if box.id is None:  
                    continue  # Skip this frame if no tracking IDs exist
                
                if int(box.cls) == 0:
                        if int(box.id) == 2 or int(box.id) == 6 or int(box.id) == 5:
                            id = 2
                        else:
                            id = int(box.id)
                      
                        data_point = {
                                    "id": id,
                                    "bbox": box.xyxy[0].tolist(),
                                    "keypoint": keypoint.xy[0].tolist() 
                                    }
                        

                annotations[i].append(data_point)

    with open(pkl_path, 'wb') as file:
        pickle.dump(annotations, file)

def load_annotations_pkl(pkl_path):
    """
    Load annotations from a pickle file.
    """
    with open(pkl_path, 'rb') as file:
        data_annotations = pickle.load(file)
    
    return data_annotations


def calc_center(bbox):
    """
    Calculate the center of a bounding box.
    """
    x1,y1,x2,y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return(center_x, center_y)

def calc_distance(p1,p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    