import cv2 
from ultralytics import YOLO 
import matplotlib.pyplot as plt
import pickle as pl
from utils import read_video, save_annotaion_pkl, load_annotations_pkl, calc_center, calc_distance, save_video
from utils import first_person_fight

# Video path
video_path = "F:\programming\computer_vision_nanodegree\projects\Fight_Detection\input_video/fight.mp4"

# Read model file 
model = YOLO("yolo11s-pose.pt")
# Run the whole model to detect bboxs and keypoints 
# results = model.track(video_path , save=True)

# Read Video as a separated frames 
frames = read_video(video_path)

# Save the results in pikcle file to use it fast 
save_annotaion_pkl(frames,"yolo11s-pose.pt", "pickle_data/fight.pkl" )

# Load the saved annotations from pickle file
annotations = load_annotations_pkl("pickle_data/fight.pkl")

# Variables for distance calculation and output frame
dist = 1000000
output_frames = []
is_fight = False

# Loop through all frames 
for frame_num, frame in enumerate(frames):
    

    image = frame.copy()

    # Draw rectangles for the current frame
    # for rect in annotations[frame_num]:
    #     x1, y1, x2, y2 = rect['bbox']
    #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
    # Calculate distance and detect if there fight or not 
    if dist < 50 and not is_fight :
        text = f"Close people: Possible Fight"
    
    elif dist < 50 and is_fight:
        text = f"Fight Detected"
    
    else:
        text = f"Separated people: Safe Enviroment"
    
    # Draw text on the frame related to the current case
    cv2.putText(image, f"{text}", (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    output_frames.append(image)
    
    # Load all annotations related to this frame
    frame_annotaions = annotations[frame_num]
    
    # Loop through all boxes in this frame
    for first in range(len(frame_annotaions)): 
    
        bbox = frame_annotaions[first]['bbox']
        
        # Calculate center of the current box and compare it with others
        center = calc_center(bbox)
        for second in range(len(frame_annotaions)):
            if first != second:
                
                bbox_other = frame_annotaions[second]['bbox']
                center_other = calc_center(bbox_other)
                
                # If distance is less than 50 Loop through closed bboxes to detecect fight or not
                if calc_distance(center, center_other) < 50:
                    dist = calc_distance(center, center_other)  
                    
                    # Get the distance between keypoints
                    fight = first_person_fight(frame_annotaions[first]['keypoint'] 
                                          ,frame_annotaions[second]['keypoint'])

                    # if there are more than 3 keypoints closed the other person wrist then fight
                    if fight[0] > 3 or fight[1] > 3:
                        is_fight = True
                        text = f"Fight Detected"


# save frames to the video 
save_video(output_frames, "out_video/test.avi")              
 