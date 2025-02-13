import cv2

def read_video(video_path):
    """
    Read a video from a given path and return a list of frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        frames.append(frame)

    return frames 

def save_video(frames,output_video_path):
    """
    Save a list of frames as a video to a given path.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()