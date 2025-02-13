from utils import calc_distance
def first_person_fight(keypoint_f , keypoint_l):
    """
    Calculates the distance between the first person's keypoints.
    """
    right_wrist = keypoint_f[10]
    left_wrist = keypoint_f[9]
    
    right_wrist_fight = [] 
    left_wrist_fight = []

    for k in keypoint_l[:8]:
        
        if calc_distance(k, right_wrist)< 10:
            right_wrist_fight.append(True)
        
        if calc_distance(k, left_wrist) < 10:
            left_wrist_fight.append(True)
    
    return [right_wrist_fight.count(True) , left_wrist_fight.count(True)] 
