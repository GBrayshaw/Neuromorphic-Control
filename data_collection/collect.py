from utils.utils_data import generate_target_df
from utils.utils_cam import EventCamera, FrameCamera
from utils.utils_FT import FTSensor

def main():
    # Generate target dataframe
    dataPath = "target_data.csv"
    poses_rng = [[0, 0, 1.0, 25, 25, 0], [0, 0, 5.0, -25, -25, 0]] # static pose range (x,y,z,Rx,Ry,Rz). essentially depth + orientation
    num_poses = 5000 # how many poses
    num_frames = 1
    moves_rng = [[-5, -5, 0, -10, -10, -5], [5, 5, 0, 10, 10, 5]] # moves range (x,y,z,Rx,Ry,Rz). Tangential and rotational moves, slides.
    
    target_df = generate_target_df(f"{dataPath}/targets.csv", poses_rng, num_poses, num_frames, obj_poses=[[0,]*6], moves_rng=moves_rng)

    # Initialize sensors
    event_cam = EventCamera()
    frame_cam = FrameCamera()
    ft_sensor = FTSensor()

    # Collection loop here:
    # for each row in target_df:
    #     move robot to pose
    #     collect data from sensors
    #     save data with corresponding target info

    # Some async magic will be required to collect data from all sensors at the same time, and save them with the correct target info.

    

