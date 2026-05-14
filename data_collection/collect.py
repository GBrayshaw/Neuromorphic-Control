from utils.utils_data import generate_target_df
from utils.utils_cam import EventCamera, FrameCamera
from utils.utils_FT import FTSensor

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import PyfrankaController

def main():
    # Generate target dataframe
    dataPath = "target_data.csv"
    poses_rng = [[0, 0, 1.0, 0, 0, 0], [0, 0, 5.0, 0, 0, 0]] # static pose range (x,y,z,Rx,Ry,Rz). essentially depth + orientation
    num_poses = 5000 # how many poses
    num_frames = 1
    moves_rng = [[-5, -5, 0, -30, -30, -10], [5, 5, 0, 30, 30, 10]] # moves range (x,y,z,Rx,Ry,Rz). Tangential and rotational moves, slides.
    
    target_df = generate_target_df(f"{dataPath}/targets.csv", poses_rng, num_poses, num_frames, obj_poses=[[0,]*6], moves_rng=moves_rng)

    # Initialize sensors
    event_cam = EventCamera()
    frame_cam = FrameCamera()
    ft_sensor = FTSensor()

    # Robot moves example:
    with AsyncRobot(SyncRobot(PyfrankaController(ip='ip'))) as robot:
        for pose, move in zip(target_df['pose'], target_df['move']):

            robot.move_linear(move) # move to wound-up orientation above surface

            robot.async_move_linear(pose + move) # contact surface
            while not robot.async_done():
                pass
                # Collect data from sensors
            robot.async_result() 

            robot.async_move_linear([0, 0, pose[2], 0, 0, 0]) #unwind to 0, maintaining depth by sliding across surface
            while not robot.async_done():
                pass
                # Collect data from sensors
            robot.async_result() 
                

    

    

