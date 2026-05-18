import argparse
import threading

from utils.utils_data import generate_target_df
from utils.utils_FT import ATI
from core.sensor.tactile_sensor_neuro import NeuroTac

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import PyfrankaController

def make_robot() -> AsyncRobot:
    return AsyncRobot(SyncRobot(PyfrankaController(ip='ip')))

def make_sensor():
    return NeuroTac(save_events_video=False, save_acc_video=False, display=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Data collection for force estimation")
    parser.add_argument("--data-path", type=str, default="target_data.csv", help="Path to save collected data (default: target_data.csv)")
    parser.add_argument("--num-poses", type=int, default=5000, help="Number of poses to collect (default: 5000)")
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames per pose (default: 1)")
    return parser.parse_args()

def main():
    args = parse_args()
    # Generate target dataframe
    dataPath = args.data_path
    poses_rng = [[0, 0, 1.0, 0, 0, 0], [0, 0, 5.0, 0, 0, 0]] # static pose range (x,y,z,Rx,Ry,Rz). essentially depth + orientation
    num_poses = args.num_poses # how many poses
    num_frames = args.num_frames
    moves_rng = [[-5, -5, 0, -30, -30, -10], [5, 5, 0, 30, 30, 10]] # moves range (x,y,z,Rx,Ry,Rz). Tangential and rotational moves, slides.
    
    target_df = generate_target_df(f"{dataPath}/targets.csv", poses_rng, num_poses, num_frames, obj_poses=[[0,]*6], moves_rng=moves_rng)

    # Robot moves example:
    with make_robot() as robot, make_sensor() as camera, ATI() as ft_sensor:
        # Precollection setup
        ft_sensor.tare()

        for pose, move in zip(target_df['pose'], target_df['move']):
            sample_name = f"{dataPath}/{pose[2]}_{move[2]}"

            camera.set_filenames(events_on_file=events_on_file, events_off_file=events_off_file)

            camera.reset_variables()

            robot.move_linear(move) # move to wound-up orientation above surface

            # Init sensor threads
            camera.start_logging()
            t = threading.Thread(target=camera.get_events, args=())
            t.start()

            # Execute movement sequence
            robot.async_move_linear(pose + move) # contact surface
            while not robot.async_done():
                pass
            robot.async_result() 

            robot.async_move_linear([0, 0, pose[2], 0, 0, 0]) #unwind to 0, maintaining depth by sliding across surface
            while not robot.async_done():
                pass
            robot.async_result() 
            
            # Stop sensor threads
            camera.stop_logging()
            t.join()

            # Collate proper timestamp values in ms.
            camera.value_cleanup()      # TODO: Match to ft_sensor timestamps

            # Save data
            camera.save_events_on(events_on_file)
            camera.save_events_off(events_off_file)
            print("saved data")

    

    

