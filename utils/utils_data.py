import numpy as np
import pandas as pd

def generate_target_df(target_df_file, poses_rng, num_poses, num_frames=1, obj_poses=[[0,]*6], moves_rng=[[0,]*6,]*2, **kwargs):

    np.random.seed(0) # make predictable
    poses = np.random.uniform(low=poses_rng[0], high=poses_rng[1], size=(num_poses, 6))
    poses = poses[np.lexsort((poses[:,1], poses[:,5]))]
    moves = np.random.uniform(low=moves_rng[0], high=moves_rng[1], size=(num_poses, 6))

    pose_ = [f"pose_{_+1}" for _ in range(6)]
    move_ = [f"move_{_+1}" for _ in range(6)]
    target_df = pd.DataFrame(columns=["image_name", "data_name", "obj_id", "pose_id", *pose_, *move_])

    for i in range(num_poses * len(obj_poses)):
        data_name = f"frame_{i}"
        i_pose, i_obj = (int(i%num_poses), int(i/num_poses))
        pose = poses[i_pose,:] + obj_poses[i_obj]
        move = moves[i_pose,:]        
        for f in range(num_frames):
            frame_name = f"frame_{i}_{f}.png"
            target_df.loc[-1] = np.hstack((frame_name, data_name, i_obj+1, i_pose+1, pose, move))
            target_df.index += 1

    target_df.to_csv(target_df_file, index=False)
    return target_df