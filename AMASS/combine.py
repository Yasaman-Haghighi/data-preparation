import numpy as np
from pathlib import Path
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import argparse
from scipy.spatial.transform import Slerp

def load_motion(path):
  motion = np.load(path)
  return dict(motion)

def find_transform(motion_first, motion_second):
  r0 = R.from_rotvec(motion_first['global_ori'][-1])
  r_0= r0.as_matrix()

  r1 = R.from_rotvec(motion_second['global_ori'][0])
  r_1= r1.as_matrix()

  R21=r_0@r_1.T
  R21[0,1] = 0
  R21[1,0] = 0
  R21[1,1] = 1
  R21[1,2] = 0
  R21[2,1] = 0


  trasn_2 = motion_second['trans'][0]@R21.T
  trans_diff = motion_first['trans'][-1] - trasn_2
  trans_diff[1] = 0

  return R21, trans_diff

def apply_transform(motion, Rotation, Translation):
  r=R.from_rotvec(motion['global_ori'])
  rotation = R.from_matrix(Rotation@(r.as_matrix()))
  motion['global_ori'] = rotation.as_rotvec() 
  motion['poses'][:,0:3] = rotation.as_rotvec()

  motion['trans'] = motion['trans']@Rotation.T + Translation

  return motion

def append_motions(motion_dict_list, n_interpolation=10):
  final_motion = motion_dict_list[0].copy()
  for i in range(1,len(motion_dict_list)):

    n_joints = final_motion['poses'][-1,:].shape[0]//3
    pose_0 = final_motion['poses'][-1,:].reshape(-1,3)
    pose_1 = motion_dict_list[i]['poses'][0,:].reshape(-1,3)
    
    inter_poses = np.zeros((n_interpolation, n_joints*3))

    for j in range(n_joints):  
      poses = np.stack((pose_0[j,:], pose_1[j,:]), axis=0)
      interpolater = Slerp([0,n_interpolation+1], R.from_rotvec(poses))
      pose_interpolate = interpolater(range(1,n_interpolation+1))
      inter_poses[:, j*3:(j*3+3)] = pose_interpolate.as_rotvec()

    inter_trans = np.linspace(0,1,n_interpolation+2)
    inter_trans = inter_trans[1:n_interpolation+1].reshape(n_interpolation,1)
    
    trans_interpolated = (1-inter_trans)@(final_motion['trans'][-1].reshape(1,3)) + inter_trans@(motion_dict_list[i]['trans'][0].reshape(1,3))

    #append interpolations
    final_motion['poses'] = np.vstack((final_motion['poses'], inter_poses))
    final_motion['global_ori'] = np.vstack((final_motion['global_ori'], inter_poses[:,0:3]))
    final_motion ['trans'] = np.vstack((final_motion['trans'], trans_interpolated))

    
    final_motion['poses']=np.vstack((final_motion['poses'], motion_dict_list[i]['poses']))
    final_motion['global_ori']=np.vstack((final_motion['global_ori'], motion_dict_list[i]['global_ori']))
    final_motion['trans']=np.vstack((final_motion['trans'], motion_dict_list[i]['trans']))
    
   
      

  return final_motion

def save_motion(motion_dict, path):
  np.savez(path,motion_info=motion_dict['motion_info'], betas=motion_dict['betas'], 
           poses=motion_dict['poses'], global_ori=motion_dict['global_ori'], 
           trans=motion_dict['trans'], mocap_frame_rate=motion_dict['mocap_frame_rate'], gender=motion_dict['gender'])

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, default='')
  parser.add_argument('--output_dir', type=str, default='')

  args =parser.parse_args()

  folder_path = Path(args.input_dir)
  directories = [entry for entry in folder_path.iterdir() if entry.is_dir()]

  for directory in directories:

    base_dir = Path(directory)

    # Use glob to find all files with a specific extension (e.g., .txt) in subdirectories
    file_pattern = '**/*.npz'  # Adjust the pattern to match your file extensions
    files = base_dir.glob(file_pattern)


    motion_list = []
    # Iterate through the matched files
    for file_path in files:
        # Append the NumPy array to the list
        motion = load_motion(file_path)
        motion_list.append(motion)

    for i in range(len(motion_list)-1): 
    
      Rot, T = find_transform(motion_list[i], motion_list[i+1])
      motion_list[i+1] = apply_transform(motion_list[i+1], Rot, T)

    output_path = Path(args.output_dir)
    appended_motion = append_motions(motion_list, 10)

    save_motion(appended_motion, output_path.joinpath(directory.name+ '.npz'))


