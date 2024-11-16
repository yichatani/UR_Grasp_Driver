import numpy as np
import json
from pathlib import Path

'''this will be the mother of all my libraries. always keep this updated. second test'''

class Camera:
  def __init__(self, cam_name, intrinsic, distortion, base_2_cam_transform):
    self.cam_name = cam_name
    self.intrinsic = intrinsic
    self.intrinsic_mtx = np.zeros((3,3), dtype=np.float64)
    self.intrinsic_mtx[0, 0] = self.intrinsic[2]
    self.intrinsic_mtx[0, 2] = self.intrinsic[4]
    self.intrinsic_mtx[1, 1] = self.intrinsic[3]
    self.intrinsic_mtx[1, 2] = self.intrinsic[5]
    self.intrinsic_mtx[2, 2] = 1

    self.distortion = np.array(distortion)
    self.base_2_cam_transform = base_2_cam_transform
  
  @classmethod
  def load_from_path(cls, cam_name, file_path):
    '''give the name of the camera manually and the file_path as a pathlib object. No need in string type'''
    assert file_path.exists(), f"{file_path.name} does not exist in the path"
    with open(str(file_path), 'r') as infile:
      results_dict = json.load(infile)

    try:
      results_dict["Intrinsics_values"]
      results_dict["Distortion_coefficients"]
      results_dict["Final_handeye"]
      results_dict["Final_handeye"]["xyzquat"]

    except KeyError as e:
      print(f"KeyError! {e} is missing.")
      print(f"Are you sure the below path has all the values?")
      print(f"{file_path}")
      exit()

    
    return cls(cam_name,
               results_dict.get("Intrinsics_values"), 
               results_dict.get("Distortion_coefficients"),
               results_dict.get("Final_handeye").get("xyzquat"))
