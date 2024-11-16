#! /usr/bin/env python3

'''this file is to be executed after step 0 [step 0 is replaced by the file 'script_01.py', which helps to capture images and poses], where the images and robot poses were captured. 
You may configure it as eye to hand or eye in hand.
if the program complains that certain image/data sets are problematic (because the grid cannot be found), please remove those image/data pair.
The output will be printed as xyzquat, which can be used in step_2'''
'''due to the way the folder paths are constructed, this script is not meant to be run using ros2 run. Just run it directly from inside vscode'''

import cv2
import numpy as np
from pathlib import Path
from xml.dom import minidom
from myframe import Myframe
from myfileorganiser import FileLoader
from scipy.spatial.transform import Rotation as R
from enum import Enum, auto
import json
from copy import deepcopy

class GridType(Enum):
  SQUARES = auto()
  CIRCLES = auto()

class HandEyeType(Enum):
  EYE_IN_HAND = auto()
  EYE_TO_HAND = auto()

class Calibration:
  def __init__(self, full_path, grid_type = None, pattern = None, distance_mm = None):
    self.full_path = full_path
    self.FL = FileLoader(self.full_path)

    self.images  = self.FL.get_images()
          
    assert grid_type in [GridType.SQUARES, GridType.CIRCLES], f"gridtype can only be 'circles' or 'squares'"
    assert pattern is not None, f"pattern must be filled up with a size i.e. (11,7)"
    assert distance_mm is not None, f"'distance_mm' must not be empty"
    self.grid_type = grid_type
    self.pattern = pattern
    self.dist = distance_mm / 1000   #always enter in mm, it will be converted to m
    
        
    self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    
    self.all_centers = []
    self.calc_centers = []

    if self.grid_type == GridType.CIRCLES:
      obj_centers = self.circles_grid_maker()
    elif self.grid_type == GridType.SQUARES:
      obj_centers = self.chessboard_grid_maker()



    ###check image for grids, and if successful, load that set of data into the master list### 
    self.nice_window("main")
    self.idx_to_pop = []
    for idx, img in enumerate(self.images):
      print(f"Doing: {idx}")
      if self.grid_type == GridType.CIRCLES:
      # retval, centers = cv2.findCirclesGrid(cv2.bitwise_not(img), self.pattern, flags = cv2.CALIB_CB_ASYMMETRIC_GRID |cv2.CALIB_CB_CLUSTERING)
        retval, centers = cv2.findCirclesGrid(cv2.bitwise_not(img), self.pattern, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)
      elif self.grid_type == GridType.SQUARES:
        while True:
          retval, centers, meta = cv2.findChessboardCornersSBWithMeta(img, self.pattern, cv2.CALIB_CB_MARKER+cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_EXHAUSTIVE)
          if retval:
            break
          else:
            # img = self.brighten_image(img)
            # cv2.imshow("brighter", img)
            # cv2.waitKey(300)
            # print(f"Brightened: {idx}")
            print(f"failed {idx}")
            self.idx_to_pop.append(idx)
            break
        cv2.destroyAllWindows()


      if retval:
        cv2.circle(img, np.intp(centers[0][0]), img.shape[1]//100, (255, 0, 0), cv2.FILLED)
        cv2.drawChessboardCorners(img, self.pattern, centers, retval)
        self.all_centers.append(centers)
        self.calc_centers.append(obj_centers)
      else:
        print(f"Please remove image/data set: {self.FL.images_path[idx]} and corresponding pose")
        print("Skipping...")
        # assert False, f"You should remove the above image/pose set as the grid could not be found."                
        
      #uncomment to visualise each image and grid
      cv2.imshow('main', img)
      cv2.waitKey(200)
    cv2.destroyAllWindows()
    
    retval, camMatrix, dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.calc_centers, self.all_centers, (img.shape[1], img.shape[0]), None, None)

    text_intrin = {}
    text_intrin["Reprojection_error"] = retval
    # text_intrin["Intrinsics_matrix"] = newcamMatrix.tolist()
    text_intrin["Intrinsics_matrix"] = camMatrix.tolist()
    text_intrin["Intrinsics_labels"] = ["width", "height", "fx", "fy", "cx", "cy"]
    # text_intrin["Intrinsics_values"] = [img.shape[1], img.shape[0], newcamMatrix[0, 0], newcamMatrix[1, 1], newcamMatrix[0, 2], newcamMatrix[1, 2]]
    text_intrin["Intrinsics_values"] = [img.shape[1], img.shape[0], camMatrix[0, 0], camMatrix[1, 1], camMatrix[0, 2], camMatrix[1, 2]]
    text_intrin["Distortion_coefficients"] = list(dist.flatten())

    self.FL.write_cam_calib_results(text_intrin)
    for key, value in text_intrin.items():
      print(f"{key}: {value}")
    print("Intrinsic camera calibration completed.")
     
  def brighten_image(self, img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] += 15
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

  #if we are doing handeye calibration, then we read all the poses that has been recorded.
  def conduct_Handeye(self, handeye_type):
    assert handeye_type in [HandEyeType.EYE_IN_HAND, HandEyeType.EYE_TO_HAND], f"'handeye_type' can only be 'EYE_IN_HAND' or 'EYE_TO_HAND' if 'handeye' is True"
    

    self.handeye_type = handeye_type
    print(f"Starting hand eye calibration for type '{self.handeye_type}'.")
    self.R_base2gripper = []
    self.T_base2gripper = []
    self.poses = self.FL.get_poses()
    self.idx_to_pop.reverse()
    for pop_idx in self.idx_to_pop:
      self.poses.pop(pop_idx)
    print(f"{len(self.rvecs)} sets of image/pose pairs will be used.")

    for pose_values in self.poses:
      frame_temp = Myframe.from_xyzquat(pose_values)
      if self.handeye_type == HandEyeType.EYE_TO_HAND:
        temp_inv = np.linalg.inv(frame_temp.Tmat)
        self.R_base2gripper.append(temp_inv[:3,:3])
        self.T_base2gripper.append(temp_inv[:3,3])
        
      elif self.handeye_type == HandEyeType.EYE_IN_HAND:
        self.R_base2gripper.append(frame_temp.Tmat[:3,:3])
        self.T_base2gripper.append(frame_temp.Tmat[:3,3])
      else:
        assert False, "You should not be here"    

    R_result, T_result = cv2.calibrateHandEye(self.R_base2gripper, self.T_base2gripper, self.rvecs, self.tvecs, cv2.CALIB_HAND_EYE_TSAI)
    
    Tmat = np.hstack((R_result, T_result))
    Tmat = np.vstack((Tmat, (0, 0, 0, 1)))
    F_result = Myframe.from_Tmat(Tmat)

    key_dict = "Extrinsics"
    sub_dict = {"Translation": F_result.posit.tolist(), "Quaternion": F_result.R.as_quat().tolist(), "xyzquat": F_result.as_xyzquat()}
    self.FL.append_handeye_results(key_dict, sub_dict)

    for key, value in sub_dict.items():
      print(f"{key}: {value}")


  def circles_grid_maker(self):
    coords = []
    for row in range(self.pattern[1]):
      for col in range(self.pattern[0]):
        coords.append(((2 * col + row % 2)*self.dist, row * self.dist, 0))
    self.blank = np.zeros((400, 500, 3), dtype=np.int8)

    # centers = [list(map(int, value)) for value in coords]
    # for each in centers:
    #   cv2.circle(self.blank, each[:2], 5, (255, 200, 200), -1)
    # cv2.imshow('main', self.blank)
    # cv2.waitKey()
    
    return np.array(coords, dtype=np.float32)

  def chessboard_grid_maker(self):
    #have not check functionality, do not use this function yet
    objp = np.zeros((self.pattern[0] * self.pattern[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:self.pattern[0],0:self.pattern[1]].T.reshape(-1,2)
    objp = objp * self.dist
    return objp
  
  def nice_window(self, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280, 720)


if __name__ == "__main__":
  #all collected checkerboard images MUST have the marker. black dot towards bottom left of image.

  # full_path = Path(__file__).parents[2].joinpath("my_robot", "data", "20240319_1553H")      #Realsense_D455 40mm board, mounted based on 3d printed part (furthest)
  full_path = Path(__file__).parents[0].joinpath("data", "20241007_184458H")
  
  img = cv2.imread(str(full_path.joinpath("00.png")))
  cv2.imshow("main", img)
  cv2.waitKey()
  cv2.destroyAllWindows()
  # abc = Calibration(full_path, grid_type = gridtype, pattern = pattern, distance_mm=12)
  # abc = Calibration(full_path, grid_type = GridType.SQUARES, pattern = (8, 11) , distance_mm=20)
  abc = Calibration(full_path, grid_type = GridType.SQUARES, pattern = (11, 8) , distance_mm=40)
  abc.conduct_Handeye(handeye_type=HandEyeType.EYE_IN_HAND)
  



###due to the way the folder paths are constructed, this script is not meant to be run using ros2 run. Just run it direct from inside vscode
