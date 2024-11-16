import numpy as np
from pathlib import Path
import json
from datetime import datetime
import cv2

class FileSaver:
  def __init__(self, timeless_path):
    self.timeless_path = timeless_path
    
  def save(self, poses=None, images=None, pose_frame_id=None, image_frame_id=None):
    assert len(poses) == len(images), f"The number of poses({len(poses)}) does not match the number of images({len(images)})."

    #only at the instance of saving, do we add in a subfolder with the timestamp
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%SH")
    folderpath = self.timeless_path.joinpath(dt_string)
    folderpath.mkdir(parents=True, exist_ok=True)
    
    posepath = folderpath.joinpath("poses.json")
    # pose_dict = {index: pose for index, pose in enumerate(poses)}
    pose_dict = {"pose_frame_id": pose_frame_id, "image_frame_id": image_frame_id, "poses": poses}

    json_obj = json.dumps(pose_dict, indent=4)
    with open(str(posepath), "w") as outfile:
      outfile.write(json_obj)
    
    for idx, each_img in enumerate(images):
      cv2.imwrite(str(folderpath.joinpath(f"{idx:02}.png")), each_img)

class FileLoader:
  def __init__(self, full_path):
    self.full_path = full_path
    self.get_poses_dictionary()

  def get_poses_dictionary(self):
    with open(str(self.full_path.joinpath("poses.json")), 'r') as infile:
      self.poses_dict = json.load(infile)

  def get_results_dictionary(self):
    assert self.full_path.joinpath("results.json").exists(), "results.json does not exist in the path"
    with open(str(self.full_path.joinpath("results.json")), 'r') as infile:
      self.results_dict = json.load(infile)

    
  def get_images(self):
    assert self.full_path.exists(), "Path does not exist."
    
    self.images_path = list(self.full_path.glob("*.png"))
    self.images_path.sort()
    print(f"Images found: {len(self.images_path)}")
    
    images = []
    for ea in self.images_path:
      images.append(cv2.imread(str(ea)))
    
    return images
  
  def get_poses(self):
    assert len(self.images_path) == len(self.poses_dict["poses"]), f"Poses found: {len(self.poses_dict['poses'])}\t\tImages found: {len(self.images_path)}"
    print(f"Poses found: {len(self.poses_dict['poses'])}")
    return self.poses_dict["poses"]
  
  def get_image_frame_id(self):
    return self.poses_dict["image_frame_id"]

  def write_cam_calib_results(self, results_dict):
    json_obj = json.dumps(results_dict, indent=1)
    with open(str(self.full_path.joinpath("results.json")), "w") as outfile:
      outfile.write(json_obj)
      print("Saved camera calibration into results.json file")

  def append_handeye_results(self, key, value):
    #we copy the entire data that is already in there, and then rewrite every single thing all over again.
    with open(str(self.full_path.joinpath("results.json")), "r") as outfile:
      ori_data = json.load(outfile)

    ori_data[key] = value
    json_obj = json.dumps(ori_data, indent=1)

    with open(str(self.full_path.joinpath("results.json")), "w") as outfile:
      outfile.write(json_obj)
      print("Saved handeye calibration into results.json file")

  def append_final_handeye_results(self, key, value):
    #we copy the entire data that is already in there, and then rewrite every single thing all over again.
    with open(str(self.full_path.joinpath("results.json")), "r") as outfile:
      ori_data = json.load(outfile)

    ori_data[key] = value
    json_obj = json.dumps(ori_data, indent=1)

    with open(str(self.full_path.joinpath("results.json")), "w") as outfile:
      outfile.write(json_obj)
      print("Saved final handeye into results.json file")

if __name__ == "__main__":
  abc = FileSaver("xyz")