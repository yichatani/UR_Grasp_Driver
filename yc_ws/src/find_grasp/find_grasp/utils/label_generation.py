""" Dynamically generate grasp labels during training.
    Author: chenxi-wang
"""

import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'knn'))

#from knn.knn_modules import knn
from utils.loss_utils import GRASP_MAX_WIDTH, batch_viewpoint_params_to_matrix, \
    transform_point_cloud, generate_grasp_views


def process_grasp_labels(end_points):
    pass

def match_grasp_view_and_label(end_points):
    pass

def process_meta_grasp_labels(end_points):
    pass