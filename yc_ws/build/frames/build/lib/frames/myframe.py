import numpy as np
from scipy.spatial.transform import Rotation as R
import math3d as m3d
from geometry_msgs.msg import Pose, TransformStamped
import rclpy


'''this will be the mother of all my libraries. always keep this updated. second test'''

class Myframe:
  def __init__(self, R, posit = (0, 0, 0)):
    self.R = R
    self.posit = np.array(posit, dtype=float)
  
  @property
  def Tmat(self):
    mat = np.zeros((4,4), dtype=float)
    mat[-1, -1] = 1
    mat[:3, :3] = self.R.as_matrix()
    mat[:3, 3] = self.posit
    return mat

  @classmethod
  def from_geom_pose(cls, pose):
    position = pose.position
    posit = (position.x, position.y, position.z)
    orient = pose.orientation
    R_obj = R.from_quat((orient.x, orient.y, orient.z, orient.w))
    return cls(R_obj, posit)

  @classmethod
  def from_UR_pose(cls, URpose):
    posit = (URpose[:3])
    R_obj = R.from_rotvec(URpose[3:])
    return cls(R_obj, posit)

  @classmethod
  def from_Tmat(cls, mat):
    posit = mat[:3, 3]
    R_obj = R.from_matrix(mat[:3, :3])
    return cls(R_obj, posit)

  @classmethod
  def from_rotation_only(cls, dir, ang):
    assert dir in ['x', 'y', 'z'], f"expected only 'x', 'y' or 'z' rotation. received '{dir}'"
    R_obj = R.from_euler(dir, ang, degrees = True)
    return cls(R_obj)

  @classmethod
  def from_m3d_transform(cls, m3dtransform):
    posit = m3dtransform.get_pos()
    R_obj = R.from_matrix(m3dtransform.get_orient().get_array())
    return cls(R_obj, posit)

  @classmethod
  def from_transformstamped(cls, ts):
    trans = ts.transform.translation
    posit = (trans.x, trans.y, trans.z)
    orient = ts.transform.rotation
    R_obj = R.from_quat((orient.x, orient.y, orient.z, orient.w))
    return cls(R_obj, posit)

  @classmethod
  def from_xyzquat(cls, values):
    assert len(values) == 7, f"{__class__}.from_xyzquat() only accepts list of 7 values."
    posit = values[:3]
    R_obj = R.from_quat(values[3:])
    return cls(R_obj, posit)
  
  @classmethod
  def from_xacro_format(cls, posit, rpy):
    """Input should be in order: (x, y, z), (roll pitch yaw) (radians)"""
    roll, pitch, yaw = rpy
    R_obj = R.from_euler('ZYX', [yaw, pitch, roll])
    return cls(R_obj, posit)
    
  def rotate_by(self, dir, ang):
    assert dir in ['x', 'y', 'z'], f"expected only 'x', 'y' or 'z' rotation. received '{dir}'"
    t_rot = __class__.from_rotation_only(dir, ang)
    return self.pose_trans(t_rot)

  def translate_by(self, dir, meters):
    assert dir in ['x', 'y', 'z'], f"expected only 'x', 'y' or 'z' translation. received '{dir}'"
    t_trans_mat = np.identity(4)
    if dir == 'x':
      t_trans_mat[0, 3] = meters
    elif dir == 'y':
      t_trans_mat[1, 3] = meters
    elif dir == 'z':
      t_trans_mat[2, 3] = meters
    else:
      assert True, "you shouldn't be here"

    t_trans_frame = __class__.from_Tmat(t_trans_mat)
    return self.pose_trans(t_trans_frame)

  def as_UR_pose(self):
    return (*self.posit, *self.R.as_rotvec())

  def as_geom_pose(self):
    result = Pose()   #from geometry message
    result.position.x = self.posit[0]
    result.position.y = self.posit[1]
    result.position.z = self.posit[2]

    quat = self.R.as_quat()
    result.orientation.x = quat[0]
    result.orientation.y = quat[1]
    result.orientation.z = quat[2]
    result.orientation.w = quat[3]
    return result

  def as_transform(self, parent, child, node=None):
    result = TransformStamped()
    result.header.frame_id = parent
    result.child_frame_id = child
    result.transform.translation.x = self.posit[0]
    result.transform.translation.y = self.posit[1]
    result.transform.translation.z = self.posit[2]
    quat = self.R.as_quat()
    result.transform.rotation.x = quat[0]
    result.transform.rotation.y = quat[1]
    result.transform.rotation.z = quat[2]
    result.transform.rotation.w = quat[3]
    #result.header.stamp = node.get_clock().now().to_msg()   ####mee
    return result

  def as_m3d_transform(self):
    orient = m3d.Orientation(self.Tmat[:3, :3])
    vect = m3d.Vector(self.Tmat[:3, 3])
    result = m3d.Transform(orient, vect)
    return result

  def as_xyzquat(self):
    """expressed as x, y, z, qx, qy, qz, qw"""
    return (*self.posit, *self.R.as_quat().flatten())
  
  def as_xacro_format(self):
    """Returns in order: x, y, z, roll pitch yaw (radians)"""
    yaw, pitch, roll = self.R.as_euler('ZYX')
    return(*self.posit, roll, pitch, yaw)
  
  def as_ypr_only(self):
    """Returns in order: yaw pitch roll (radians) """
    return self.R.as_euler('ZYX')

  def pose_trans(self, to):
    return __class__.from_Tmat(self.Tmat@to.Tmat)
  
  def inv(self):
    return __class__.from_Tmat(np.linalg.inv(self.Tmat))
  
  def is_orthogonal(self):
    """Checks if a frame is orthogonal, i.e. all the 3 vectors are perpendicular to each other"""
    R_mat = self.R.as_matrix()
    should_be_identity = np.allclose(np.dot(R_mat, np.transpose(R_mat)), np.identity(3))
    should_be_one = np.allclose(np.linalg.det(R_mat), 1)
    return should_be_identity and should_be_one
  
  
  def rad_btw_frames(self, other):
    """Measures the angle in radians between self frame and other frame"""
    return self.rad_btw_quats(self.R.as_quat(), other.R.as_quat())
  
  @classmethod
  def ang_btw_vects(cls, v1, v2):
    """Measures the angle in radians between vector1 and vector2"""
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
  
  @classmethod
  def rad_btw_quats(cls, q1, q2):
    """Measures the angle in radians between quat1 to quat2."""
    d_q1q2 = np.dot(q1, q2)
    inside = (2 * (d_q1q2**2)) - 1
    return np.arccos(inside)



