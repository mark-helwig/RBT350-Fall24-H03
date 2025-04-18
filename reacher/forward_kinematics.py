import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """
  c = math.cos(angle)
  s = math.sin(angle)
  C = 1 - c
  x = axis[0]
  y = axis[1]
  z = axis[2]


  rot_mat = np.matrix([[x*x*C + c, x*y*C - z*s, x*z*C + y*s],
                       [y*x*C + z*s, y*y*C + c, y*z*C - x*s], 
                       [z*x*C - y*s, z*y*C + x*s, z*z*C + c]])


  #rot_mat = np.eye(3)
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """

  R = np.row_stack(rotation_matrix(axis,angle), [0,0,0])
  T = np.column_stack
  v = np.numpy.append(v_A,1)
  v = v.reshape(-1,1)
  T = np.hstack((T,v))


  #T = np.eye(4)
  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """
  angle = joint_angles[0]

  hip_frame = np.matrix([[np.cos(angle), -1*np.sin(angle), 0, 0],
                         [np.sin(angle),np.cos(angle), 0 , 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1]])
  
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """
  angle = joint_angles[1]
  hip_frame = fk_hip(joint_angles)
  shoulder_hip_frame = np.matrix([[np.cos(angle), 0, np.sin(angle), 0],
                         [0, 1, 0 , -1*HIP_OFFSET], 
                         [-1*np.sin(angle), 0, np.cos(angle),  0], 
                         [0, 0, 0, 1]])
  shoulder_frame = np.matmul(hip_frame,shoulder_hip_frame)

  return shoulder_frame

def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame
  """
  angle = joint_angles[2]
  shoulder_frame = fk_shoulder(joint_angles)
  elbow_shoulder_frame = np.matrix([[np.cos(angle), 0, np.sin(angle), 0],
                         [0, 1, 0 , 0], 
                         [-1*np.sin(angle), 0, np.cos(angle),  0.1], 
                         [0, 0, 0, 1]])
  
  elbow_frame = np.matmul(shoulder_frame, elbow_shoulder_frame)

  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """
  joint_angles = np.array(joint_angles)
  elbow_frame = fk_elbow(joint_angles)

  end_effector_elbow_frame = np.matrix([[1, 0, 0, 0],
                         [0, 1, 0 , 0], 
                         [0, 0, 1, 0.13], 
                         [0, 0, 0, 1]])
  
  end_effector_frame = np.matmul(elbow_frame, end_effector_elbow_frame)
  
  return end_effector_frame