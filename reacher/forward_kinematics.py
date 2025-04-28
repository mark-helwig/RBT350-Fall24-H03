import math
import numpy as np
import copy

# Constants from the assignment description
HIP_OFFSET = 0.0335            # Shoulder offset from hip (in meters)
UPPER_LEG_OFFSET = 0.10        # Length of link 1 (upper leg)
LOWER_LEG_OFFSET = 0.13        # Length of link 2 (lower leg)

def get_axis_angle(T):
    
    # Extract the rotation matrix from the transformation matrix
    R = T[0:3, 0:3]
    
    # Calculate the angle of rotation using the trace of the rotation matrix
    angle = math.acos((np.trace(R) - 1) / 2)
    
    # Calculate the axis of rotation using the skew-symmetric matrix
    if angle == 0:
        axis = np.array([0, 0, 0])
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    
    return axis, angle

def rotation_matrix(axis, angle):
    
    # Create the skew-symmetric matrix K from the axis.
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    # Rodrigues' rotation formula: R = I + sin(angle)*K + (1-cos(angle))*K^2
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def homogenous_transformation_matrix(axis, angle, v_A):
    
    R = rotation_matrix(axis, angle)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.array(v_A).flatten()
    return T

def fk_hip(joint_angles):
    
    hip_angle = joint_angles[0]
    T = np.eye(4)

    T[0:3, 0:3] = rotation_matrix([0, 0, 1], hip_angle)
    return T

def fk_shoulder(joint_angles):
   
    hip_T = fk_hip(joint_angles)
    
    T_translate = np.eye(4)
    T_translate[1, 3] = -HIP_OFFSET
    
    shoulder_angle = joint_angles[1]
    T_rotation = homogenous_transformation_matrix([0, 1, 0], shoulder_angle, [0, 0, 0])
    
    shoulder_T = hip_T.dot(T_translate).dot(T_rotation)
    return shoulder_T

def fk_elbow(joint_angles):
    
    shoulder_T = fk_shoulder(joint_angles)
    
    T_translate = np.eye(4)
    T_translate[2, 3] = UPPER_LEG_OFFSET
    
    elbow_angle = joint_angles[2]
    T_rotation = homogenous_transformation_matrix([0, 1, 0], elbow_angle, [0, 0, 0])
    
    elbow_T = shoulder_T.dot(T_translate).dot(T_rotation)
    return elbow_T

def fk_foot(joint_angles):
    
    elbow_T = fk_elbow(joint_angles)
    
    T_translate = np.eye(4)
    T_translate[2, 3] = LOWER_LEG_OFFSET
    foot_T = elbow_T.dot(T_translate)
    return foot_T
