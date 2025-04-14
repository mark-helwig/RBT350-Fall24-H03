import math
import numpy as np
import copy

# Constants from the assignment description
HIP_OFFSET = 0.0335            # Shoulder offset from hip (in meters)
UPPER_LEG_OFFSET = 0.10        # Length of link 1 (upper leg)
LOWER_LEG_OFFSET = 0.13        # Length of link 2 (lower leg)

def rotation_matrix(axis, angle):
    """
    Create a 3x3 rotation matrix which rotates about a specific axis.

    Args:
      axis: Array. Unit vector in the direction of the axis of rotation.
      angle: Number. The rotation angle in radians.

    Returns:
      3x3 rotation matrix as a numpy array.
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
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
    """
    Create a 4x4 transformation matrix that rotates about an axis by a given angle and translates by v_A.

    Args:
      axis: Array. Unit vector for the axis of rotation.
      angle: Number. Rotation angle in radians.
      v_A: Vector. Translation from frame A to frame B defined in the A frame.

    Returns:
      4x4 homogeneous transformation matrix as a numpy array.
    """
    R = rotation_matrix(axis, angle)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.array(v_A).flatten()
    return T

def fk_hip(joint_angles):
    """
    Calculate the pose of the hip frame from the base frame.
    
    The hip is assumed to rotate about the z-axis by the hip joint angle.

    Args:
      joint_angles: numpy array of 3 elements [hip_angle, shoulder_angle, elbow_angle] in radians.

    Returns:
      4x4 homogeneous transformation matrix representing the hip frame.
    """
    hip_angle = joint_angles[0]
    T = np.eye(4)
    # Rotate about the z-axis for the hip joint.
    T[0:3, 0:3] = rotation_matrix([0, 0, 1], hip_angle)
    return T

def fk_shoulder(joint_angles):
    """
    Calculate the pose of the shoulder joint.
    
    The shoulder frame is obtained from the hip frame by:
      1. Translating along negative Y by HIP_OFFSET.
      2. Applying the shoulder joint rotation (assumed about y-axis).

    Args:
      joint_angles: numpy array of 3 elements [hip_angle, shoulder_angle, elbow_angle] in radians.

    Returns:
      4x4 homogeneous transformation matrix representing the shoulder frame.
    """
    hip_T = fk_hip(joint_angles)
    
    # Translation from hip to shoulder (offset in negative Y)
    T_translate = np.eye(4)
    T_translate[1, 3] = -HIP_OFFSET
    
    # Shoulder joint rotation about the y-axis.
    shoulder_angle = joint_angles[1]
    T_rotation = homogenous_transformation_matrix([0, 1, 0], shoulder_angle, [0, 0, 0])
    
    # Combined shoulder transformation.
    shoulder_T = hip_T.dot(T_translate).dot(T_rotation)
    return shoulder_T

def fk_elbow(joint_angles):
    """
    Calculate the pose of the elbow joint.

    The elbow frame is obtained from the shoulder frame by:
      1. Translating along the local z-axis by UPPER_LEG_OFFSET.
      2. Applying the elbow joint rotation (assumed about y-axis).

    Args:
      joint_angles: numpy array of 3 elements [hip_angle, shoulder_angle, elbow_angle] in radians.

    Returns:
      4x4 homogeneous transformation matrix representing the elbow frame.
    """
    shoulder_T = fk_shoulder(joint_angles)
    
    # Translation from shoulder to elbow along local z-axis.
    T_translate = np.eye(4)
    T_translate[2, 3] = UPPER_LEG_OFFSET
    
    # Elbow joint rotation about the y-axis.
    elbow_angle = joint_angles[2]
    T_rotation = homogenous_transformation_matrix([0, 1, 0], elbow_angle, [0, 0, 0])
    
    elbow_T = shoulder_T.dot(T_translate).dot(T_rotation)
    return elbow_T

def fk_foot(joint_angles):
    """
    Calculate the pose of the foot (end-effector).

    The foot frame is obtained from the elbow frame by translating along
    the local z-axis by LOWER_LEG_OFFSET.

    Args:
      joint_angles: numpy array of 3 elements [hip_angle, shoulder_angle, elbow_angle] in radians.

    Returns:
      4x4 homogeneous transformation matrix representing the end-effector frame.
    """
    elbow_T = fk_elbow(joint_angles)
    
    # Translation from elbow to foot along local z-axis.
    T_translate = np.eye(4)
    T_translate[2, 3] = LOWER_LEG_OFFSET
    foot_T = elbow_T.dot(T_translate)
    return foot_T
