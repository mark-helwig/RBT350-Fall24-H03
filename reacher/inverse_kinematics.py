import math
import numpy as np
import copy
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 10

# def ik_cost(end_effector_pos, guess):
#     """Calculates the inverse kinematics cost.

#     This function computes the inverse kinematics cost, which represents the Euclidean
#     distance between the desired end-effector position and the end-effector position
#     resulting from the provided 'guess' joint angles.

#     Args:
#         end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
#             A numpy array with 3 elements.
#         guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
#             position. A numpy array with 3 elements.

#     Returns:
#         float: The Euclidean distance between end_effector_pos and the calculated end-effector
#         position based on the guess.
#     """
#     # Initialize cost to zero
#     cost = 0.0

#     # Add your solution here.
#     curr_pos = forward_kinematics.fk_foot(guess)[0:3,3] #first three elements of the fourt column
#     cost = np.linalg.norm(end_effector_pos - curr_pos)
#     return cost

# def calculate_jacobian_FD(joint_angles, delta):
#     """
#     Calculate the Jacobian matrix using finite differences.

#     This function computes the Jacobian matrix for a given set of joint angles using finite differences.

#     Args:
#         joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
#         delta (float): The perturbation value used to approximate the partial derivatives.

#     Returns:
#         numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
#         between joint velocity and end-effector linear velocity.
#     """

#     J = np.zeros((3, 3))
#     base_position = forward_kinematics.fk_foot(joint_angles)[0:3, 3] # first three elements of the fourt column
#     # now we have the initi position

#     # each joint angle is its own column
#     for i in range(3):
#         # Perturb the joint angle and compute the forward kinematics
#         per_angles = joint_angles.copy()
#         per_angles[i] += delta

#         per_angles = forward_kinematics.fk_foot(per_angles)[0:3, 3]
#         # finite difference approximation
#         J[0:3,i] = np.array(((per_angles - base_position) / delta).reshape(3,))[0] # this is the finite difference approximation
      
#     return J

# def calculate_inverse_kinematics(end_effector_pos, guess):
#     """
#     Calculate the inverse kinematics solution using the Newton-Raphson method.

#     This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
#     It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

#     Args:
#         end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
#             A numpy array with 3 elements.
#         guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

#     Returns:
#         numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
#     """

#     np.set_printoptions(precision=20)

#     previous_cost = np.inf
#     for iters in range(MAX_ITERATIONS):
#         J = calculate_jacobian_FD(guess, PERTURBATION) # Jacobian defined above
#         current_position = np.array(forward_kinematics.fk_foot(guess)[0:3, 3].reshape(3,))[0] # first three elements of the fourt column
#         # residual (difference between desired and current position)
#         residual = (end_effector_pos - current_position)
        
#         # update the joint angles and the guess using the inverse of the Jacobian
#         step = np.linalg.pinv(J) @ residual
#         guess = (guess + step.reshape(3,)).flatten() # update guess

#         # cost/convergence check
#         cost = ik_cost(end_effector_pos, guess)
        
#         if abs(previous_cost - cost) < TOLERANCE:
#             break
        
#         previous_cost = cost # need to check if cost is increasing?
#     print(guess)
#     return guess

def get_translation(homogeneous_matrix):
    return homogeneous_matrix[:3, 3] / homogeneous_matrix[3, 3]
  
def ik_cost(end_effector_pos, guess):
    guess_pos_homo = forward_kinematics.fk_foot(guess)
    guess_pos = get_translation(guess_pos_homo)
    cost = np.linalg.norm(guess_pos-end_effector_pos)
    return cost
  
def calculate_jacobian_FD(joint_angles, delta):
    foot_pos0 = get_translation(forward_kinematics.fk_foot(joint_angles))
    hip_rotates = joint_angles + [delta,0,0]
    shoulder_rotates = joint_angles + [0,delta,0]
    elbow_rotates = joint_angles + [0,0,delta]
    hip_deltas = get_translation(forward_kinematics.fk_foot(hip_rotates)) - foot_pos0
    shoulder_deltas = get_translation(forward_kinematics.fk_foot(shoulder_rotates)) - foot_pos0
    elbow_deltas = get_translation(forward_kinematics.fk_foot(elbow_rotates)) - foot_pos0
    J = np.column_stack([hip_deltas, shoulder_deltas, elbow_deltas]) / delta
    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    cost = ik_cost(end_effector_pos, guess)
    for iters in range(MAX_ITERATIONS):
      J = calculate_jacobian_FD(guess, PERTURBATION)
      foot_pos = np.array(get_translation(forward_kinematics.fk_foot(guess)).flatten())[0]
      delta_pos = end_effector_pos - foot_pos
      print('end_effector_pos',end_effector_pos)
      print('delta_pos',delta_pos)
      print('foot_pos',foot_pos)
      delta_guess = np.array((np.linalg.pinv(J) @ delta_pos).flatten())[0]
      print('guess',guess)
      print('delta_guess',delta_guess)
      guess += delta_guess
      
      prev_cost = cost
      cost = ik_cost(end_effector_pos, guess)
      if abs(prev_cost - cost) < TOLERANCE:
          break
      print('working')
    print('done',guess)
    return guess