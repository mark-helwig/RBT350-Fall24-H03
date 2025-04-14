import math
import numpy as np
import copy
from reacher import forward_kinematics

# Constants (should match those in forward_kinematics)
HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10   # length of link 1
LOWER_LEG_OFFSET = 0.13   # length of link 2

# Inverse kinematics solver parameters
TOLERANCE = 0.01        # Tolerance for convergence (in meters)
PERTURBATION = 0.0001   # Perturbation size for finite difference approximation
MAX_ITERATIONS = 10     # Maximum number of iterations

def ik_cost(end_effector_pos, guess):
    """
    Calculates the inverse kinematics cost, defined as the Euclidean distance
    between the desired end-effector position and the actual end-effector position
    computed from the current guess of joint angles.

    Args:
      end_effector_pos (numpy.ndarray): Desired XYZ coordinates of the end-effector.
      guess (numpy.ndarray): Current guess for the joint angles [hip, shoulder, elbow].

    Returns:
      float: The Euclidean distance (cost).
    """
    forward_position = forward_kinematics.fk_foot(guess)[0:3, 3]
    cost = np.linalg.norm(end_effector_pos - forward_position)
    return cost

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix numerically using finite differences.

    Each column i of the 3x3 Jacobian represents the change in the foot's XYZ
    position with respect to a small change in joint_angles[i].

    Args:
      joint_angles (numpy.ndarray): The current joint angles (3 elements).
      delta (float): The small perturbation added to compute finite differences.

    Returns:
      numpy.ndarray: The 3x3 Jacobian matrix.
    """
    J = np.zeros((3, 3))
    current_pos = forward_kinematics.fk_foot(joint_angles)[0:3, 3]
    for i in range(3):
        perturbed_angles = np.array(joint_angles, dtype=float)
        perturbed_angles[i] += delta
        perturbed_pos = forward_kinematics.fk_foot(perturbed_angles)[0:3, 3]
        # Compute the finite difference derivative for joint i.
        J[:, i] = (perturbed_pos - current_pos) / delta
    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Compute joint angles that achieve the desired end-effector position using
    the Newton-Raphson method.

    In each iteration the algorithm:
      1. Computes the current end-effector position via forward kinematics.
      2. Evaluates the residual (error) between the current and desired positions.
      3. Computes a Jacobian matrix (using finite differences).
      4. Updates the joint angles using the pseudoinverse of the Jacobian.

    The iterations stop when the change in the cost between iterations is below TOLERANCE
    or MAX_ITERATIONS is reached.

    Args:
      end_effector_pos (numpy.ndarray): Desired XYZ position of the end-effector.
      guess (numpy.ndarray): Initial guess for the joint angles [hip, shoulder, elbow].

    Returns:
      numpy.ndarray: Refined joint angles that (approximately) achieve the desired position.
    """
    previous_cost = np.inf
    for it in range(MAX_ITERATIONS):
        current_transform = forward_kinematics.fk_foot(guess)
        current_pos = current_transform[0:3, 3]
        residual = end_effector_pos - current_pos
        cost = np.linalg.norm(residual)
        # Check for convergence; if cost changes less than TOLERANCE, we stop.
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost
        
        # Compute the Jacobian matrix via finite differences.
        J = calculate_jacobian_FD(guess, PERTURBATION)
        # Compute the change in joint angles using the Moore-Penrose pseudoinverse.
        delta_theta = np.dot(np.linalg.pinv(J), residual)
        # Update the joint angle guess.
        guess = guess + delta_theta
    return guess
