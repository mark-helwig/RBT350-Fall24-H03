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

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    # Initialize cost to zero
    cost = 0.0

    # Add your solution here.
    forward_position = forward_kinematics.fk_foot(guess)[0:3,3]
    cost = np.linalg.norm(end_effector_pos - forward_position)
    return cost

    return cost

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """

    # Initialize Jacobian to zero
    J = np.zeros((3, 3))

    # Compute the forward kinematics for the current joint angles
    base_position = forward_kinematics.fk_foot(joint_angles)[0:3, 3]

    # Loop through each joint angle to compute partial derivatives
    for i in range(3):
        # Create a perturbed copy of the joint angles
        perturbed_angles = joint_angles.copy()
        perturbed_angles[i] += delta

        # Compute the forward kinematics for the perturbed joint angles
        perturbed_position = forward_kinematics.fk_foot(perturbed_angles)[0:3, 3]

        # Compute the partial derivative (finite difference approximation)
        J[:, i] = (perturbed_position - base_position) / delta

    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """

    # Initialize previous cost to infinity
    previous_cost = np.inf

    for iters in range(MAX_ITERATIONS):
        # Calculate the Jacobian matrix using finite differences
        J = calculate_jacobian_FD(guess, PERTURBATION)

        # Calculate the current end-effector position using forward kinematics
        current_position = forward_kinematics.fk_foot(guess)[0:3, 3]

        # Calculate the residual (difference between desired and current position)
        residual = end_effector_pos - current_position

        # Compute the step to update the joint angles using the Moore-Penrose pseudoinverse
        step = np.linalg.pinv(J) @ residual

        # Update the guess for joint angles
        guess = guess + step

        # Calculate the cost based on the updated guess
        cost = ik_cost(end_effector_pos, guess)

        # Check for convergence
        if abs(previous_cost - cost) < TOLERANCE:
            break

        # Update the previous cost
        previous_cost = cost

    return guess