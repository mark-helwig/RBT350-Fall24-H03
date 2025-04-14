import math
import numpy as np
import copy
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10   # length of link 1
LOWER_LEG_OFFSET = 0.13   # length of link 2
TOLERANCE = 0.01        # Tolerance for inverse kinematics
PERTURBATION = 0.0001   # perturbation for finite difference method
MAX_ITERATIONS = 10     

def ik_cost(end_effector_pos, guess):
    forward_position = forward_kinematics.fk_foot(guess)[0:3, 3]
    cost = np.linalg.norm(end_effector_pos - forward_position)
    return cost

def calculate_jacobian_FD(joint_angles, delta):
    
    J = np.zeros((3, 3))
    current_pos = forward_kinematics.fk_foot(joint_angles)[0:3, 3]
    for i in range(3):
        perturbed_angles = np.array(joint_angles, dtype=float)
        perturbed_angles[i] += delta
        perturbed_pos = forward_kinematics.fk_foot(perturbed_angles)[0:3, 3]
        
        J[:, i] = (perturbed_pos - current_pos) / delta
    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    
    #previous_cost = np.inf
    for it in range(MAX_ITERATIONS):
        current_transform = forward_kinematics.fk_foot(guess)
        current_pos = current_transform[0:3, 3]
        residual = end_effector_pos - current_pos
        cost = np.linalg.norm(residual)

        # able to get convergence with cost difference and cost value
        #if abs(previous_cost - cost) < TOLERANCE:
        if cost < TOLERANCE:
            break
        #previous_cost = cost
        
        J = calculate_jacobian_FD(guess, PERTURBATION)
        delta_theta = np.dot(np.linalg.pinv(J), residual)

        guess = guess + delta_theta
    return guess
