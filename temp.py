def get_translation(homogeneous_matrix):
    return homogeneous_matrix[:3, 3] / homogeneous_matrix[3, 3]
  
def ik_cost(end_effector_pos, guess):
    guess_pos = forward_kinematics.fk_foot(guess)
    guess_pos = get_translation(guess_pos)
    cost = np.linalg.norm(guess_pos-end_effector_pos)
    return cost
  
def calculate_jacobian_FD(joint_angles, delta):
    foot_pos0 = forward_kinematics.fk_foot(joint_angles)
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
      foot_pos = get_translation(forward_kinematics.fk_foot(guess))
      delta_pos = end_effector_pos - foot_pos
      
      delta_guess = np.linalg.pinv(J) @ delta_pos
      guess += delta_guess
      
      prev_cost = cost
      cost = ik_cost(end_effector_pos, guess)
      if abs(prev_cost - cost) < TOLERANCE:
          break
    return guess