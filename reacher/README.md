### `rotation_matrix(axis, angle)`
Create a 3x3 rotation matrix which rotates about a specific axis.

- **Args:**
  - `axis`: Array. Unit vector in the direction of the axis of rotation
  - `angle`: Number. The amount to rotate about the axis in radians
- **Returns:** 3x3 rotation matrix as a numpy array

---

### `homogenous_transformation_matrix(axis, angle, v_A)`
Create a 4x4 transformation matrix which transforms from frame A to frame B.

- **Args:**
  - `axis`: Array. Unit vector in the direction of the axis of rotation
  - `angle`: Number. The amount to rotate about the axis in radians
  - `v_A`: Vector. The vector translation from A to B defined in frame A
- **Returns:** 4x4 transformation matrix as a numpy array

---

### `fk_hip(joint_angles)`
Use forward kinematics equations to calculate the xyz coordinates of the hip frame given the joint angles of the robot.

- **Args:**
  - `joint_angles`: Numpy array of 3 elements stored as `[hip_angle, shoulder_angle, elbow_angle]`. Angles are in radians.
- **Returns:** 4x4 matrix representing the pose of the hip frame in the base frame

---

### `fk_shoulder(joint_angles)`
Use forward kinematics equations to calculate the xyz coordinates of the shoulder joint.

- **Args:**
  - Same as above
- **Returns:** 4x4 matrix representing the pose of the shoulder frame in the base frame

---

### `fk_elbow(joint_angles)`
Use forward kinematics equations to calculate the xyz coordinates of the elbow joint.

- **Args:** Same as above  
- **Returns:** 4x4 matrix representing the pose of the elbow frame in the base frame

---

### `fk_foot(joint_angles)`
Use forward kinematics equations to calculate the xyz coordinates of the foot.

- **Args:** Same as above  
- **Returns:** 4x4 matrix representing the pose of the end-effector frame in the base frame

---

### `ik_cost(end_effector_pos, guess)`
Calculates the inverse kinematics cost.

- **Description:** Computes the Euclidean distance between the desired end-effector position and the position calculated from the current guess.
- **Args:**
  - `end_effector_pos`: Numpy array with desired XYZ position
  - `guess`: Numpy array guess of joint angles
- **Returns:** Float cost value

---

### `calculate_jacobian_FD(joint_angles, delta)`
Calculate the Jacobian matrix using finite differences.

- **Description:** Computes partial derivatives of end-effector position with respect to joint angles.
- **Args:**
  - `joint_angles`: Numpy array of current joint angles
  - `delta`: Float perturbation value
- **Returns:** 3x3 Jacobian matrix

---

### `calculate_inverse_kinematics(end_effector_pos, guess)`
Calculate the inverse kinematics solution using the Newton-Raphson method.

- **Description:** Iteratively refines joint angles to match the desired end-effector position using the Jacobian and cost functions.
- **Args:**
  - `end_effector_pos`: Desired position (3D)
  - `guess`: Initial joint angle guess
- **Returns:** Final joint angles as a numpy array

---

### `get_translation(homogeneous_matrix)`
Utility function to extract the 3D translation vector from a 4x4 homogeneous matrix.

---