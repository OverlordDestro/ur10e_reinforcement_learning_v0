import numpy as np
import os
import mujoco
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# ur10e pick and place task
# the registration call below will automatically add the environment to
# Gymnasium's registry when this module is imported.  You can still
# instantiate the class directly if you prefer to skip the registry.
from gymnasium.envs.registration import register


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ur10eEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = os.path.join(os.path.dirname(__file__), "ur10e_gripper.xml"),
        frame_skip: int = 2,
        no_early_termination: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            no_early_termination,
            **kwargs,
        )

        self.observation_space = spaces.Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float64),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        })

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=self.observation_space,
            **kwargs,
        )

        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.cartesian_action_scale = 0.16  # meters per action step
        self.max_joint_delta = 1.0  # maximum joint target step per control update
        self.joint_lower_limits = self.model.jnt_range[:6, 0].copy()
        self.joint_upper_limits = self.model.jnt_range[:6, 1].copy()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.gripper_state = 0.0
        
        self.stage = 0

        self.elbow_z = 0.0
        self.wrist2_rot = np.eye(3).flatten()
        self.shoulder_rot = np.eye(3).flatten()
        self.object0_pos = np.zeros(3)
        self.object0_rot = np.eye(3).flatten()
        self.episode_success = False
        self.no_early_termination = no_early_termination
        self.hover_pos = None
        self.hover_offset = 0.30
        self.prev_distance = 0.0
        self.gamma = 0.99
        self._gripper_qpos_idx = self.model.jnt_qposadr[self.model.joint("left_driver_joint").id]
        self.grasp_verify_steps = 0
        self.grasp_verified = False
        self.object_grasped = False
        self.close_steps = 0

    def _current_goal_pos(self):
        object0_pos = self.data.site("object0").xpos.copy()
        if self.stage == 0:
            return self.hover_pos
        elif self.stage == 1:
            return object0_pos + np.array([0.0, 0.0, 0.05])
        return self.data.body("target").xpos.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        if action.shape == (7,):
            cart_action = action[:3]
            gripper_signal = action[6]
        elif action.shape == (4,):
            cart_action = action[:3]
            gripper_signal = action[3]
        else:
            raise ValueError(f"Expected action shape (4,) or (7,), got {action.shape}")

        cart_action = np.clip(cart_action, -1.0, 1.0) * self.cartesian_action_scale

        # --- PROXIMITY SLOW-DOWN (approaches 1 & 2) ---
        # Scale action magnitude down when close to the goal so the arm
        # decelerates naturally near the target without changing the policy.
        # Full speed beyond 0.20 m, tapers to 25 % at contact.
        current_dist = float(np.linalg.norm(
            self.data.site("UR10E_TCP").xpos - self._current_goal_pos()
        ))
        proximity_scale = float(np.clip(current_dist / 0.20, 0.25, 1.0))
        cart_action = cart_action * proximity_scale

        # --- POSITION ERROR ---
        pos_error = cart_action

        # --- ORIENTATION ERROR (keep TCP Z pointing down) ---
        flange_rot = self.data.site("UR10E_TCP").xmat.reshape(3, 3)
        current_z = flange_rot[:, 2]
        desired_z = np.array([0, 0, -1])

        rot_error = action[3:6] * 1

        # Scale rotation (important!)
        rot_gain = 0.5
        rot_error *= rot_gain

        # Combine into 6D task space command
        target_delta_6d = np.concatenate([pos_error, rot_error])

        # --- FULL 6D JACOBIAN (position + rotation) ---
        jacp = np.zeros((3, self.model.nv), dtype=np.float64, order="C")
        jacr = np.zeros((3, self.model.nv), dtype=np.float64, order="C")

        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.model.site("UR10E_TCP").id
        )

        # Combine into 6xN Jacobian
        jacobian = np.vstack([jacp[:, :6], jacr[:, :6]])
                
        # Damped least-squares (DLS) for numerical stability near singularities
        damping = 0.05
        jac_t = jacobian.T
        identity = np.eye(jacobian.shape[1])
        joint_delta = (
            jac_t
            @ np.linalg.inv(jacobian @ jac_t + damping**2 * np.eye(6))
            @ target_delta_6d
        )
        dynamic_max_delta = self.max_joint_delta * proximity_scale
        joint_delta = np.clip(joint_delta, -dynamic_max_delta, dynamic_max_delta)

        desired_qpos = self.data.qpos[:6].copy() + joint_delta
        desired_qpos = np.clip(desired_qpos, self.joint_lower_limits, self.joint_upper_limits)

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[:6] = desired_qpos
        ctrl[6] = np.clip((gripper_signal + 1.0) * 0.5 * 255.0, 0.0, 255.0)

        prev_achieved = self.data.site("UR10E_TCP").xpos.copy()
        prev_desired = self._current_goal_pos()
        self.prev_distance = float(np.linalg.norm(prev_achieved - prev_desired))

        if self.stage == 0:
            distance = np.linalg.norm(self.data.site("UR10E_TCP").xpos - self.hover_pos)
        elif self.stage == 1:
            distance = np.linalg.norm(self.data.site("UR10E_TCP").xpos - (self.object0_pos + np.array([0, 0, 0.05])))
        else:
            distance = np.linalg.norm(self.data.site("UR10E_TCP").xpos - self.goal)

        self.do_simulation(ctrl, self.frame_skip)
        self.gripper_state = float(self.data.qpos[self._gripper_qpos_idx]) / 0.9  # normalize to [0, 1], assuming 0.9 is fully closed

        observation = self._get_obs()

        # Cache sim state to self.* BEFORE compute_reward so it can read them
        self.elbow_z      = self.data.body("UR10E_forearm_link").xpos[2]
        self.wrist2_rot   = self.data.body("UR10E_wrist_2_link").xmat.copy().flatten()
        self.shoulder_rot = self.data.body("UR10E_shoulder_link").xmat.copy().flatten()
        self.object0_pos  = self.data.site("object0").xpos.copy()
        self.object0_rot  = self.data.site("object0").xmat.copy().flatten()
        
        # Compute z_dot for orientation reward
        flange_rot = self.data.site("UR10E_TCP").xmat.copy().reshape(3, 3)
        flange_z = flange_rot[:, 2]
        z_dot = float(np.dot(flange_z, np.array([0, 0, -1])))

        # Build info BEFORE compute_reward so it can be passed in
        info = {
            "elbow_z":       self.elbow_z,
            "wrist2_rot":    self.wrist2_rot,
            "shoulder_rot":  self.shoulder_rot,
            "object0_pos":   self.object0_pos,
            "object0_rot":   self.object0_rot,
            "hover_pos":     self.hover_pos,
            "gripper_state": self.gripper_state,   
            "stage":         self.stage,
            "z_dot":         z_dot,
            "prev_distance": self.prev_distance,
            "is_success":    False,
            "grasp_verify_steps": self.grasp_verify_steps,
            "grasp_verified": self.grasp_verified,
            "object_grasped": self.object_grasped,
            "close_steps": self.close_steps,
        }

        
        reward, reward_components = self._compute_reward_components(
            observation["achieved_goal"],
            observation["desired_goal"],
            info,
        )

        info["reward_components"] = reward_components

        ag_pos    = observation["achieved_goal"][:3]
        dg_pos    = observation["desired_goal"][:3]
        flange_rot = self.data.site("UR10E_TCP").xmat.copy().reshape(3, 3)
        flange_z   = flange_rot[:, 2]
        target_down = np.array([0, 0, -1])
        z_dot = np.sum(flange_z * target_down, axis=-1)
        distance   = np.linalg.norm(ag_pos - dg_pos)

        terminated = False
        gripper_open = self.gripper_state < 0.3
        gripper_close = self.gripper_state > 0.45
        if self.stage == 0:
            if distance < 0.06 and z_dot > 0.96 and gripper_open:
                self.stage = 1

        elif self.stage == 1:
            if distance < 0.02 and z_dot > 0.96 and gripper_close:
                terminated = True
                self.episode_success = True
                if self.no_early_termination:
                    terminated = False

        elif self.stage == 2:
            if distance < 0.02 and z_dot > 0.96 and gripper_close and self.object0_pos[2] > 0.05:
                terminated = True
                self.episode_success = True
                if self.no_early_termination:
                    terminated = False
            """if self.close_steps < 50:
                # waiting phase: just close the gripper, don't move
                self.close_steps += 1
                # still reward gripper closing during wait
                arm_vel = np.linalg.norm(self.data.qvel[:6])
                reward = self.gripper_state * 1.0 - arm_vel * 0.1
            else:
                # after 50 steps of closing, start verifying the lift
                self.grasp_verify_steps += 1
                obj_xy_err = np.linalg.norm(self.object0_pos[:2] - ag_pos[:2])

                if self.object0_pos[2] > 0.08 and gripper_close:
                    if not self.grasp_verified:
                        self.grasp_verified = True
                        self.object_grasped = True
                        reward += 10.0
                    reward += max(0.0, 5.0 * (1.0 - obj_xy_err / 0.05))

                    if self.object0_pos[2] > 0.15:
                        self.episode_success = True
                        reward += 10.0
                        terminated = True
                        if self.no_early_termination:
                            terminated = False

                elif self.grasp_verify_steps > 100:
                    self.object_grasped = False
                    self.grasp_verified = False
                    self.stage = 1
                    self.grasp_verify_steps = 0
                    self.close_steps = 0
                    reward -= 25.0"""
        else:
            print("Invalid stage:", self.stage)
            "so far the 2 stages are somewhat successful but at this point I also need to add stage 2 which will lift the box up towards the target"

        # Now that compute_reward has run, update is_success in info
        info["is_success"] = bool(self.episode_success)

        truncated = False
        if self.render_mode == "human":
            target_geom_id = self.model.geom("target_geom").id
            if self.stage == 0:
                self.model.geom_rgba[target_geom_id] = [1, 0, 0, 1]  # red = hover stage
            elif self.stage == 1:
                self.model.geom_rgba[target_geom_id] = [1, 0.5, 0, 1]    # orange = reach stage
            elif self.stage == 2:   
                self.model.geom_rgba[target_geom_id] = [0, 1, 0, 1]      # green = lift stage
            if self.episode_success:
                self.model.geom_rgba[target_geom_id] = [0, 0, 1, 1]  # blue = success 
            self.render()


        return observation, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        total, _ = self._compute_reward_components(achieved_goal, desired_goal, info)
        return total
    
    def _compute_reward_components(self, achieved_goal, desired_goal, info):

        # ── Unpack goals ──────────────────────────────────────────────────────
        ag_pos = achieved_goal[..., :3]
        dg_pos = desired_goal[..., :3]

        # ── Extract per-transition state from info ────────────────────────────
        if isinstance(info, list):
            prev_distance = np.array([i.get("prev_distance", 0.0) for i in info], dtype=np.float64)
        else:
            prev_distance = float(info.get("prev_distance", 0.0))

        if isinstance(info, list):
            z_dot = np.array([i.get("z_dot", 0.0) for i in info])
            object0_pos = np.array([i.get("object0_pos", 0.0) for i in info])
            orientation_reward = (z_dot * 0.5 - 0.5) * 1.0
        else:
            z_dot = info.get("z_dot", 0.0)
            object0_pos = info.get("object0_pos", 0.0)
            orientation_reward = (z_dot * 0.5 - 0.5) * 1.0
        # ── Position reward ───────────────────────────────────────────────────

        diff_vec = ag_pos - dg_pos
        dist_total = np.linalg.norm(diff_vec, axis=-1)

        # Potential-based shaping: phi(s) = -distance
        phi_prev = -prev_distance
        phi_current = -dist_total
        potential_reward = self.gamma * phi_current - phi_prev

        base_reward = -dist_total
        if self.stage == 0:
            reward = base_reward + orientation_reward + potential_reward
        if self.stage == 1:
            reward = base_reward * 1.1 + orientation_reward + potential_reward    
        if self.stage == 2:
            reward = base_reward + orientation_reward + potential_reward
            if object0_pos[2] > 0.05:  # if the object is lifted off the table, give a big bonus
                reward += 10.0  # bonus for lifting the object off the table

        components = {
            "distance_reward": float(np.mean(base_reward)),
            "potential_reward": float(np.mean(potential_reward)),
            "orientation_reward": float(np.mean(orientation_reward)),
            "total_reward": float(np.mean(reward)),
        }
        components_float = {k: float(v) for k, v in components.items()}

        return reward, components_float
    #since gym 0.26, the step function should return (obs, reward, terminated, truncated, info) instead of (obs, reward, done, info) to distinguish between episode termination and truncation due to time limits or other factors.
    #I should use terminated to indicate if the episode ended due to success or failure, and truncated to indicate if it ended due to time limits or other factors. In this case, I will set terminated to False for now and rely on the TimeLimit wrapper to handle episode truncation after a certain number of steps.

    
    def reset_model(self):#resetira model na začetne pogoje
        self.stage = 0
        self.prev_distance_xy = None
        self.prev_distance_z = None
        self.grasp_verify_steps = 0
        self.grasp_verified = False
        self.object_grasped = False
        self.close_steps = 0

        key_id = self.model.key("UR10E_home").id
        #qpos = self.init_qpos#initial position and velocity of the robot arm
        #qvel = self.init_qvel
        qpos = self.model.key_qpos[key_id].copy()
        qvel = self.model.key_qvel[key_id].copy()
        ctrl = self.model.key_ctrl[key_id].copy()

        self.gripper_state = 0.0
        self.episode_success = False

        object0_id = self.model.body("object0").id
        object0_jnt_adr = self.model.body_jntadr[object0_id]

        #distance of the target from origin
        while True:
            xy = self.np_random.uniform(low=-0.8, high=0.8, size=2)#distance of x y z from origin
            z = self.np_random.uniform(low=0.2, high=0.7, size=1)

            self.goal = np.concatenate([xy, z])#combine them together
            
            #donut shaped range
            dist = np.linalg.norm(self.goal[:2])
            if 0.7 < dist < 1.0: 
                break
        
        yaw  = self.np_random.uniform(0, 2 * np.pi)
        half = yaw / 2.0
        goal_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # wxyz

        while True:
            xy2 = self.np_random.uniform(low=-0.8, high=0.8, size=2)
            #z2  = self.np_random.uniform(low=0.025, high=0.025, size=1)
            z2 = np.array([0.025])  # fixed height for the object, can be adjusted as needed
            object_pos = np.concatenate([xy2, z2])
            if 0.3 < np.linalg.norm(object_pos[:2]) < 0.8:
                break
        yaw2  = self.np_random.uniform(0, 2 * np.pi)
        half2 = yaw2 / 2.0
        object_quat = np.array([np.cos(half2), 0.0, 0.0, np.sin(half2)])
        
        #random_rot = random_rotation_matrix(self.np_random)  # see helper below
        #self.goal_rot = random_rot.flatten()
        #self.data.body("target").xpos[:] = self.goal#da ne skace vec tocka
        self.model.body("target").pos[:] = self.goal
        self.model.body("target").quat[:] = goal_quat

        object0_qpos_start = object0_jnt_adr
        qpos[object0_qpos_start:object0_qpos_start+3] = object_pos#the stuff to make the cube get rotated and moved is so garbage
        qpos[object0_qpos_start+3:object0_qpos_start+7] = object_quat#I know theres a better way but I forgor
        # Zero out object0 velocities (6 DOF: linear + angular)
        qvel[object0_qpos_start:object0_qpos_start+6] = 0

        self.hover_pos = object_pos + np.array([0.0, 0.0, self.hover_offset])  # hover above the object


        self.set_state(qpos, qvel)
        self.data.ctrl[:] = ctrl
        self.prev_distance = float(np.linalg.norm(self.data.site("UR10E_TCP").xpos - self.hover_pos))

        # Initialize prev_dist to the actual initial distance after setting the goal
        return self._get_obs()

    def _get_obs(self):#dobi obzervacijo da pol lahko reagira na okolje 
        position = self.data.qpos[:6].flatten()#joint angles
        velocity = self.data.qvel[:6].flatten()#joint velocities

        flange_pos = self.data.site("UR10E_TCP").xpos.copy()
        flange_rot = self.data.site("UR10E_TCP").xmat.copy()
        flange_z_axis = flange_rot.reshape(3, 3)[:, 2]

        target_pos = self.data.body("target").xpos

        object0_pos = self.data.site("object0").xpos.copy()
        if self.stage == 0:
            # hover point: object XY, object Z + 0.15
            desired_pos = self.hover_pos
            #desired_pos = object0_pos
        elif self.stage == 1:
            # stage 1: reach the box itself
            desired_pos = object0_pos
        else:
            # stage 2: lift the box up to the target
            desired_pos = object0_pos
        stage = np.array([self.stage], dtype=np.float64)

        return {
            "observation": np.concatenate([
                np.cos(position),
                np.sin(position),
                velocity,
                flange_z_axis,
                object0_pos,
                self.data.site("object0").xmat.copy().flatten(),
                stage
            ]).astype(np.float64),
            "achieved_goal": flange_pos.astype(np.float64),
            "desired_goal": desired_pos.astype(np.float64),
        }

# register custom environment automatically on import
# you can call `gym.make("UR10E-v0")` once this module is imported
register(
    id="UR10E-v0",
    entry_point="ur10e:ur10eEnv",
    max_episode_steps=500,
) 