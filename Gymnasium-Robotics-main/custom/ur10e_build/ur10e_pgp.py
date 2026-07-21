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
        # the observation space contains these dimensions:
        # 6 cos(q) + 6 sin(q) + 6 qvel + 3 TCP z-axis + 3 object position + 9 object rotation + 1 stage = 34

        #the achieved and desired goal contain xyz positions of the current observed achieved goal and desired goal
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
        # the action space contains 7 dimensions:
        # 0-2: XYZ position of TCP
        # 3-5: XYZ rotation of TCP
        # 6: gripper signal (how open is the gripper)
        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        #custom settings for the jacobian matrix cartesian controller
        #most of the time you do not change these values, but the cartesian_action_scale and max_joint_delta can be tuned to have the robot move faster or slower
        #changing values can cause instability and will require further tuning of other parameters or experimenting which values are stable
        #try to maintain cartesian_action_scale at rates of 0.04, 0.08, 0.16, 0.32
        #try to maintain max_joint_delta at rates of 0.25, 0.5, 1.0
        #imagine cartesian_action_scale as the speed of the robot (0.04 = 4cm per step)
        #imagine max_joint_delta as the maximum joint angle change per step (0.25 = 0.25 rad per step)
        #slower is stable/precise but slower will have a hard time getting to the target
        #faster will be unprecise but will have an easy time moving between targets
        self.cartesian_action_scale = 0.16  # meters per action step
        self.max_joint_delta = 1.0  # maximum joint target step per control update
        self.rotation_gain = 0.5
        self.damping = 0.05
        self.gamma = 0.99

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
        
        self._gripper_qpos_idx = self.model.jnt_qposadr[self.model.joint("left_driver_joint").id]


    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        #could do the same parse_action as in reach file but would need to restructure due to gripper
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        cart_action = action[:3]
        gripper_signal = action[6]

        cart_action = np.clip(cart_action, -1.0, 1.0) * self.cartesian_action_scale
        # Position error
        pos_error = cart_action

        # TCP orientation error
        flange_rot = self.data.site("UR10E_TCP").xmat.reshape(3, 3)
        current_z = flange_rot[:, 2]
        desired_z = np.array([0, 0, -1])

        rot_error = np.cross(current_z, desired_z)

        rot_error *= self.rotation_gain


        # Combine into 6D task space command
        target_delta_6d = np.concatenate([pos_error, rot_error])

        # 6D jacobian
        jacp = np.zeros((3, self.model.nv), dtype=np.float64, order="C")#pos
        jacr = np.zeros((3, self.model.nv), dtype=np.float64, order="C")#rot

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
        
        jac_t = jacobian.T
        identity = np.eye(jacobian.shape[1])
        joint_delta = (
            jac_t
            @ np.linalg.inv(jacobian @ jac_t + self.damping**2 * np.eye(6))
            @ target_delta_6d
        )
        joint_delta = np.clip(joint_delta, -self.max_joint_delta, self.max_joint_delta)

        desired_qpos = self.data.qpos[:6].copy() + joint_delta
        desired_qpos = np.clip(desired_qpos, self.joint_lower_limits, self.joint_upper_limits)

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[:6] = desired_qpos
        ctrl[6] = np.clip((gripper_signal + 1.0) * 0.5 * 255.0, 0.0, 255.0)

        if self.stage < 2:
            distance = np.linalg.norm(self.data.site("UR10E_TCP").xpos - self.object0_pos)
        else:
            distance = np.linalg.norm(self.object0_pos - self.data.body("target").xpos )

        self.do_simulation(ctrl, self.frame_skip)
        self.gripper_state = float(self.data.qpos[self._gripper_qpos_idx]) / 0.9  # normalize to [0, 1], assuming 0.9 is fully closed

        observation = self._get_obs()

        # cache sim state to self.* BEFORE compute_reward so it can read them
        self.elbow_z      = self.data.body("UR10E_forearm_link").xpos[2]
        self.wrist2_rot   = self.data.body("UR10E_wrist_2_link").xmat.copy().flatten()
        self.shoulder_rot = self.data.body("UR10E_shoulder_link").xmat.copy().flatten()
        self.object0_pos  = self.data.site("object0").xpos.copy()
        self.object0_rot  = self.data.site("object0").xmat.copy().flatten()
        
        # Compute z_dot for orientation reward
        flange_rot = self.data.site("UR10E_TCP").xmat.copy().reshape(3, 3)
        flange_z = flange_rot[:, 2]
        z_dot = float(np.dot(flange_z, np.array([0, 0, -1])))

        #we can pass information about the current state so we can use them indirectly in the reward function and elsewere
        info = {
            "elbow_z":       self.elbow_z,
            "wrist2_rot":    self.wrist2_rot,
            "shoulder_rot":  self.shoulder_rot,
            "object0_pos":   self.object0_pos,
            "object0_rot":   self.object0_rot,
            "gripper_state": self.gripper_state,   
            "stage":         self.stage,
            "z_dot":         z_dot,
            "is_success":    False,
        }

        #lets retrieve all the observations and info we require for checking episode success
        reward, reward_components = self._compute_reward_components(
            observation["achieved_goal"],
            observation["desired_goal"],
            info,
        )

        # the rest of the info is complete only after compute reward
        info["reward_components"] = reward_components

        ag_pos    = observation["achieved_goal"][:3]
        dg_pos    = observation["desired_goal"][:3]
        flange_rot = self.data.site("UR10E_TCP").xmat.copy().reshape(3, 3)
        flange_z   = flange_rot[:, 2]
        target_down = np.array([0, 0, -1])
        z_dot = np.sum(flange_z * target_down, axis=-1)
        distance   = np.linalg.norm(ag_pos - dg_pos)

        terminated = False
        gripper_open = self.gripper_state < 0.3#gripper is open when its less than 0.3
        gripper_close = self.gripper_state > 0.4 #gripper is closed when its more than 0.4
        #small buffer between 0.3 and 0.4 to prevent inbetweens
        if self.stage == 0:
            if distance < 0.06 and gripper_open:
                self.stage = 1
                reward += 5.0  # bonus for reaching the first stage
        elif self.stage == 1:
            if distance < 0.02 and gripper_open:
                self.stage = 2
                reward += 5.0  # bonus for reaching the second stage
        elif self.stage == 2:
            if gripper_close and self.object0_pos[2] > 0.2:
                terminated = True
                self.episode_success = True
                reward += 10.0  # bonus for successful completion
                if self.no_early_termination:
                    terminated = False
            
        else:
            print("Invalid stage:", self.stage)

        # Now that compute_reward has run, update is_success in info
        info["is_success"] = bool(self.episode_success)

        truncated = False

        #for better debugging and viewing, this will change the color of the target depending on the state of the simulation
        if self.render_mode == "human":
            target_geom_id = self.model.geom("target_geom").id
            if self.stage == 0:
                self.model.geom_rgba[target_geom_id] = [1, 0, 0, 1]         # red = reach stage
            elif self.stage == 1:
                self.model.geom_rgba[target_geom_id] = [1, 0.5, 0, 1]       # orange = reach closer stage
            elif self.stage == 2:   
                self.model.geom_rgba[target_geom_id] = [0, 1, 0, 1]         # green = lift stage
            if self.episode_success:
                self.model.geom_rgba[target_geom_id] = [0, 0, 1, 1]         # blue = success 
            self.render()


        return observation, reward, terminated, truncated, info

    #compute reward acts as a intermediary for HER and the tester/loader, enabling us to recover rewards for the plots
    def compute_reward(self, achieved_goal, desired_goal, info):
        total, _ = self._compute_reward_components(achieved_goal, desired_goal, info)
        return total
    
    def _compute_reward_components(self, achieved_goal, desired_goal, info):
        #remember that ag_pos is for achieved goal and dg_pos is for desired goal
        ag_pos = np.asarray(achieved_goal, dtype=np.float64)
        dg_pos = np.asarray(desired_goal, dtype=np.float64)

        single_transition = ag_pos.ndim == 1
        if single_transition:
            ag_pos = ag_pos[None, :]
            dg_pos = dg_pos[None, :]

        #standard reward for distance difference
        dist_total = np.linalg.norm(ag_pos - dg_pos, axis=-1)
        base_reward = -dist_total

        # Pull prev_distance and z_dot from info. Supports scalar info and HER batch info.
        # Since HER batches combine multiple simulations values into one array, we use this to properly use batched values or single values
        if isinstance(info, (list, tuple, np.ndarray)):
            prev_distance = np.array(
                [i.get("prev_distance", 0.0) for i in info],
                dtype=np.float64,
            )
            z_dot = np.array(
                [i.get("z_dot", 0.0) for i in info],
                dtype=np.float64,
            )
        else:
            prev_distance = np.atleast_1d(
                np.asarray(info.get("prev_distance", 0.0), dtype=np.float64)
            )
            z_dot = np.atleast_1d(
                np.asarray(info.get("z_dot", 0.0), dtype=np.float64)
            )

        # Potential-based distance shaping.
        # phi(s) = -distance, so moving closer gives positive reward.
        # there are papers that go deeper into potential shaped rewards
        phi_prev = -prev_distance
        phi_current = -dist_total
        potential_reward = self.gamma * phi_current - phi_prev

        # Small orientation reward for keeping TCP Z-axis downward.
        orientation_reward = (z_dot * 0.5 - 0.5)

        # reward is the combination of all other values that our agent will use
        reward = base_reward + potential_reward + orientation_reward

        #here is where you want to retrieve components for plotting
        components = {
            "distance_reward": float(np.mean(base_reward)),
            "potential_reward": float(np.mean(potential_reward)),
            "orientation_reward": float(np.mean(orientation_reward)),
            "total_reward": float(np.mean(reward)),
        }

        if reward.shape == (1,):
            reward = reward[0]

        return reward, components
    #since gym 0.26, the step function should return (obs, reward, terminated, truncated, info) instead of (obs, reward, done, info) to distinguish between episode termination and truncation due to time limits or other factors.
    #I should use terminated to indicate if the episode ended due to success or failure, and truncated to indicate if it ended due to time limits or other factors. In this case, I will set terminated to False for now and rely on the TimeLimit wrapper to handle episode truncation after a certain number of steps.

    
    def reset_model(self):
        #resetting settings so that next episode has a clean slate, never carry over from previous episodes unless you want bad learning

        self.stage = 0
        self.grasp_verify_steps = 0
        self.grasp_verified = False
        self.object_grasped = False
        self.close_steps = 0

        key_id = self.model.key("UR10E_home").id#settings so the robot resets to its home position instead of its 0 position
                                                #you can change it to some other pose if you know how to
        qpos = self.model.key_qpos[key_id].copy()
        qvel = self.model.key_qvel[key_id].copy()
        ctrl = self.model.key_ctrl[key_id].copy()

        self.gripper_state = 0.0
        self.episode_success = False

        object0_id = self.model.body("object0").id
        object0_jnt_adr = self.model.body_jntadr[object0_id]

        #distance of the target from origin
        x = self.np_random.uniform(low=-0.4, high=0.4, size=1)
        y = self.np_random.uniform(low=0.6,  high=0.8, size=1)
        z = np.array([0.025])
        object_pos = np.concatenate([x, y, z])

        yaw2  = self.np_random.uniform(0, 2 * np.pi)
        half2 = yaw2 / 2.0
        object_quat = np.array([np.cos(half2), 0.0, 0.0, np.sin(half2)])
        
        zt = np.array([0.1])
        self.goal = np.concatenate([x, y, zt])
        yaw  = self.np_random.uniform(0, 2 * np.pi)
        half = yaw / 2.0
        goal_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # wxyz

        self.model.body("target").pos[:] = self.goal
        self.model.body("target").quat[:] = goal_quat

        object0_qpos_start = object0_jnt_adr
        qpos[object0_qpos_start:object0_qpos_start+3] = object_pos#code to move and rotate object isn't the best 
        qpos[object0_qpos_start+3:object0_qpos_start+7] = object_quat#
        # Zero out object0 velocities (6 DOF: linear + angular)
        qvel[object0_qpos_start:object0_qpos_start+6] = 0

        self.set_state(qpos, qvel)
        self.data.ctrl[:] = ctrl

        return self._get_obs()
    
    #getting the observation, nothing special
    def _get_obs(self):
        #joint velocities in theory could be used to punish the robot for moving too fast, but tests have shown no improvement in performance
        position = self.data.qpos[:6].flatten()#joint angles
        velocity = self.data.qvel[:6].flatten()#joint velocities

        flange_pos = self.data.site("UR10E_TCP").xpos.copy()
        flange_rot = self.data.site("UR10E_TCP").xmat.copy()
        flange_z_axis = flange_rot.reshape(3, 3)[:, 2]

        target_pos = self.data.body("target").xpos

        object0_pos = self.data.site("object0").xpos.copy()
        object0_xmat = self.data.site("object0").xmat.copy().flatten()
        if self.stage < 2:
            # stage 0 and 1: reach the box with the TCP
            achieved_pos = flange_pos
            desired_pos = object0_pos
        else:
            # stage 2: lift the box up to the target
            achieved_pos = object0_pos
            desired_pos = target_pos
        stage = np.array([self.stage], dtype=np.float64)

        return {
            "observation": np.concatenate([
                np.cos(position),# Joint angles are encoded as cos/sin to avoid discontinuities
                np.sin(position),# around ±π.
                velocity,
                flange_z_axis,
                object0_pos,
                object0_xmat,
                stage
            ]).astype(np.float64),
            "achieved_goal": achieved_pos.astype(np.float64),
            "desired_goal": desired_pos.astype(np.float64),
        }

# register custom environment automatically on import
register(
    id="UR10E-pgp-v0",
    entry_point="ur10e_pgp:ur10eEnv",
    max_episode_steps=1000,#everything was tested with 1000 steps/ about 10 seconds, anything less or more is bad for learning
) 
