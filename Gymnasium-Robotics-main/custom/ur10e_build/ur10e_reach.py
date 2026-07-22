import numpy as np
import os
import mujoco
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
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
        # 6 cos(q) + 6 sin(q) + 6 qvel + 3 TCP z-axis + 3 object position + 9 object rotation = 33

        #the achieved and desired goal contain xyz positions of the current observed achieved goal (TCP) and desired goal (target)
        self.observation_space = spaces.Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float64),
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
        self.cartesian_action_scale = 0.16
        self.max_joint_delta = 1.0
        self.rotation_gain = 0.5
        self.damping = 0.05
        self.gamma = 0.99

        self.success_distance = 0.05
        self.require_downward_tcp = True
        self.success_z_dot = 0.96

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

        self.no_early_termination = no_early_termination
        self.episode_success = False
        self.prev_distance = 0.0
        self.goal = np.zeros(3, dtype=np.float64)

        self.gripper_state = 0.0
        self._gripper_qpos_idx = self.model.jnt_qposadr[
            self.model.joint("left_driver_joint").id
        ]

        # logging values
        self.elbow_z = 0.0
        self.wrist2_rot = np.eye(3).flatten()
        self.shoulder_rot = np.eye(3).flatten()
        self.object0_pos = np.zeros(3)
        self.object0_rot = np.eye(3).flatten()

    #preparing for jacob
    def _parse_action(self, action):
        action = np.asarray(action, dtype=np.float64).copy()

        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        cart_action = action[:3]
        rot_action = action[3:6]

        cart_action = np.clip(cart_action, -1.0, 1.0) * self.cartesian_action_scale
        rot_action = np.clip(rot_action, -1.0, 1.0) * self.rotation_gain
        return cart_action, rot_action


    def step(self, action):
        cart_action, rot_action = self._parse_action(action)

        # Previous distance for potential-based shaping
        tcp_pos_before = self.data.site("UR10E_TCP").xpos.copy()
        target_pos_before = self.data.body("target").xpos.copy()
        self.prev_distance = float(np.linalg.norm(tcp_pos_before - target_pos_before))

        # The policy commands a small Cartesian TCP delta and rotation residual.
        target_delta_6d = np.concatenate([cart_action, rot_action])

        # 6D Jacobian-based inverse kinematics to compute joint deltas.
        # could probably be computed in its own separate file or definition but for now its here
        jacp = np.zeros((3, self.model.nv), dtype=np.float64, order="C")
        jacr = np.zeros((3, self.model.nv), dtype=np.float64, order="C")

        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.model.site("UR10E_TCP").id,
        )

        jacobian = np.vstack([jacp[:, :6], jacr[:, :6]])
        jac_t = jacobian.T

        joint_delta = (
            jac_t
            @ np.linalg.inv(jacobian @ jac_t + self.damping ** 2 * np.eye(6))
            @ target_delta_6d
        )
        joint_delta = np.clip(joint_delta, -self.max_joint_delta, self.max_joint_delta)

        desired_qpos = self.data.qpos[:6].copy() + joint_delta
        desired_qpos = np.clip(
            desired_qpos,
            self.joint_lower_limits,
            self.joint_upper_limits,
        )

        # Control vector.
        # Gripper is kept open and ignored for this reach-only task.
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[:6] = desired_qpos
        ctrl[6] = 0.0
        #in reach we don't really need the gripper so just keep it open for simplicity
        self.do_simulation(ctrl, self.frame_skip)

        # the gripper values don't go from 0 to 1, so we are normalizing it for easier learning and usage
        self.gripper_state = float(self.data.qpos[self._gripper_qpos_idx]) / 0.9

        # Cache useful sim state for info/logging
        self.elbow_z = self.data.body("UR10E_forearm_link").xpos[2]
        self.wrist2_rot = self.data.body("UR10E_wrist_2_link").xmat.copy().flatten()
        self.shoulder_rot = self.data.body("UR10E_shoulder_link").xmat.copy().flatten()
        self.object0_pos = self.data.site("object0").xpos.copy()
        self.object0_rot = self.data.site("object0").xmat.copy().flatten()

        observation = self._get_obs()

        #how rotated downwards the TCP is, we are trying to make it point directly downwards
        flange_rot = self.data.site("UR10E_TCP").xmat.copy().reshape(3, 3)
        flange_z = flange_rot[:, 2]
        z_dot = float(np.dot(flange_z, np.array([0.0, 0.0, -1.0])))

        distance = float(np.linalg.norm(
            observation["achieved_goal"] - observation["desired_goal"]
        ))
        #we can pass information about the current state so we can use them indirectly in the reward function and elsewere
        info = {
            "elbow_z": self.elbow_z,
            "wrist2_rot": self.wrist2_rot,
            "shoulder_rot": self.shoulder_rot,
            "object0_pos": self.object0_pos,
            "object0_rot": self.object0_rot,
            "gripper_state": self.gripper_state,
            "z_dot": z_dot,
            "prev_distance": self.prev_distance,
            "distance_to_target": distance,
            "is_success": False,
        }

        #lets retrieve all the observations and info we require for checking episode success
        reward, reward_components = self._compute_reward_components(
            observation["achieved_goal"],
            observation["desired_goal"],
            info,
        )

        # the rest of the info is complete only after compute reward
        info["reward_components"] = reward_components

        orientation_ok = True
        if self.require_downward_tcp:
            orientation_ok = z_dot > self.success_z_dot

        terminated = False
        # make shure that the requirements are satisfactory for completion and marking episode as success
        if distance < self.success_distance and orientation_ok:
            self.episode_success = True
            terminated = True
            reward += 10.0
            if self.no_early_termination:
                terminated = False

        # Now that compute_reward has run, update is_success in info
        info["is_success"] = bool(self.episode_success)

        truncated = False

        #for better debugging and viewing, this will change the color of the target depending on the state of the simulation
        if self.render_mode == "human":
            target_geom_id = self.model.geom("target_geom").id
            if self.episode_success:
                self.model.geom_rgba[target_geom_id] = [0, 0, 1, 1]  # blue = success
            else:
                self.model.geom_rgba[target_geom_id] = [1, 0, 0, 1]  # red = reach target
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

        # prevents errors, dont touch
        if reward.shape == (1,):
            reward = reward[0]

        return reward, components
    #since gym 0.26, the step function should return (obs, reward, terminated, truncated, info) instead of (obs, reward, done, info) to distinguish between episode termination and truncation due to time limits or other factors.
    #I should use terminated to indicate if the episode ended due to success or failure, and truncated to indicate if it ended due to time limits or other factors. In this case, I will set terminated to False for now and rely on the TimeLimit wrapper to handle episode truncation after a certain number of steps.

    def reset_model(self):
        #resetting settings so that next episode has a clean slate, never carry over from previous episodes unless you want bad learning
        self.episode_success = False
        self.prev_distance = 0.0

        key_id = self.model.key("UR10E_home").id#settings so the robot resets to its home position instead of its 0 position
                                                #you can change it to some other pose if you know how to
        qpos = self.model.key_qpos[key_id].copy()
        qvel = self.model.key_qvel[key_id].copy()
        ctrl = self.model.key_ctrl[key_id].copy()

        self.gripper_state = 0.0

        # Random target around the robot
        # This creates a donut-shaped reachable area.
        while True:
            xy = self.np_random.uniform(low=-0.8, high=0.8, size=2)
            z = self.np_random.uniform(low=0.2, high=0.7, size=1)
            self.goal = np.concatenate([xy, z])

            dist_xy = np.linalg.norm(self.goal[:2])
            if 0.7 < dist_xy < 1.0:
                break

        yaw = self.np_random.uniform(0, 2 * np.pi)
        half = yaw / 2.0
        goal_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

        self.model.body("target").pos[:] = self.goal
        self.model.body("target").quat[:] = goal_quat

        # The object is not used in this reach-only task
        object0_id = self.model.body("object0").id
        object0_jnt_adr = self.model.body_jntadr[object0_id]
        object_pos = np.array([0.0, 0.55, 0.025], dtype=np.float64)
        object_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        qpos[object0_jnt_adr:object0_jnt_adr + 3] = object_pos
        qpos[object0_jnt_adr + 3:object0_jnt_adr + 7] = object_quat
        qvel[object0_jnt_adr:object0_jnt_adr + 6] = 0.0

        self.set_state(qpos, qvel)
        self.data.ctrl[:] = ctrl

        # Recompute body/site positions after manually changing qpos/model target.
        mujoco.mj_forward(self.model, self.data)

        tcp_pos = self.data.site("UR10E_TCP").xpos.copy()
        target_pos = self.data.body("target").xpos.copy()
        self.prev_distance = float(np.linalg.norm(tcp_pos - target_pos))

        return self._get_obs()

    #getting the observation, nothing special
    def _get_obs(self):
        position = self.data.qpos[:6].flatten()#joint angles
        velocity = self.data.qvel[:6].flatten()#joint velocities

        flange_pos = self.data.site("UR10E_TCP").xpos.copy()
        flange_rot = self.data.site("UR10E_TCP").xmat.copy()
        flange_z_axis = flange_rot.reshape(3, 3)[:, 2]

        target_pos = self.data.body("target").xpos.copy()

        object0_pos = self.data.site("object0").xpos.copy()
        object0_xmat = self.data.site("object0").xmat.copy().flatten()

        return {
            "observation": np.concatenate([
                np.cos(position),# Joint angles are encoded as cos/sin to avoid discontinuities
                np.sin(position),# around ±π.
                velocity,
                flange_z_axis,
                object0_pos,
                object0_xmat,
            ]).astype(np.float64),
            "achieved_goal": flange_pos.astype(np.float64),
            "desired_goal": target_pos.astype(np.float64),
        }


# Register custom environment automatically on import.
register(
    id="UR10E-reach-v0",
    entry_point="ur10e_reach:ur10eEnv",
    max_episode_steps=1000,#everything was tested with 1000 steps/ about 10 seconds, anything less or more is bad for learning
)
