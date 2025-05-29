import os
import time # For controlling loop frequency
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

# Mujoco Playground imports - For utility functions if still needed
from mujoco_playground._src.mjx_env import get_qpos_ids, get_qvel_ids

# --- Configuration ---
ONNX_POLICY_PATH = "/home/theochoi/Playground/mujoco_playground/learning/my_panda_model.onnx"
PANDA_PICK_CUBE_XML_PATH = "/home/theochoi/Playground/mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda/xmls/mjx_single_cube.xml"

# --- Panda Constants (Manually defined based on mujoco_playground/panda.py) ---
PANDA_JOINT_NAMES = tuple(f'joint{i}' for i in range(1, 8)) # ('joint1', ..., 'joint7')
GRIPPER_JOINT_NAMES = tuple(f'finger_joint{i}' for i in range(1, 3)) # ('finger_joint1', 'finger_joint2')

# ALL_PANDA_JOINTS provided by you (matches standard naming if '' prefix is omitted in XML for joints)
# Using your definition:
ALL_PANDA_JOINTS_QPOS = [ # For qpos/qvel IDs, ensure these names match the <joint name="..."> in your XML
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2"
]
# Actuator names based on mjx_single_cube.xml structure (7 arm + 1 gripper)
# The XML uses "joint{i}" for arm actuators and "gripper" for the gripper actuator.
PANDA_ARM_ACTUATOR_NAMES = tuple(f'joint{i}' for i in range(1, 8))
PANDA_GRIPPER_ACTUATOR_NAME = "gripper" # Check your XML for this name, it controls finger_joint1
ALL_ACTUATOR_NAMES = PANDA_ARM_ACTUATOR_NAMES + (PANDA_GRIPPER_ACTUATOR_NAME,)


# Simulation and Control Parameters
ENV_OBS_DIM = 66
ENV_ACTION_DIM = 8 # 7 arm + 1 gripper
SIM_DT = 0.005
CTRL_DT = 0.02
N_SUBSTEPS = int(round(CTRL_DT / SIM_DT)) if SIM_DT > 0 else 1
ACTION_SCALE = 0.04

class PandaOnnxController:
    def __init__(
        self,
        policy_path: str,
        model: mujoco.MjModel,
    ):
        print(f"Loading ONNX policy from: {policy_path}")
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self._output_names = [o.name for o in self._policy.get_outputs()]
        self._input_name = self._policy.get_inputs()[0].name
        print(f"  ONNX Model Input: '{self._input_name}', Outputs: {self._output_names}")

        self._model = model
        self._data = None 

        # Get qpos and qvel IDs for the Panda (using your provided joint names)
        self._robot_qpos_ids = get_qpos_ids(model, ALL_PANDA_JOINTS_QPOS)
        self._robot_qvel_ids = get_qvel_ids(model, ALL_PANDA_JOINTS_QPOS)
        if not self._robot_qpos_ids or len(self._robot_qpos_ids) != 9:
            print(f"WARNING: Problem getting all 9 qpos IDs. Found: {self._robot_qpos_ids}. Check ALL_PANDA_JOINTS_QPOS against XML.")
        if not self._robot_qvel_ids or len(self._robot_qvel_ids) != 9:
            print(f"WARNING: Problem getting all 9 qvel IDs. Found: {self._robot_qvel_ids}. Check ALL_PANDA_JOINTS_QPOS against XML.")
        
        print(f"  Robot QPos IDs ({len(self._robot_qpos_ids)}): {self._robot_qpos_ids}")
        print(f"  Robot QVel IDs ({len(self._robot_qvel_ids)}): {self._robot_qvel_ids}")
        
        self._sensor_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in [
                "gripper_width", "gripper_vel", "tcp_pos", "obj_pos", 
                "obj_quat", "target_pos", "target_quat"
            ]
        }
        print(f"  Sensor IDs: {self._sensor_ids}")
        for name, id_val in self._sensor_ids.items():
            if id_val == -1:
                print(f"    WARNING: Sensor '{name}' not found in the model! Observations will be incorrect.")
        
        # Verify actuator count
        if model.nu != ENV_ACTION_DIM:
            print(f"  WARNING: Model actuator count ({model.nu}) does not match ENV_ACTION_DIM ({ENV_ACTION_DIM}). Control will be misaligned.")
        print(f"  Model has {model.nu} actuators.")


        self._action_scale = ACTION_SCALE
        self._last_action_scaled = np.zeros(ENV_ACTION_DIM, dtype=np.float32) # Store scaled action if needed by obs
        self._last_action_raw = np.zeros(ENV_ACTION_DIM, dtype=np.float32) # Store raw [-1,1] action

        self._time_since_last_control = 0.0

    def set_data(self, data: mujoco.MjData):
        self._data = data

    def get_obs(self) -> np.ndarray:
        if self._data is None:
            raise ValueError("MjData object not set in controller.")
        data = self._data

        # --- Construct the 66-dimensional observation vector ---
        # This MUST match the training observation structure precisely.
        
        robot_qpos = data.qpos[self._robot_qpos_ids] if self._robot_qpos_ids else np.zeros(9)
        robot_qvel = data.qvel[self._robot_qvel_ids] if self._robot_qvel_ids else np.zeros(9)
        
        gripper_width_val = data.sensordata[self._sensor_ids["gripper_width"]] if self._sensor_ids["gripper_width"]!=-1 else 0.0
        gripper_vel_val = data.sensordata[self._sensor_ids["gripper_vel"]] if self._sensor_ids["gripper_vel"]!=-1 else 0.0
        
        tcp_pos_val = data.sensordata[self._sensor_ids["tcp_pos"]] if self._sensor_ids["tcp_pos"]!=-1 else np.zeros(3)
        obj_pos_val = data.sensordata[self._sensor_ids["obj_pos"]] if self._sensor_ids["obj_pos"]!=-1 else np.zeros(3)
        obj_quat_val = data.sensordata[self._sensor_ids["obj_quat"]] if self._sensor_ids["obj_quat"]!=-1 else np.array([1.0,0,0,0])
        
        target_pos_val = data.sensordata[self._sensor_ids["target_pos"]] if self._sensor_ids["target_pos"]!=-1 else np.zeros(3)
        target_quat_val = data.sensordata[self._sensor_ids["target_quat"]] if self._sensor_ids["target_quat"]!=-1 else np.array([1.0,0,0,0])

        tcp_to_obj_pos_vec = tcp_pos_val - obj_pos_val
        obj_to_target_pos_err_vec = obj_pos_val - target_pos_val
        
        q_diff = np.empty(4)
        q_obj_conj = np.empty(4)
        mujoco.mju_negQuat(q_obj_conj, obj_quat_val)
        mujoco.mju_mulQuat(q_diff, target_quat_val, q_obj_conj)
        obj_to_target_quat_err_3d = q_diff[1:] * np.sign(q_diff[0]) if q_diff[0] != 0 else q_diff[1:]


        obs_list = [
            robot_qpos.flatten(),                 # 9
            robot_qvel.flatten(),                 # 9
            np.array([gripper_width_val]),        # 1
            np.array([gripper_vel_val]),          # 1
            tcp_to_obj_pos_vec.flatten(),         # 3
            obj_pos_val.flatten(),                # 3
            obj_quat_val.flatten(),               # 4
            obj_to_target_pos_err_vec.flatten(),  # 3
            obj_to_target_quat_err_3d.flatten(),  # 3
        ]
        
        # Current total: 9+9+1+1+3+3+4+3+3 = 36
        # Needed: 66. Remaining: 30.
        
        # === Placeholder for the REMAINING 30 dimensions ===
        # YOU MUST IDENTIFY THESE AND THEIR EXACT ORDER from your
        # training environment's observation function (`PandaPickCube.observation`).
        # 
        # Common candidates:
        # 1. Previous action (self._last_action_raw or self._last_action_scaled): ENV_ACTION_DIM (8 dimensions)
        #    obs_list.append(self._last_action_raw.flatten())
        #
        # If previous action is included (36 + 8 = 44), 22 dimensions remain.
        # Other possibilities for the remaining 22:
        # - Object linear velocity (3)
        # - Object angular velocity (3)
        # - Target qpos for robot arm (if part of obs, typically 7 for Panda arm)
        # - etc.
        
        # Example: Add previous raw action (output of tanh, before scaling)
        obs_list.append(self._last_action_raw.flatten()) # Adds 8 dimensions
        current_dims = 36 + ENV_ACTION_DIM
        
        # Padding the rest for now - REPLACE THIS
        padding_dims = ENV_OBS_DIM - current_dims
        if padding_dims > 0:
            # print(f"    DEBUG: Padding observation with {padding_dims} zeros. Update get_obs()!")
            obs_list.append(np.zeros(padding_dims))
        elif padding_dims < 0:
            current_obs_temp = np.concatenate(obs_list).astype(np.float32)
            raise ValueError(f"Constructed obs ({current_obs_temp.shape[0]} dims) exceeds expected {ENV_OBS_DIM} before padding.")

        final_obs = np.concatenate(obs_list).astype(np.float32)
        
        if final_obs.shape[0] != ENV_OBS_DIM:
            raise ValueError(f"Final observation dim {final_obs.shape[0]} != expected {ENV_OBS_DIM}")
            
        return final_obs.reshape(1, -1) 

    def get_control(self) -> np.ndarray | None:
        self._time_since_last_control += SIM_DT
        
        if self._time_since_last_control >= CTRL_DT - 1e-5: # Add tolerance
            self._time_since_last_control = self._time_since_last_control % CTRL_DT 
            obs = self.get_obs()
            
            onnx_input = {self._input_name: obs}
            action_raw = self._policy.run(self._output_names, onnx_input)[0][0] 
            self._last_action_raw = action_raw.copy() # Store raw action for potential use in next obs

            control_signal = action_raw * self._action_scale
            
            ctrl_range = self._model.actuator_ctrlrange.copy() # (nu, 2)
            # Ensure control_signal has same first dim as ctrl_range
            if control_signal.shape[0] == ctrl_range.shape[0]:
                 control_signal = np.clip(control_signal, ctrl_range[:, 0], ctrl_range[:, 1])
            else:
                 print(f"  WARNING: control_signal shape {control_signal.shape} mismatch with ctrl_range {ctrl_range.shape}. Clipping might be incorrect.")
                 # Fallback: clip element-wise assuming broadcast or take first ENV_ACTION_DIM if nu > ENV_ACTION_DIM
                 control_signal = np.clip(control_signal, ctrl_range[:ENV_ACTION_DIM, 0], ctrl_range[:ENV_ACTION_DIM, 1])


            self._last_action_scaled = control_signal.copy() # Store scaled action
            return control_signal
        return None

    def apply_control(self, ctrl_signal: np.ndarray):
        if self._data is None: return
        if ctrl_signal is not None:
            if len(ctrl_signal) == self._model.nu :
                self._data.ctrl[:] = ctrl_signal
            elif len(ctrl_signal) == ENV_ACTION_DIM and ENV_ACTION_DIM <= self._model.nu:
                 self._data.ctrl[:ENV_ACTION_DIM] = ctrl_signal
            else:
                 print(f"  WARNING: ctrl_signal length {len(ctrl_signal)} mismatch with model.nu {self._model.nu} and ENV_ACTION_DIM {ENV_ACTION_DIM}. Control not applied fully/correctly.")


_controller = None

def load_mujoco_model_and_data():
    global _controller
    print(f"Loading XML from: {PANDA_PICK_CUBE_XML_PATH}")
    if not os.path.exists(PANDA_PICK_CUBE_XML_PATH):
        raise FileNotFoundError(f"Panda XML not found at {PANDA_PICK_CUBE_XML_PATH}")
    
    model = mujoco.MjModel.from_xml_path(PANDA_PICK_CUBE_XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    print(f"Model loaded. nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensordata={model.nsensordata}")

    _controller = PandaOnnxController(ONNX_POLICY_PATH, model)
    
    mujoco.mj_resetData(model, data)
    # Example: Initialize cube and target to known positions for testing if needed
    # obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_cube")
    # target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
    # if obj_body_id != -1 :
    #     obj_joint_adr = model.body_jntadr[obj_body_id]
    #     obj_joint_num = model.body_jntnum[obj_body_id]
    #     if obj_joint_num == 7: # Free joint
    #         # data.qpos[obj_joint_adr : obj_joint_adr+7] = [0.0, 0.1, 0.05, 1,0,0,0] # x,y,z, w,x,y,z
    #         pass 
    # if target_site_id != -1:
    #     # model.site_pos[target_site_id] = [0.1, -0.1, 0.1] # tx,ty,tz
    #     pass
    # mujoco.mj_forward(model, data) # Recalculate sensors

    return model, data

def control_callback(model: mujoco.MjModel, data: mujoco.MjData):
    if _controller is None: return
    _controller.set_data(data) 
    ctrl_signal = _controller.get_control()
    if ctrl_signal is not None:
        _controller.apply_control(ctrl_signal)

def main():
    model, data = load_mujoco_model_and_data()
    if model is None or data is None: return

    mujoco.set_mjcb_control(control_callback)
    print("Launching MuJoCo viewer...")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            step_start = time.time()
            v.sync()
            time_until_next_step = SIM_DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        print("Viewer closed.")

if __name__ == "__main__":
    # Your paths are hardcoded now, so the initial check can be simplified or removed
    main()