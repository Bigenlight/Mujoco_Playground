import os
import functools
import json

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import flax.linen as nn
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import orbax.checkpoint as ocp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics 
import tree

from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

ENV_NAME = "PandaPickCube"
CHECKPOINT_DIR = '/home/theochoi/Playground/mujoco_playground/learning/logs/DSRPickCube-20250508-154014/checkpoints/40304640'

OUTPUT_ONNX_PATH = '/home/theochoi/Playground/mujoco_playground/learning/dsr_pickcube_model.onnx'
def convert_panda_checkpoint_to_onnx(env_name: str, checkpoint_dir: str, output_onnx_path: str):
    if checkpoint_dir == "PATH_TO_YOUR_CHECKPOINT_DIRECTORY_HERE":
        print("ERROR: Please update CHECKPOINT_DIR in the script..."); return

    resolved_output_path = os.path.abspath(output_onnx_path) if not os.path.isabs(output_onnx_path) and output_onnx_path != "panda_pick_cube_policy.onnx" else os.path.abspath(os.path.join(os.getcwd(), output_onnx_path) if output_onnx_path == "panda_pick_cube_policy.onnx" else output_onnx_path)
    if output_onnx_path == "panda_pick_cube_policy.onnx" and resolved_output_path != os.path.join(os.getcwd(), output_onnx_path):
        print(f"INFO: Defaulting ONNX output to current directory: {os.path.join(os.getcwd(), output_onnx_path)}")
        resolved_output_path = os.path.join(os.getcwd(), output_onnx_path)


    print(f"Starting conversion for environment: {env_name}")
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"Output ONNX will be saved to: {resolved_output_path}")

    base_log_dir = os.path.dirname(checkpoint_dir)
    config_json_path = os.path.join(base_log_dir, "config.json")
    env_train_cfg_dict = {}
    try:
        with open(config_json_path, 'r') as f: env_train_cfg_dict = json.load(f)
        print(f"Loaded environment training config from: {config_json_path}")
    except FileNotFoundError:
        print(f"WARNING: Could not find training config.json at {config_json_path}.")
    
    env_config = registry.get_default_config(env_name)
    if env_train_cfg_dict: env_config.update(env_train_cfg_dict)
    
    ppo_train_params = manipulation_params.brax_ppo_config(env_name)
    print(f"PPO training parameters loaded for {env_name}.")
    env = registry.load(env_name, config=env_config)
    
    policy_obs_key_from_config = ppo_train_params.network_factory.policy_obs_key
    policy_input_obs_size_val_intermediate = None
    if isinstance(env.observation_size, dict):
        if policy_obs_key_from_config not in env.observation_size:
            raise ValueError(f"Key '{policy_obs_key_from_config}' not in obs_size dict.")
        policy_input_obs_size_val_intermediate = env.observation_size[policy_obs_key_from_config]
    elif isinstance(env.observation_size, int):
        print(f"INFO: env.observation_size is int ({env.observation_size}). Using for key '{policy_obs_key_from_config}'.")
        policy_input_obs_size_val_intermediate = env.observation_size
    else:
        raise ValueError(f"Unsupported env.observation_size type: {type(env.observation_size)}")

    if isinstance(policy_input_obs_size_val_intermediate, int):
        policy_input_obs_size = policy_input_obs_size_val_intermediate
    elif isinstance(policy_input_obs_size_val_intermediate, (tuple, jp.ndarray)):
        current_shape = policy_input_obs_size_val_intermediate.shape if isinstance(policy_input_obs_size_val_intermediate, jp.ndarray) else policy_input_obs_size_val_intermediate
        if not all(isinstance(dim, int) for dim in current_shape): raise ValueError("Shape has non-int dims.")
        policy_input_obs_size = functools.reduce(lambda x, y: x * y, current_shape) if current_shape else 1
        if policy_input_obs_size != policy_input_obs_size_val_intermediate or \
            (hasattr(current_shape, '__len__') and len(current_shape) > 1): # Check if it was a multi-element tuple/array
             print(f"Derived/Flattened policy_input_obs_size from {policy_input_obs_size_val_intermediate} (shape {current_shape}) to {policy_input_obs_size}")
    else:
        raise ValueError(f"Unsupported intermediate obs size type: {type(policy_input_obs_size_val_intermediate)}")

    act_size = env.action_size
    print(f"Policy input observation size (flat state): {policy_input_obs_size}")
    print(f"Action size: {act_size}")
    activation_fn_flax = getattr(ppo_train_params.network_factory, 'activation', nn.swish)
    print(f"JAX network conceptual activation: {activation_fn_flax}.")

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        loaded_training_params_raw = checkpointer.restore(checkpoint_dir)
    except Exception as e:
        print(f"ERROR: Failed to load Orbax checkpoint. Error: {e}"); return
    
    print(f"Checkpoint loaded. Raw object type: {type(loaded_training_params_raw)}")

    jax_policy_params_dict = None # This will hold the dict with 'hidden_0', 'hidden_1', etc.
    jax_normalizer_params = None

    if isinstance(loaded_training_params_raw, (list, tuple)) and len(loaded_training_params_raw) >= 2: # Expect at least 2 for normalizer & policy container
        print(f"  Raw loaded object is {type(loaded_training_params_raw).__name__} of length {len(loaded_training_params_raw)}.")
        
        # Detailed print for relevant elements
        for i in range(min(len(loaded_training_params_raw), 3)): # Print first few elements
            element = loaded_training_params_raw[i]
            print(f"  Inspecting Element {i}: Type: {type(element)}")
            if isinstance(element, dict):
                print(f"    Element {i} Keys: {list(element.keys())}")
                if 'params' in element and isinstance(element.get('params'), dict):
                    print(f"      Element {i}['params'] is a dict with Keys: {list(element['params'].keys())}")

        # Assign normalizer_params (likely Element 0)
        el0 = loaded_training_params_raw[0]
        if isinstance(el0, dict) and {'count', 'mean', 'std', 'summed_variance'}.issubset(el0.keys()):
            jax_normalizer_params = el0
            print(f"  SUCCESS: Assigned Element 0 as jax_normalizer_params.")
        else:
            print(f"  WARN: Element 0 does not fit expected normalizer_params structure based on keys.")

        # Assign policy_params (likely from Element 1['params'])
        el1 = loaded_training_params_raw[1]
        if isinstance(el1, dict) and 'params' in el1 and isinstance(el1['params'], dict):
            # Check if the sub-dict looks like network layers
            if any(k.startswith('hidden_') for k in el1['params'].keys()):
                 jax_policy_params_dict = el1['params']
                 print(f"  SUCCESS: Assigned Element 1['params'] as jax_policy_params_dict.")
            else:
                print(f"  WARN: Element 1['params'] does not contain 'hidden_X' keys. Sub-keys: {list(el1['params'].keys())}")
        else:
            print(f"  WARN: Element 1 does not have ['params'] dict structure for policy.")
            # Simple fallback: if el1 itself is the dict of layers (less likely for Brax state)
            if isinstance(el1, dict) and any(k.startswith('hidden_') for k in el1.keys()):
                jax_policy_params_dict = el1
                print(f"  SUCCESS: Assigned Element 1 directly as jax_policy_params_dict (fallback).")


    elif isinstance(loaded_training_params_raw, dict): # If the raw loaded object itself is a dict (e.g. from older checkpoint version)
        print(f"  Raw loaded object is a dict. Keys: {list(loaded_training_params_raw.keys())}")
        if 'policy_params' in loaded_training_params_raw and 'normalizer_params' in loaded_training_params_raw:
            # This assumes 'policy_params' contains the 'hidden_X' layer dict
            potential_policy_params = loaded_training_params_raw['policy_params']
            if isinstance(potential_policy_params, dict) and any(k.startswith('hidden_') for k in potential_policy_params.keys()):
                 jax_policy_params_dict = potential_policy_params
            elif isinstance(potential_policy_params, dict) and 'params' in potential_policy_params and \
                 isinstance(potential_policy_params['params'], dict) and \
                 any(k.startswith('hidden_') for k in potential_policy_params['params'].keys()):
                 jax_policy_params_dict = potential_policy_params['params'] # Nested like training_state.params.policy
            
            jax_normalizer_params = loaded_training_params_raw['normalizer_params']
            print("  Assigned params from single top-level dictionary (old assumption structure).")
    
    if jax_policy_params_dict is None or jax_normalizer_params is None:
        print("ERROR: Failed to extract jax_policy_params_dict AND/OR jax_normalizer_params.")
        print(f"  jax_policy_params_dict found: {jax_policy_params_dict is not None}")
        print(f"  jax_normalizer_params found: {jax_normalizer_params is not None}")
        return
    
    print("Successfully assigned jax_policy_params_dict and jax_normalizer_params.")

    obs_key_for_norm = ppo_train_params.network_factory.policy_obs_key
    mean_val, std_val = None, None
    try:
        if 'running_mean_std' in jax_normalizer_params and \
           isinstance(jax_normalizer_params['running_mean_std'], dict) and \
           obs_key_for_norm in jax_normalizer_params['running_mean_std'].get('mean', {}) and \
           obs_key_for_norm in jax_normalizer_params['running_mean_std'].get('std', {}):
            mean_val = jax_normalizer_params['running_mean_std']['mean'][obs_key_for_norm]
            std_val = jax_normalizer_params['running_mean_std']['std'][obs_key_for_norm]
        elif obs_key_for_norm in jax_normalizer_params.get('mean', {}) and \
             obs_key_for_norm in jax_normalizer_params.get('std', {}):
            mean_val = jax_normalizer_params['mean'][obs_key_for_norm]
            std_val = jax_normalizer_params['std'][obs_key_for_norm]
        elif 'mean' in jax_normalizer_params and 'std' in jax_normalizer_params and \
             not isinstance(jax_normalizer_params['mean'], dict): 
            mean_val = jax_normalizer_params['mean']
            std_val = jax_normalizer_params['std']
        else:
            raise KeyError("Norm mean/std not in expected structures.")
        print(f"Extracted norm mean (shape {getattr(mean_val, 'shape', 'N/A')}), std (shape {getattr(std_val, 'shape', 'N/A')}) for key '{obs_key_for_norm}'.")
    except KeyError as e:
        print(f"WARN: Could not extract norm mean/std. Error: {e}. Using identity.")
        mean_val = jp.zeros(policy_input_obs_size, dtype=jp.float32)
        std_val = jp.ones(policy_input_obs_size, dtype=jp.float32)

    mean_tf = tf.convert_to_tensor(jp.asarray(mean_val).flatten())
    std_tf = tf.convert_to_tensor(jp.asarray(std_val).flatten())
    if mean_tf.shape[0] != policy_input_obs_size:
        print(f"WARN: Norm mean shape {mean_tf.shape} != obs_size {policy_input_obs_size}. Using identity.")
        mean_tf = tf.zeros(policy_input_obs_size, dtype=tf.float32)
        std_tf = tf.ones(policy_input_obs_size, dtype=tf.float32)
    mean_std_for_tf = (mean_tf, std_tf)

    if activation_fn_flax == nn.swish: tf_activation_fn = tf.nn.swish
    elif activation_fn_flax == nn.relu: tf_activation_fn = tf.nn.relu
    else: tf_activation_fn = tf.nn.swish; print(f"WARN: Defaulting JAX activation {activation_fn_flax} to swish.")

    # Dynamically determine TF MLP layer sizes from the checkpoint's policy parameters
    # jax_policy_params_dict has keys like 'hidden_0', 'hidden_1', ...
    num_jax_layers = 0
    temp_tf_mlp_layer_sizes = []
    
    sorted_layer_keys = sorted([k for k in jax_policy_params_dict.keys() if k.startswith('hidden_')])
    num_jax_layers = len(sorted_layer_keys)

    if num_jax_layers == 0:
        print(f"ERROR: No 'hidden_X' layers found in jax_policy_params_dict. Keys: {jax_policy_params_dict.keys()}"); return

    for i in range(num_jax_layers):
        layer_name = f"hidden_{i}"
        if layer_name not in jax_policy_params_dict:
            print(f"ERROR: Expected layer {layer_name} not found in jax_policy_params_dict."); return
        
        # Output size of a Dense layer is kernel.shape[1]
        layer_output_size = jax_policy_params_dict[layer_name]['kernel'].shape[1]
        temp_tf_mlp_layer_sizes.append(layer_output_size)
    
    # The last size in temp_tf_mlp_layer_sizes is the output of the final dense layer (action_size * 2)
    # The preceding ones are the hidden layer output sizes.
    tf_mlp_layer_sizes_from_checkpoint = temp_tf_mlp_layer_sizes
    print(f"Dynamically determined TF MLP layer sizes from checkpoint: {tf_mlp_layer_sizes_from_checkpoint}")


    class TFMLP(tf.keras.Model):
        def __init__(self, layer_sizes_from_ckpt, activation_fn, mean_std_tuple_for_norm): # Updated signature
            super().__init__()
            self._mean_norm, self._std_norm = None, None
            if mean_std_tuple_for_norm is not None:
                self._mean_norm = tf.Variable(mean_std_tuple_for_norm[0], trainable=False, name="obs_mean")
                self._std_norm = tf.Variable(tf.maximum(mean_std_tuple_for_norm[1], 1e-8), trainable=False, name="obs_std")
            
            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            # Create layers based on sizes derived from checkpoint
            # The last size in layer_sizes_from_ckpt is the final output before split (action_size * 2)
            # All layers before that are hidden layers with activation_fn
            for i, size in enumerate(layer_sizes_from_ckpt[:-1]): # Hidden layers
                self.mlp_block.add(layers.Dense(size, activation=activation_fn, name=f"hidden_{i}"))
            # Output layer (no activation here, split and tanh happen after)
            self.mlp_block.add(layers.Dense(layer_sizes_from_ckpt[-1], name=f"hidden_{len(layer_sizes_from_ckpt)-1}"))

        def call(self, inputs):
            inputs = tf.cast(inputs[0] if isinstance(inputs, list) else inputs, tf.float32)
            norm_inputs = (inputs - self._mean_norm) / self._std_norm if self._mean_norm is not None else inputs
            raw_mlp_output = self.mlp_block(norm_inputs)
            # Ensure raw_mlp_output has a known last dimension for split
            if raw_mlp_output.shape[-1] is None or raw_mlp_output.shape[-1] % 2 != 0:
                 raise ValueError(f"Output of MLP block has unexpected shape for splitting: {raw_mlp_output.shape}")
            loc, _ = tf.split(raw_mlp_output, 2, axis=-1)
            return tf.tanh(loc)

    tf_policy_network = TFMLP(tf_mlp_layer_sizes_from_checkpoint, tf_activation_fn, mean_std_for_tf)
    example_input_shape = (1, policy_input_obs_size)
    example_input_tf = tf.zeros(example_input_shape, dtype=tf.float32)
    _ = tf_policy_network(example_input_tf) # Build the model
    print(f"TF model built dynamically. Input shape: {example_input_shape}")

    # 6. Weight Transfer - using jax_policy_params_dict directly for layers
    print(f"Starting weight transfer using jax_policy_params_dict with keys: {list(jax_policy_params_dict.keys())}")
    for idx, tf_layer in enumerate(tf_policy_network.mlp_block.layers):
        if isinstance(tf_layer, tf.keras.layers.Dense):
            jax_layer_name = f"hidden_{idx}" # Matches how layers are named in Brax MLP
            if jax_layer_name in jax_policy_params_dict:
                jax_p_layer = jax_policy_params_dict[jax_layer_name]
                kernel_jax = jp.asarray(jax_p_layer['kernel'])
                bias_jax = jp.asarray(jax_p_layer['bias'])
                
                if kernel_jax.shape != tf_layer.kernel.shape or bias_jax.shape != tf_layer.bias.shape:
                    print(f"WARN: Shape Mismatch for layer '{tf_layer.name}' (JAX '{jax_layer_name}')!")
                    print(f"  JAX kernel: {kernel_jax.shape}, bias: {bias_jax.shape}")
                    print(f"  TF  kernel: {tf_layer.kernel.shape}, bias: {tf_layer.bias.shape}. Skipping transfer.")
                    continue
                
                tf_layer.set_weights([kernel_jax, bias_jax])
                print(f"Transferred weights for TF layer '{tf_layer.name}' from JAX layer '{jax_layer_name}'.")
            else:
                print(f"WARN: JAX params for layer '{jax_layer_name}' not found in jax_policy_params_dict. TF layer '{tf_layer.name}' weights random.")
    print("Weight transfer done.")

    input_signature_onnx = [tf.TensorSpec(shape=example_input_shape, dtype=tf.float32, name="obs")]
    tf_policy_network.output_names = ['continuous_actions']
    try:
        model_proto, _ = tf2onnx.convert.from_keras(tf_policy_network, input_signature_onnx, opset=11, output_path=resolved_output_path)
        print(f"Model converted to ONNX: {resolved_output_path}")
    except Exception as e:
        print(f"ERROR: Failed to convert TF to ONNX. Error: {e}"); import traceback; traceback.print_exc(); return

    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(resolved_output_path, providers=['CPUExecutionProvider'])
        out_name = sess.get_outputs()[0].name
        pred = sess.run([out_name], {sess.get_inputs()[0].name: example_input_tf.numpy()})[0]
        print(f"ONNX model verified. Test output shape: {pred.shape}")
    except Exception as e:
        print(f"Error during ONNX verification: {e}")
    print("Conversion script finished.")

if __name__ == "__main__":
    if CHECKPOINT_DIR == "PATH_TO_YOUR_CHECKPOINT_DIRECTORY_HERE":
        print("ERROR: Please update CHECKPOINT_DIR and OUTPUT_ONNX_PATH in the script.")
    else:
        if not os.path.isabs(CHECKPOINT_DIR): CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)
        convert_panda_checkpoint_to_onnx(ENV_NAME, CHECKPOINT_DIR, OUTPUT_ONNX_PATH)