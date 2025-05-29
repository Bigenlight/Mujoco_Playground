import functools
from etils import epath
import jax
import tensorflow as tf
import tf2onnx
from jax.experimental import jax2tf
from flax.training import checkpoints
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import registry, manipulation
from mujoco_playground.config import manipulation_params

from orbax.checkpoint import PyTreeCheckpointer

from etils import epath

import functools
import jax
from brax.training.agents.ppo import networks as ppo_networks



ENV_NAME = "DSRPickCube"
# Point at the parent “checkpoints” dir or a specific step:
CKPT_DIR = epath.Path(
  "/home/theochoi/Playground/mujoco_playground/learning/"
  "logs/DSRPickCube-20250512-081601/checkpoints/8519680/"
)


# — build the Flax PPO network and the inference function —
ppo_params = manipulation_params.brax_ppo_config(ENV_NAME)
env_cfg    = manipulation.get_default_config(ENV_NAME)
env        = registry.load(ENV_NAME, config=env_cfg)
obs_dim    = env.observation_size
act_dim    = env.action_size

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **ppo_params.network_factory
)
ppo_network = network_factory(obs_dim, act_dim)
# make_inf(params, deterministic=True) → (obs, key) -> (action, extras)
make_inf    = ppo_networks.make_inference_fn(ppo_network)

# 1) find the correct step_path
if CKPT_DIR.name.isdigit():
  # CKPT_DIR is already the numeric folder
  step_path = CKPT_DIR
else:
  # scan children for the latest numeric folder
  step_dirs = [d for d in CKPT_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
  assert step_dirs, f"No numeric step dirs found under {CKPT_DIR}"
  latest_step = max(int(d.name) for d in step_dirs)
  step_path   = CKPT_DIR / str(latest_step)

# 2) restore with Orbax
ckpt     = checkpointer.restore(step_path.as_posix())

# 3) extract params
restored = ckpt[0] if isinstance(ckpt, (list, tuple)) else ckpt

# get the raw params object
raw = getattr(restored, "params", restored)

# Brax PPO saved params as a dict with keys 
#    "normalizer_params" and "policy_params"
if isinstance(raw, dict):
  normalizer = raw["normalizer_params"]
  policy     = raw["policy_params"]
  params     = (normalizer, policy)
else:
  # already a (norm, policy) tuple
  params     = raw
# 4) jit inference
inf_fn   = make_inf(params, deterministic=True)  # (obs, key)->(action, _)
inf_jit  = jax.jit(lambda x: inf_fn(x, jax.random.PRNGKey(0))[0])

# 5) convert to TF with a fixed batch of 1
#    we pass polymorphic_shapes=["1,{}".format(obs_dim)] so no `None` slips in.
poly = [f"1,{obs_dim}"]
tf_converted = jax2tf.convert(inf_jit, with_gradient=False, polymorphic_shapes=poly)
tf_fun       = tf.function(
    tf_converted,
    input_signature=[tf.TensorSpec([1, obs_dim], tf.float32)],
)

# get a concrete function so tf2onnx won’t re‐infer shapes
concrete = tf_fun.get_concrete_function(tf.TensorSpec([1, obs_dim], tf.float32))

# 6) export to ONNX
onnx_path = epath.Path(__file__).parent / "onnx" / f"{ENV_NAME.lower()}.onnx"
tf2onnx.convert.from_function(
    concrete,
    input_signature=[tf.TensorSpec([1, obs_dim], tf.float32)],
    opset=13,
    output_path=onnx_path.as_posix(),
)
print("Wrote ONNX model to", onnx_path)