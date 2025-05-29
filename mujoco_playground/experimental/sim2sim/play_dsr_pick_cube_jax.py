from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import jax
from brax.training.agents.ppo import networks as ppo_networks
from orbax.checkpoint import PyTreeCheckpointer
from mujoco_playground import registry, manipulation
from mujoco_playground.config import manipulation_params
import functools

ENV_NAME      = "DSRPickCube"
CKPT_STEP_DIR = epath.Path(
    "/home/theochoi/Playground/mujoco_playground/learning/"
    "logs/DSRPickCube-20250512-081601/checkpoints/8519680/"
)

def build_policy():
  # 1) build network
  cfg      = manipulation_params.brax_ppo_config(ENV_NAME)
  env      = registry.load(ENV_NAME, config=manipulation.get_default_config(ENV_NAME))
  obs_d    = env.observation_size
  print("obs_d =", obs_d)
  act_d    = env.action_size
  print("act_d =", act_d)
  net_fac  = functools.partial(ppo_networks.make_ppo_networks, **cfg.network_factory)
  ppo_net  = net_fac(obs_d, act_d)
  make_inf = ppo_networks.make_inference_fn(ppo_net)

  # 2) restore orbax
  ckpt     = PyTreeCheckpointer().restore(CKPT_STEP_DIR.as_posix())
  ts       = ckpt[0] if isinstance(ckpt, (list, tuple)) else ckpt
  raw      = getattr(ts, "params", ts)

  # 3) massage into (normalizer, policy) tuple
  if isinstance(raw, dict):
    # if you saved under {"normalizer_params":…, "policy_params":…}
    if "normalizer_params" in raw and "policy_params" in raw:
      params = (raw["normalizer_params"], raw["policy_params"])
    # or maybe nested under raw["params"]?
    elif "params" in raw:
      p = raw["params"]
      params = (p[0], p[1]) if isinstance(p, (list,tuple)) else p
    else:
      raise ValueError(f"Unrecognized param dict keys: {list(raw)}")
  elif isinstance(raw, (list,tuple)) and len(raw)==2:
    params = tuple(raw)
  else:
    raise ValueError(f"Can't decode params from {raw!r}")

  # 4) jit the inference function
  inf_fn  = make_inf(params, deterministic=True)   # (obs, key)->(action,_)
  inf_jit = jax.jit(lambda o, k: inf_fn(o, k)[0])
  return inf_jit, obs_d, act_d

_inf_jit, _obs_d, _act_d = build_policy()
_key = jax.random.PRNGKey(0)

# MuJoCo control callback
def control_cb(model: mujoco.MjModel, data: mujoco.MjData):
  global _key
  # simple obs: qpos|qvel
  qpos = np.array(data.qpos[:_obs_d//2], dtype=np.float32)
  qvel = np.array(data.qvel[:_obs_d//2], dtype=np.float32)
  obs  = np.concatenate([qpos, qvel])[None,:]
  _key, sub = jax.random.split(_key)
  act = np.array(_inf_jit(obs, sub)[0], dtype=np.float32)
  data.ctrl[:] = act
  # clamp 
  lo, hi = model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1]
  np.clip(data.ctrl, lo, hi, out=data.ctrl)

def load_cb(model=None, data=None):
  mujoco.set_mjcb_control(control_cb)
  xml   = epath.Path(registry.get_default_config(ENV_NAME).xml_path).read_text()
  m     = mujoco.MjModel.from_xml_string(xml, assets=manipulation.get_assets())
  d     = mujoco.MjData(m)
  return m, d

if __name__=="__main__":
  viewer.launch(loader=load_cb)