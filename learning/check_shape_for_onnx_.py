import jax
from mujoco_playground import registry

# 1) env 인스턴스 생성
env = registry.load("DSRPickCube")
# env = registry.load("PandaPickCube")

# 2) reset 후 state.obs 에서 차원 확인
state = env.reset(jax.random.PRNGKey(0))
print("obs_dim =", state.obs.shape[-1])

# 3) action_dim 은 env.action_size 로 확인
#    (없으면 state.data.ctrl.shape[-1] 로도 확인 가능)
print("action_dim =", env.action_size)