# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.doosan_robotics import dsr_avoid as dsr
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      # episode_length = 150,
      episode_length=250, 
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.3,
              # 추가
              collision_obstacle=0.25,  # 장애물과 충돌하지 않도록
          )
      ),
  )
  return config

# 박스를 주워 특정 위치로 옮겨 놓는 task
class DSRPickCubeAvoiding(dsr.PandaBase):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    
    # scene 파일 + panda 모델 파일, 계속 들어가면 있음
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "doosan_robotics"
        / "xmls"
        / "mjx_single_cube_obstacle.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="box", keyframe="home")
    self._sample_orientation = sample_orientation

  # 에피소드 시작시 초기화 작업, 하나의 에피소드가 한번의 pick-and-place 작업 시도를 의미
  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    # intialize box position
    box_pos = (
        jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.0]), # 박스 시작 위치 최소 # 아마 xyz
            maxval=jp.array([0.2, 0.1, 0.0]), # 최대
        )
        + self._init_obj_pos
    )

    # # initialize target position
    # target_pos = (
    #     jax.random.uniform(
    #         rng_target,
    #         (3,),
    #         minval=jp.array([-0.2, -0.2, 0.2]), # 박스 목표 위치 최소
    #         maxval=jp.array([0.2, 0.2, 0.4]), # 최대
    #     )
    #     + self._init_obj_pos
    # )

    # Y 축으로 평행 이동
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.2, 0.3, 0.0]), # 박스 목표 위치 최소
            maxval=jp.array([0.2, 0.6, 0.0]), # 최대
        )
        + self._init_obj_pos
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # PandaPickCubeOrientation일 경우에만 목표 orientation을 랜덤화
    if self._sample_orientation:
      # sample a random direction
      rng, rng_axis, rng_theta = jax.random.split(rng, 3)
      perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
      target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)


    # initialize data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    ) # mj안에 있는 박스 위치 덮어쓰기
    data = mjx_env.init(
        self._mjx_model,
        init_q,
        jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
    ) # 그것을 mjx_env.init에 넣어줌, 이를 통해서 시뮬레이션을 초기 상태를 바꿈

    # set target mocap position # 위에서 설정했던 목표가 이제 data에 들어감
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos), # 위에서 설정한 목표 위치
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),# 위에서 설정한 목표 orientation
    )

    # initialize env state and info
    # 강화학습 루프가 사용할 초기값.
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float), # 로봇이 영역을 벗어났는지
        **{k: 0.0 for k in self._config.reward_config.scales.keys()}, # 보상 스케일별 초기값
    }
    # 다음 스텝용: rng(JAX의 의사난수 생성을 위한 PRNG 키(PRNGKey)), 목표 위치 , 아직 박스를 위치를 잡지 못했다는 0.0
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    # 아래 줄 중요!
    # 위 info 만든 data로 벡터를 만듬 
    # 즉, 로봇의 관절 위치 및 속도, 박스의 위치와 방향, 목표까지 오차도 주어진 정보를 통해 전달, 그리고 이 정브들을 1열로 만듬. 
    # 이 정보가 네트워크, 신경망으로 들감.
    obs = self._get_obs(data, info)  # 아래에 자세히 정의 됨
    reward, done = jp.zeros(2) # 에피소드 시작시 보상과 종료 플래그를 0으로 초기화
    state = State(data, obs, reward, done, metrics, info) # 위 정보를 state로 반환
    return state 

  # 에피소드 중에 계속 반복 되는 스텝
  def step(self, state: State, action: jax.Array) -> State:
    # action은 네트워크에서 나온 값, 이걸로 로봇을 움직임
    delta = action * self._action_scale # action_scale은 config에 정의된 값
    # state.data.ctrl은 이전에 내려진 제어 명령어, 거기다 새로 움직일 delta를 더함, 결국 이 값이 로봇을 움직임.
    ctrl = state.data.ctrl + delta 
    ctrl = jp.clip(ctrl, self._lowers, self._uppers) # 안전 제어 범위 한정
    
    # 무조코 시뮬레이션 진행
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    # 보상 계산 # _get_reward은 아래에 정의
    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k] # 각 가중치 만큼 곱
        for k, v in raw_rewards.items() 
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    # 종료 조건 판단
    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) # 절대 값이 1을 넘는 축이 하나라도 있으면 True
    out_of_bounds |= box_pos[2] < 0.0 # OR # 한번이라도 범위를 벗어나면 True
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() # qpos나 qvel에 NaN이 있으면 종료
    # 위 모든 조건이 전부 OR임
    done = done.astype(float) # done은 state에 들어감

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    # 위에서 중요하다고 했던 현재 상태 관측 -> 네트워크로 들어가는 값
    obs = self._get_obs(data, state.info)

    # 새 state 객체 반환
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state
  
  # data와 state의 info를 기준으로 보상 계산
  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    # state.info에서 박스 목표 배치 위치를 가져옴
    target_pos = info["target_pos"]
    # 현재 박스의 위치를 가져옴
    box_pos = data.xpos[self._obj_body]
    # 그리퍼의 위치를 가져옴
    gripper_pos = data.site_xpos[self._gripper_site]

    # 목표 위치와 박스 현재 위치의 거리
    pos_err = jp.linalg.norm(target_pos - box_pos)
    # 목표 위치와 박스 현재 위치의 회전 오차 (아래 3줄)
    box_mat = data.xmat[self._obj_body] # 박스 회전 행렬
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    # 위치와 회전을 가중치 기반으로 보상 합산후 가중치 곱 + 아크탄젠트 사용 
    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err)) 
    # 아크 탄젠트를 사용하는 이유는 오차가 클수록 보상이 0에 수렴하고, 오차가 작을수록 보상이 1에 포화되도록 유도하기 위함.
    # 이를 통해 오차가 매우 작은 구간에서도 충분한 보상의 기울기를(gradient) 얻을 수 있고, 또한 보상이 0~1 사이에 깔끔하게 들어가게 됨.
    # 그리퍼와 박스 사이의 거리 보상 계산
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))

    # 해당 보상은 로봇이 초기 관절 포즈에서 벗어난 정도를 고려하는 것으로, 가능한 적게 벗아나게 하는 것이 목적.
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        collision.geoms_colliding(data, self._floor_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._hand_geom,
            # pinch 또한 지면에 닿지 않도록
            self._gripper_site,
        ]
    ]
    floor_collision = sum(hand_floor_collision) > 0 # True False를 합산해서 하나라도 1이면 True 반환
    no_floor_collision = (1 - floor_collision).astype(float)

    # Check for collisions with the floor
    hand_obstacle_collision = [
        collision.geoms_colliding(data, self._obstacle_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._hand_geom,
            self._box_geom,
        ]
    ]
    obstacle_collision = sum(hand_obstacle_collision) > 0 # True False를 합산해서 하나라도 1이면 True 반환
    no_obstacle_collision = (1 - obstacle_collision).astype(float)

    # target_pos를 활성화하는 함수 # 그리퍼가 박스와 거리가 0.012이하가 되지전 까지는 0이 반환됨, 만약 그리퍼가 박스와 충분히 가까워지고 
    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    # 보상 정리 및 반환
    rewards = {
        # 위 reward_config 만큼 곱해질 것 (지금은 다 0~1 사이)
        "gripper_box": gripper_box, # 그리퍼와 박스 사이의 거리 보상
        "box_target": box_target * info["reached_box"], # 박스와 목표 사이의 거리 보상 * 그러피가 박스에 도착했는지 온오프
        "no_floor_collision": no_floor_collision, # 바닥과 충돌은 안했는지 (충돌 했으면 보상 x)
        "robot_target_qpos": robot_target_qpos, # 초기 로봇 관절 포즈에서 덜 벗어난 정도 (벗어나면 보상 다운)
        "no_obstacle_collision": no_obstacle_collision,  # 장애물과 충돌은 안했는지 (충돌 했으면 보상 x)
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    # 그리퍼 위치
    gripper_pos = data.site_xpos[self._gripper_site]
    # 그리퍼 방향 행렬, 펼치고 회전축 3개만 가져옴, 나머지는 어차피 redundant
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    # 목표의 방향 행렬 quat를 회전 벡터로 변환
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])

    # 추가
    arm_ctrl_diff = data.ctrl[:6] - data.qpos[self._robot_qposadr[:-1]]  
    gripper_ctrl = data.ctrl[6:7]  # Just the gripper control 
    
    # 추가
    obstacle_pos = data.xpos[self._obstacle_geom]

    # 모든 요소(피쳐)를 하나의 벡터로 통합
    obs = jp.concatenate([
        # 로봇 관절 위치, 속도
        data.qpos,
        data.qvel,
        # 위에서 정의한 그리퍼 위치와 방향 벡터
        gripper_pos,
        gripper_mat[3:],
        # 박스 방향과 위치
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        # 목표와 박스 오차 및 회전 오차
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        # 제어 입력과 진짜 현제 관절 위치 차이
        # data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        # Control difference for arm joints  
        arm_ctrl_diff,  
        gripper_ctrl,  
        obstacle_pos,
    ])
    return obs
