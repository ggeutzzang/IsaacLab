"""
Phase 4: RL Environment 구현 (RewardManager + TerminationManager)

**Phase 3 대비 추가된 내용:**
1. ✅ ManagerBasedEnv → ManagerBasedRLEnv로 확장
2. ✅ RewardManager 추가: 보상 함수 구현
3. ✅ TerminationManager 추가: Episode 종료 조건 정의
4. ✅ step() 반환값 변화: (obs, info) → (obs, reward, terminated, truncated, info)

**학습 목표:**
- RL 환경의 MDP 완성: Observation + Reward + Termination
- RewardTermCfg 사용법: 보상 함수 정의 및 가중치 조정
- TerminationTermCfg 사용법: Episode 종료 조건 설정
- Gymnasium API 호환 환경 구현

**과제 설명:**
큐브를 z=1.0m 높이에 유지하는 것이 목표입니다.
- 보상: 큐브가 목표 높이에 가까울수록 높은 보상
- 페널티: 큐브가 바닥에 떨어지면 페널티
- 종료: 큐브가 너무 낮거나 높으면 Episode 종료

.. code-block:: bash

	# Run the script
	./isaaclab.sh -p scripts/tutorials/03_envs/my_clone/phase4_rl_env.py --num_envs 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# argparse 인자 추가
parser = argparse.ArgumentParser(description="Phase4: RL Environment 학습")
parser.add_argument("--num_envs", type=int, default=2, help="생성할 환경의 개수")

# AppLauncher의 CLI 인자 추가
AppLauncher.add_app_launcher_args(parser)

# 인자 파싱
args_cli = parser.parse_args()

# Omniverse 기동
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ★ 학습 포인트 ────────────────────────────────────────────────
# AppLauncher 이후에 Isaac Lab 모듈 import!
# ────────────────────────────────────────────────────────────


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# MDP Functions (관찰값, 보상, 종료 조건 함수)
##
# Phase 3에서 구현한 관찰 함수는 그대로 재사용
# Phase 4에서는 보상 함수와 종료 조건 함수를 추가


def cube_position(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
	"""큐브의 위치를 환경 원점 기준 상대 좌표로 반환 (Phase 3 재사용)"""
	asset = env.scene[asset_cfg.name]
	return asset.data.root_pos_w - env.scene.env_origins


def cube_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
	"""큐브의 선속도 반환 (Phase 3 재사용)"""
	asset = env.scene[asset_cfg.name]
	return asset.data.root_lin_vel_w


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ Phase 4 신규: 보상 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def reward_cube_height(env, asset_cfg: SceneEntityCfg, target_height: float) -> torch.Tensor:
	"""큐브가 목표 높이에 가까울수록 높은 보상

	Args:
		env: ManagerBasedRLEnv 인스턴스
		asset_cfg: Scene entity 설정
		target_height: 목표 높이 (예: 1.0m)

	Returns:
		shape (num_envs,) 텐서 (각 환경의 보상값)
	"""
	asset = env.scene[asset_cfg.name]
	# 큐브의 z 좌표 (환경 원점 기준)
	cube_z = (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2])

	# 목표 높이와의 거리
	distance = torch.abs(cube_z - target_height)

	# 거리에 반비례하는 보상 (가까울수록 1.0, 멀수록 0.0)
	# exp(-distance) 형태로 부드러운 보상 곡선 생성
	reward = torch.exp(-distance * 2.0)  # 2.0은 곡선의 가파름 조절

	return reward


def penalty_cube_fallen(env, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
	"""큐브가 너무 낮으면 페널티

	Args:
		env: ManagerBasedRLEnv 인스턴스
		asset_cfg: Scene entity 설정
		threshold: 페널티 기준 높이 (예: 0.2m)

	Returns:
		shape (num_envs,) 텐서 (각 환경의 페널티)
	"""
	asset = env.scene[asset_cfg.name]
	cube_z = (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2])

	# threshold 이하로 떨어지면 -1.0 페널티
	penalty = torch.where(cube_z < threshold, -1.0, 0.0)

	return penalty


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ Phase 4 신규: 종료 조건 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def termination_cube_out_of_bounds(env, asset_cfg: SceneEntityCfg, min_height: float, max_height: float) -> torch.Tensor:
	"""큐브가 범위를 벗어나면 Episode 종료

	Args:
		env: ManagerBasedRLEnv 인스턴스
		asset_cfg: Scene entity 설정
		min_height: 최소 높이 (예: 0.1m)
		max_height: 최대 높이 (예: 2.0m)

	Returns:
		shape (num_envs,) bool 텐서 (True이면 종료)
	"""
	asset = env.scene[asset_cfg.name]
	cube_z = (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2])

	# 범위 벗어나면 True
	out_of_bounds = (cube_z < min_height) | (cube_z > max_height)

	return out_of_bounds


##
# Scene Configuration (Phase 3과 동일)
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
	"""Phase 4 Scene 설정: Phase 3과 동일"""

	# 지형 추가
	terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="plane",
		debug_vis=False,
	)

	# 조명 추가
	light = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
	)

	# 큐브 추가 (Phase 3과 동일)
	cube: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/cube",
		spawn=sim_utils.CuboidCfg(
			size=(0.2, 0.2, 0.2),
			rigid_props=sim_utils.RigidBodyPropertiesCfg(),
			mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
	)


##
# MDP Configuration
##


@configclass
class ActionsCfg:
	"""Phase 4: 액션 설정 (여전히 빈 클래스)

	Phase 5에서 실제 큐브에 힘 적용 예정
	"""
	pass


@configclass
class ObservationsCfg:
	"""Phase 4: 관찰값 설정 (Phase 3과 동일)"""

	@configclass
	class PolicyCfg(ObservationGroupCfg):
		"""Policy 그룹: 큐브 위치 + 속도"""

		cube_pos = ObservationTermCfg(
			func=cube_position,
			params={"asset_cfg": SceneEntityCfg("cube")},
		)

		cube_vel = ObservationTermCfg(
			func=cube_velocity,
			params={"asset_cfg": SceneEntityCfg("cube")},
		)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = True

	policy: PolicyCfg = PolicyCfg()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ Phase 4 신규: RewardManager 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@configclass
class RewardsCfg:
	"""Phase 4: 보상 설정"""

	# 큐브가 목표 높이에 가까울수록 보상
	# weight=1.0 → 매 step마다 최대 1.0점 획득 가능
	height_reward = RewardTermCfg(
		func=reward_cube_height,
		params={"asset_cfg": SceneEntityCfg("cube"), "target_height": 1.0},
		weight=1.0,
	)

	# 큐브가 바닥에 떨어지면 페널티
	# weight=2.0 → 떨어지면 -2.0점 (강한 페널티)
	fallen_penalty = RewardTermCfg(
		func=penalty_cube_fallen,
		params={"asset_cfg": SceneEntityCfg("cube"), "threshold": 0.2},
		weight=2.0,
	)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ Phase 4 신규: TerminationManager 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@configclass
class TerminationsCfg:
	"""Phase 4: 종료 조건 설정"""

	# 큐브가 범위를 벗어나면 Episode 종료
	# time_out=False → 즉시 종료 (terminated=True)
	# min_height=0.05로 설정 (큐브 크기 0.2m의 절반 이하)
	out_of_bounds = TerminationTermCfg(
		func=termination_cube_out_of_bounds,
		params={"asset_cfg": SceneEntityCfg("cube"), "min_height": 0.05, "max_height": 2.0},
		time_out=False,
	)

	# 시간 제한은 ManagerBasedRLEnv가 episode_length_s를 사용하여 자동 처리
	# time_out=True 조건을 별도로 정의하지 않아도 됨


@configclass
class EventCfg:
	"""Phase 4: 더미 Events (Phase 6에서 구현 예정)"""
	pass


##
# Environment Configuration
##


@configclass
class Phase4EnvCfg(ManagerBasedRLEnvCfg):
	"""Phase 4 Environment 설정 (ManagerBasedRLEnv)

	Phase 3과의 차이점:
	- ManagerBasedEnvCfg → ManagerBasedRLEnvCfg로 변경
	- rewards, terminations 추가
	"""

	# Simulation 설정
	sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=2)

	# Scene 설정
	scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)

	# MDP 설정
	actions: ActionsCfg = ActionsCfg()
	observations: ObservationsCfg = ObservationsCfg()
	events: EventCfg = EventCfg()

	# ★ Phase 4 신규: RL 관련 설정
	rewards: RewardsCfg = RewardsCfg()
	terminations: TerminationsCfg = TerminationsCfg()

	# Episode 설정
	episode_length_s = 5.0
	decimation = 2


##
# Main 함수
##


def main():
	"""Main function."""

	print("[Phase 4] RL Environment 학습 - RewardManager + TerminationManager\n")

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# 1. Environment 설정 및 생성
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# Phase 3: ManagerBasedEnv
	# Phase 4: ManagerBasedRLEnv (Gymnasium 호환)
	env_cfg = Phase4EnvCfg()
	env = ManagerBasedRLEnv(cfg=env_cfg)
	print(f"Environment 생성 완료 | Scene entities: {list(env.scene.keys())}")

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# 2. Reset
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	obs, _ = env.reset()
	print(f"Reset 완료 | Observation groups: {list(obs.keys())}")
	print(f"Policy observation shape: {obs['policy'].shape}\n")

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# 3. 시뮬레이션 루프 (RL 환경)
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# Phase 3: obs, info = env.step(action)  # 2개 반환
	# Phase 4: obs, reward, terminated, truncated, info = env.step(action)  # 5개 반환
	dummy_action = torch.zeros(env.num_envs, 0, device=env.device)

	episode_count = 0
	total_reward = torch.zeros(env.num_envs, device=env.device)

	for count in range(500):  # Phase 4에서는 더 많이 실행 (Episode 종료 확인)
		if not simulation_app.is_running():
			break

		# ★ Phase 4 신규: step() 반환값이 5개로 증가
		obs, reward, terminated, truncated, info = env.step(dummy_action)

		# 보상 누적
		total_reward += reward

		# 10 스텝마다 상태 출력
		if count % 10 == 0:
			cube_z = obs['policy'][0, 2]
			reward_val = reward[0].item()
			print(
				f"Step {count:3d} | "
				f"Env 0: z={cube_z:.3f}m, "
				f"reward={reward_val:+.3f}, "
				f"total={total_reward[0]:.2f}"
			)

		# Episode 종료 확인
		if terminated.any() or truncated.any():
			episode_count += 1
			print(f"\n{'='*60}")
			print(f"Episode {episode_count} 종료!")
			print(f"  - Terminated 환경: {terminated.nonzero(as_tuple=True)[0].tolist()}")
			print(f"  - Truncated 환경: {truncated.nonzero(as_tuple=True)[0].tolist()}")
			print(f"  - 누적 보상: {total_reward.tolist()}")
			print(f"{'='*60}\n")

			# 종료된 환경 리셋
			if terminated.any() or truncated.any():
				obs, _ = env.reset()
				total_reward.zero_()

	print(f"\n[완료] {count + 1}개 스텝 실행 완료 (Episode {episode_count}회 종료) ✓")
	print("\n" + "="*60)
	print("시뮬레이션이 완료되었습니다.")
	print("Enter 키를 누르면 종료됩니다...")
	print("="*60)

	# 사용자 입력 대기
	input()

	env.close()


if __name__ == "__main__":
	main()
	simulation_app.close()
