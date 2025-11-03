"""
Phase 2: Scene 설정 (Terrain + Light) - ManagerBasedEnv 방식

This script demonstrates:
1. InteractiveSceneCfg를 상속한 MySceneCfg 클래스 정의
2. terrain: TerrainImporterCfg로 지형 추가
3. light: AssetBaseCfg + DomeLightCfg로 조명 추가
4. ManagerBasedEnv의 기본 구조 학습 (Actions, Observations, Events는 더미)

.. code-block:: bash

	# Run the script
	./isaaclab.sh -p scripts/tutorials/03_envs/my_clone/phase2_scene.py --num_envs 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# argparse 인자 추가
parser = argparse.ArgumentParser(description="Phase2: Scene 설정 학습 (ManagerBasedEnv)")
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
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Phase 2 전용: 더미 관찰 함수
##


def dummy_observation(env) -> torch.Tensor:
	"""Phase 2용 더미 관찰값 (상수 0.0 반환)"""
	return torch.zeros(env.num_envs, 1, device=env.device)


##
# Scene Configuration
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
	"""Phase 2 Scene 설정: 지형 + 조명 (큐브는 Phase 3에서 추가)"""

	# 지형 추가 (원본 코드와 동일)
	terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="plane",
		debug_vis=False,
	)

	# 조명 추가 (원본 코드와 동일)
	light = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
	)


##
# MDP Configuration (Dummy - Phase 2에서는 아직 구현 안 함)
##


@configclass
class ActionsCfg:
	"""Phase 2: 더미 Actions (Phase 5에서 구현 예정)"""
	pass


@configclass
class ObservationsCfg:
	"""Phase 2: 더미 Observations (Phase 4에서 구현 예정)"""

	@configclass
	class PolicyCfg(ObservationGroupCfg):
		"""Policy에 필요한 최소 관찰값 그룹"""

		# Phase 2: 더미 관찰값 (상수 0.0)
		# Phase 3에서 큐브 위치로 교체 예정
		dummy_obs = ObservationTermCfg(func=dummy_observation)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = True

	policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
	"""Phase 2: 더미 Events (Phase 6에서 구현 예정)"""
	pass


##
# Environment Configuration
##


@configclass
class Phase2EnvCfg(ManagerBasedEnvCfg):
	"""Phase 2 Environment 설정 (ManagerBasedEnv 방식)"""

	# Simulation 설정
	sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=2)

	# Scene 설정
	scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)

	# MDP 설정 (더미)
	actions: ActionsCfg = ActionsCfg()
	observations: ObservationsCfg = ObservationsCfg()
	events: EventCfg = EventCfg()

	# Episode 설정
	episode_length_s = 5.0
	decimation = 2


##
# Main 함수
##


def main():
	"""Main function."""

	print("[Phase 2] Scene 설정 테스트 - ManagerBasedEnv\n")

	# 1. Environment 설정 및 생성
	env_cfg = Phase2EnvCfg()
	env = ManagerBasedEnv(cfg=env_cfg)
	print(f"Environment 생성 완료 | Scene entities: {list(env.scene.keys())}")

	# 2. Reset
	obs, _ = env.reset()
	print(f"Reset 완료 | Observation groups: {list(obs.keys())}\n")

	# 3. 시뮬레이션 루프 (200 스텝)
	dummy_action = torch.empty(env.num_envs, 0, device=env.device)

	for count in range(200):
		if not simulation_app.is_running():
			break
		obs, _ = env.step(dummy_action)

	print(f"\n[완료] {count + 1}개 스텝 실행 완료 ✓")
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
