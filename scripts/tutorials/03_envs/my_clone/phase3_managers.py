"""
Phase 3: Manager 시스템 학습 (ObservationManager + ActionManager)

**Phase 2 대비 추가된 내용:**
1. ✅ RigidObjectCfg로 Scene에 큐브 객체 추가 (MySceneCfg.cube)
2. ✅ 실제 MDP 함수 구현: cube_position(), cube_velocity()
3. ✅ ObservationManager 활용: 더미 관찰값 → 실제 큐브 상태 추적
4. ✅ SceneEntityCfg로 Scene 객체 참조하는 방법 학습

**학습 목표:**
- Configuration-Driven Architecture의 핵심: Cfg 정의 → 프레임워크 자동 실행
- ObservationManager의 동작 원리: ObservationTermCfg + MDP 함수
- MDP 함수의 재사용성: SceneEntityCfg를 통한 유연한 설계

.. code-block:: bash

	# Run the script
	./isaaclab.sh -p scripts/tutorials/03_envs/my_clone/phase3_managers.py --num_envs 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# argparse 인자 추가
parser = argparse.ArgumentParser(description="Phase3: Manager 시스템 학습")
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
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# MDP Functions (관찰값 및 액션 함수)
##
# ★ Phase 3 신규 ──────────────────────────────────────────
# Phase 2에서는 더미 관찰 함수만 사용
# Phase 3에서는 실제 큐브 상태를 읽는 MDP 함수 구현
# ────────────────────────────────────────────────────────


def cube_position(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
	"""큐브의 위치를 환경 원점 기준 상대 좌표로 반환

	Args:
		env: ManagerBasedEnv 인스턴스
		asset_cfg: Scene entity 설정 (큐브 이름 포함)

	Returns:
		shape (num_envs, 3) 텐서 (x, y, z 상대 위치)
	"""
	# Scene에서 asset_cfg.name으로 큐브 가져오기
	asset = env.scene[asset_cfg.name]

	# 큐브의 월드 좌표에서 환경 원점을 빼서 상대 좌표 계산
	return asset.data.root_pos_w - env.scene.env_origins


def cube_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
	"""큐브의 선속도 반환

	Args:
		env: ManagerBasedEnv 인스턴스
		asset_cfg: Scene entity 설정

	Returns:
		shape (num_envs, 3) 텐서 (vx, vy, vz)
	"""
	asset = env.scene[asset_cfg.name]
	return asset.data.root_lin_vel_w


##
# Scene Configuration
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
	"""Phase 3 Scene 설정: 지형 + 조명 + 큐브"""

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# Phase 2와 동일: 지형 + 조명
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

	# 지형 추가 (Phase 2와 동일)
	terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="plane",
		debug_vis=False,
	)

	# 조명 추가 (Phase 2와 동일)
	light = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
	)

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# ★ Phase 3 신규: 큐브 객체 추가
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# - RigidObjectCfg: 물리 시뮬레이션이 적용되는 강체 객체
	# - prim_path에 {ENV_REGEX_NS} 사용 → 각 환경마다 별도 인스턴스 생성
	#   예: num_envs=2이면 큐브 2개 생성
	#       /World/envs/env_0/cube (환경 0의 큐브)
	#       /World/envs/env_1/cube (환경 1의 큐브)
	#   주의: 코드에서는 1개만 정의하지만, 환경 개수만큼 자동 복제됨!
	# - spawn: CuboidCfg로 큐브 형태 정의 (크기, 질량, 재질 등)
	# - init_state: 초기 위치 (각 환경의 원점에서 z=1.0m 높이)
	# - env_spacing=2.5 → 환경 간격 2.5m (큐브들이 일렬로 배치)
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	cube: RigidObjectCfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/cube",  # ← 벡터화된 환경의 핵심!
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
# MDP Configuration (ObservationManager, ActionManager)
##


@configclass
class ActionsCfg:
	"""Phase 3: 액션 설정 (빈 클래스)

	Phase 5에서 실제 큐브에 힘 적용하는 ActionTerm 추가 예정
	ManagerBasedEnv는 빈 ActionsCfg도 허용함
	"""
	pass


@configclass
class ObservationsCfg:
	"""Phase 3: 실제 관찰값 설정"""

	@configclass
	class PolicyCfg(ObservationGroupCfg):
		"""Policy 그룹: 큐브 위치 + 속도"""

		# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
		# ★ Phase 3 신규: 실제 MDP 함수 기반 관찰값
		# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
		# Phase 2: dummy_observation() → 상수 0.0 반환
		# Phase 3: cube_position(), cube_velocity() → 실제 물리 상태 추적
		#
		# ObservationTermCfg 사용법:
		# - func: 관찰 함수 (env, asset_cfg 인자 받음)
		# - params: {"asset_cfg": SceneEntityCfg("cube")}
		#   → Scene의 "cube" 객체를 함수에 전달
		# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

		# 큐브 위치 (3차원): 환경 원점 기준 상대 좌표
		cube_pos = ObservationTermCfg(
			func=cube_position,
			params={"asset_cfg": SceneEntityCfg("cube")},
		)

		# 큐브 속도 (3차원): 월드 좌표계 기준 선속도
		cube_vel = ObservationTermCfg(
			func=cube_velocity,
			params={"asset_cfg": SceneEntityCfg("cube")},
		)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = True  # 위치(3) + 속도(3) = 6차원

	policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
	"""Phase 3: 더미 Events (Phase 6에서 구현 예정)"""
	pass


##
# Environment Configuration
##


@configclass
class Phase3EnvCfg(ManagerBasedEnvCfg):
	"""Phase 3 Environment 설정"""

	# Simulation 설정
	sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=2)

	# Scene 설정 (큐브 포함)
	scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)

	# MDP 설정
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

	print("[Phase 3] Manager 시스템 학습 - ObservationManager + ActionManager\n")

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# 1. Environment 설정 및 생성 (Phase 2와 동일)
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	env_cfg = Phase3EnvCfg()
	env = ManagerBasedEnv(cfg=env_cfg)
	print(f"Environment 생성 완료 | Scene entities: {list(env.scene.keys())}")
	# Phase 3에서는 'cube'가 추가됨: ['terrain', 'cube', 'light']

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# 2. Reset (Phase 2와 동일하지만 관찰값 내용이 다름)
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	obs, _ = env.reset()
	print(f"Reset 완료 | Observation groups: {list(obs.keys())}")
	print(f"Policy observation shape: {obs['policy'].shape}")  # (num_envs, 6)
	# Phase 2: obs['policy'] = [0.0] (더미)
	# Phase 3: obs['policy'] = [x, y, z, vx, vy, vz] (실제 큐브 상태)
	print(f"첫 번째 환경 관찰값: pos={obs['policy'][0, :3]}, vel={obs['policy'][0, 3:]}\n")

	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# 3. 시뮬레이션 루프 (Phase 2와 유사하지만 관찰값 출력 추가)
	# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
	# Phase 3에서는 ActionsCfg가 빈 클래스이므로 액션 차원 = 0
	# 큐브가 중력으로 떨어지는 것을 관찰값으로 확인
	dummy_action = torch.zeros(env.num_envs, 0, device=env.device)

	for count in range(200):
		if not simulation_app.is_running():
			break
		obs, _ = env.step(dummy_action)

		# ★ Phase 3 신규: 10 스텝마다 큐브 상태 출력
		# ObservationManager가 실시간으로 큐브 위치/속도 추적
		if count % 10 == 0:
			print(f"Step {count:3d} | Env 0 큐브 위치 z={obs['policy'][0, 2]:.3f}m, 속도 vz={obs['policy'][0, 5]:.3f}m/s")

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
