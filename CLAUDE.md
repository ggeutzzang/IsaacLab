# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**Isaac Lab**은 NVIDIA Isaac Sim 기반의 GPU 가속 로봇 학습 프레임워크입니다. 강화학습(RL), 모방학습(IL), 모션 플래닝 등의 워크플로우를 통합하며, 벡터화된 환경과 정확한 물리/센서 시뮬레이션을 제공합니다.

- **버전:** 2.3.0 (Isaac Sim 4.5 / 5.0 / 5.1 호환)
- **Python:** 3.10+ (3.11 권장)
- **PyTorch:** 2.7.0+cu128
- **라이선스:** BSD-3-Clause (메인), Apache-2.0 (isaaclab_mimic)

## 환경 설정 및 빌드

### 초기 설정 (v2.3.0) - 권장 방법

**중요:** Isaac Lab v2.3.0은 환경 이름을 `env_isaaclab`로 사용하는 것을 권장합니다.

```bash
# 1. Conda 환경 생성 및 활성화
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

# 2. PyTorch 설치 (CUDA 12.8)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Isaac Sim 5.1 설치 (약 4.5GB 다운로드, 10-15분 소요)
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# 4. Isaac Lab 설치
./isaaclab.sh --install  # Linux
# or
isaaclab.bat --install   # Windows

# 5. 호환성 문제 해결 (필수!)
pip install "scipy==1.11.4"  # SciPy 1.15.3 → 1.11.4 다운그레이드
conda config --env --set channel_priority strict
conda config --env --add channels conda-forge
conda install -y -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"  # GLIBCXX 3.4.30 지원
```

**환경 이름 참고:**
- Isaac Lab은 기본적으로 `env_isaaclab` 환경을 찾습니다
- `isaaclab.sh` 스크립트는 자동으로 올바른 Python 경로를 찾아 사용합니다
- 다른 이름을 사용하면 Python 경로 문제가 발생할 수 있습니다

**주의:** Isaac Sim 4.5를 사용하려면:
```bash
# Python 3.10 환경 사용
conda create -n isaaclab_4_5 python=3.10
# PyTorch 2.5.1 설치
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
# Isaac Sim 4.5 설치
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

### 공통 개발 명령어

```bash
# 환경 스크립트 사용 (isaaclab.sh를 통해 Python 실행)
./isaaclab.sh -p <script.py>  # Python 스크립트 실행
./isaaclab.sh -s             # Isaac Sim GUI 실행

# 강화학습 학습 실행
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Anymal-C-v0
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --num_envs 512
python scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --headless

# 학습된 정책 플레이
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Anymal-C-v0

# 환경 목록 확인
python scripts/environments/list_envs.py

# 환경 테스트 (텔레조작)
python scripts/environments/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-v0
```

### 테스트 실행

```bash
# 전체 테스트 실행 (CI 환경용)
python tools/run_all_tests.py

# 단일 테스트 파일 실행
pytest source/isaaclab/isaaclab/envs/test/test_environment.py

# 특정 모듈 테스트
pytest source/isaaclab/isaaclab/sensors/test/
pytest source/isaaclab_tasks/test/
```

**주의사항:**
- 테스트는 각 파일별로 독립 실행됨 (tools/conftest.py 참조)
- 일부 테스트는 긴 타임아웃 필요 (예: test_environments_training.py는 5000초)
- 스킵된 테스트: `test_argparser_launch.py`, `test_record_video.py`

### 코드 품질 도구

```bash
# Pre-commit hooks 설치 및 실행
pre-commit install
pre-commit run --all-files

# 개별 도구 실행
isort source/                        # Import 정렬
black source/ scripts/               # 코드 포매팅
flake8 source/ scripts/              # 린팅
pyright source/ scripts/             # 타입 체크
codespell                            # 철자 검사
```

### 문서 빌드

```bash
cd docs
pip install -r requirements.txt

# 현재 버전 문서 빌드
make current-docs

# 다중 버전 문서 빌드
make multi-docs
```

## 핵심 아키텍처

### 1. 두 가지 환경 설계 패턴

**Manager-Based Workflow (권장):**
- 모듈식 설계로 각 컴포넌트를 독립적으로 개발
- `ManagerBasedEnv`, `ManagerBasedRLEnv` 사용
- 재사용 가능한 환경 구성 요소:
  - `ActionManager`: 액션 처리
  - `ObservationManager`: 관찰값 생성
  - `RewardManager`: 보상 계산
  - `TerminationManager`: 종료 조건
  - `CommandManager`: 명령어 처리
  - `EventManager`: 이벤트 (도메인 랜덤화 등)
  - `CurriculumManager`: 커리큘럼 학습
  - `RecorderManager`: 데이터 기록

**Direct Workflow (고성능):**
- 세밀한 제어가 필요한 경우
- `DirectRLEnv`, `DirectMARLEnv` 사용
- PyTorch JIT, Warp로 최적화 가능

### 2. 설정 기반 아키텍처 (Configuration-Driven Architecture)

**핵심 설계 철학:**
Isaac Lab의 개발은 **~Cfg 클래스 정의가 80%**를 차지합니다. 모든 컴포넌트는 설정(Configuration)과 실행(Runtime)이 분리된 이중 레이어 구조로 설계되었습니다.

```
Configuration Layer (~Cfg 클래스)     Runtime Layer (실행 클래스)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━━━━
InteractiveSceneCfg          →    InteractiveScene
ArticulationCfg              →    Articulation
ActionsCfg                   →    ActionManager
ObservationsCfg              →    ObservationManager
RewardsCfg                   →    RewardManager
```

**작명 규칙:** `{Component}Cfg` → `{Component}` 또는 `{Component}Manager`

**기본 환경 설정 예시:**
```python
@configclass
class MyEnvCfg(ManagerBasedEnvCfg):
    """사용자는 '무엇을' 만들지 선언 (선언적 프로그래밍)"""
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

# 프레임워크가 자동으로 '어떻게' 실행할지 처리
env = ManagerBasedEnv(MyEnvCfg())  # 이 한 줄에서 모든 초기화 발생
```

**장점:**
- **타입 안전성**: Python 타입 시스템 + IDE 자동완성
- **재사용성**: 상속과 조합으로 쉽게 확장
- **가독성**: 복잡한 환경을 계층적 설정 구조로 표현
- **버전 관리**: 모든 설정이 코드이므로 Git으로 추적 가능

### 3. 프로젝트 구조

```
source/
├── isaaclab/                  # 메인 프레임워크 (21,310 LOC)
│   ├── envs/                  # 환경 시스템 (Manager-Based, Direct)
│   ├── sim/                   # Isaac Sim 통합
│   ├── assets/                # Articulation, RigidObject, Deformable
│   ├── sensors/               # Camera, ContactSensor, IMU, RayCaster
│   ├── managers/              # 8개 매니저 시스템
│   ├── controllers/           # DifferentialIK, OperationalSpace, RMPFlow
│   ├── devices/               # Gamepad, Keyboard, OpenXR, Spacemouse
│   ├── scene/                 # InteractiveScene
│   ├── terrains/              # 절차적 지형 생성
│   └── utils/                 # 수학, 설정, IO 유틸리티
├── isaaclab_tasks/            # 환경 구현체
│   ├── manager_based/         # Classic, Locomotion, Manipulation, etc.
│   └── direct/                # 직접 구현 환경
├── isaaclab_assets/           # 로봇 및 센서 구성
├── isaaclab_rl/              # RSL-RL, SB3, RL-Games, SKRL 통합
└── isaaclab_mimic/           # RoboMimic 통합 (Linux 전용)

scripts/
├── reinforcement_learning/    # RL 프레임워크별 학습 스크립트
├── imitation_learning/        # 모방학습 스크립트
├── environments/              # 환경 테스트 유틸리티
├── benchmarks/                # 성능 벤치마크
└── tutorials/                 # 튜토리얼 스크립트
```

### 4. 환경 등록 시스템

환경은 Gymnasium API로 등록:
```python
# 형식: Isaac-<Task>-<Robot>-v<X>
gym.make("Isaac-Velocity-Rough-Anymal-C-v0")
gym.make("Isaac-Lift-Cube-Franka-v0")
gym.make("Isaac-Ant-v0")
gym.make("Isaac-PickPlace-GR1T2-Abs-v0")
```

#### 주요 로봇별 환경 예시

**휴머노이드 로봇 (GR1T2 - Fourier Intelligence)**

Isaac Lab은 Fourier Intelligence의 GR1T2 휴머노이드 로봇을 지원합니다 (54 DOF).

| 환경 ID | 작업 | 제어 방식 | 설명 |
|---------|------|----------|------|
| `Isaac-PickPlace-GR1T2-Abs-v0` | Pick & Place | Pink IK | 스티어링 휠 픽앤플레이스 작업 |
| `Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0` | Pick & Place | Pink IK | 허리 관절 활용 버전 |
| `Isaac-NutPour-GR1T2-Pink-IK-Abs-v0` | Nut Pouring | Pink IK | 견과류 붓기 작업 |
| `Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0` | Exhaust Pipe | Pink IK | 배기관 조립 작업 |

**실행 예시:**
```bash
# 키보드 텔레조작 (Pink IK 컨트롤러 사용)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --teleop_device keyboard \
    --enable_pinocchio

# VR 핸드 트래킹
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --teleop_device handtracking \
    --enable_pinocchio

# 시연 데이터 기록 (RoboMimic 학습용)
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --num_demos 100 \
    --enable_pinocchio
```

**중요**: GR1T2 환경은 **Pink Inverse Kinematics** 컨트롤러를 사용하므로 `--enable_pinocchio` 플래그가 필수입니다.

**GR1T2 로봇 구성:**
- **총 DOF**: 54개 (머리 3 + 허리 3 + 다리 12 + 팔 14 + 손 22)
- **제어 방식**: Pink IK (엔드 이펙터 목표 → 조인트 각도 자동 계산)
- **액션 공간**: 왼팔/오른팔 위치(3) + 자세(4) + 손가락 조인트(11×2) = 32차원
- **로봇 설정**: `source/isaaclab_assets/isaaclab_assets/robots/fourier.py`
- **환경 설정**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/`

**조작 로봇 (Franka Emika Panda)**

| 환경 ID | 작업 | 설명 |
|---------|------|------|
| `Isaac-Lift-Cube-Franka-v0` | Lifting | 큐브 들어올리기 |
| `Isaac-Reach-Franka-v0` | Reaching | 목표 지점 도달 |

**사족 보행 로봇 (ANYmal-C)**

| 환경 ID | 작업 | 설명 |
|---------|------|------|
| `Isaac-Velocity-Rough-Anymal-C-v0` | Locomotion | 거친 지형 주행 |
| `Isaac-Velocity-Flat-Anymal-C-v0` | Locomotion | 평평한 지형 주행 |

### 5. 관찰값 구조

다중 관찰값 그룹 지원 (비대칭 액터-크리틱, 다중 에이전트):
```python
VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
# 예: {"policy": tensor([...]), "critic": tensor([...])}
```

## 개발 가이드라인

### 코드 분석 가이드라인 (Claude Code 전용)

Isaac Lab 코드를 분석하거나 수정할 때는 **Configuration-Driven Architecture** 관점을 항상 유지하세요.

#### 1. 코드 분석 시 필수 확인 사항

**모든 코드 분석 시 다음 구조를 먼저 파악:**

```
1. Configuration Layer (~Cfg 클래스)
   - 어떤 설정 클래스들이 정의되어 있는가?
   - 각 설정 클래스의 계층 구조는?
   - 기본값과 필수값(MISSING)은 무엇인가?

2. Runtime Layer (실행 클래스)
   - Cfg 클래스를 받아서 어떻게 초기화하는가?
   - 어떤 Manager들이 생성되는가?
   - 실제 실행 로직은 어디에 있는가?

3. MDP 함수 (isaaclab/envs/mdp/)
   - 관찰값, 보상, 종료 조건 함수는 어디에 정의되어 있는가?
   - 어떤 SceneEntityCfg를 사용하는가?
```

#### 2. 코드 설명 시 포함할 내용

**파일 분석 시 항상 다음 순서로 설명:**

1. **Configuration Layer 설명**
   ```python
   @configclass
   class SomeCfg:  # ← 이것이 무엇을 설정하는지
       param1 = ...  # ← 각 파라미터의 의미
   ```

2. **Runtime Layer 연결**
   ```python
   # SomeCfg → SomeManager 또는 Some 클래스로 변환
   some_instance = Some(SomeCfg())
   ```

3. **실행 흐름**
   - 사용자가 Cfg 정의 → 프레임워크가 자동 실행
   - 어떤 Manager가 생성되고 어떤 순서로 실행되는지

4. **재사용성과 확장성**
   - 이 설정 클래스를 어떻게 상속/조합하여 확장할 수 있는지
   - `replace()` 메서드로 어떻게 변형할 수 있는지

#### 3. 코드 수정 시 원칙

**항상 Configuration Layer를 먼저 수정:**

```python
# ✅ 올바른 순서
# 1단계: Cfg 클래스 수정/생성
@configclass
class NewFeatureCfg:
    param = default_value

# 2단계: 기존 Cfg에 통합
@configclass
class EnvCfg(ManagerBasedEnvCfg):
    new_feature: NewFeatureCfg = NewFeatureCfg()

# 3단계: 필요한 경우에만 Runtime 로직 수정
class NewFeatureManager:
    def __init__(self, cfg: NewFeatureCfg):
        # Cfg 기반 초기화
```

**❌ 피해야 할 패턴:**
```python
# Runtime 클래스에 직접 하드코딩
class SomeManager:
    def __init__(self):
        self.value = 10  # ❌ Cfg로 설정 가능해야 함!
```

#### 4. 파일 참조 시 작명 규칙 준수

**Cfg 패턴 인식:**
- `{Component}Cfg` → Configuration Layer
- `{Component}` 또는 `{Component}Manager` → Runtime Layer
- `mdp.{function_name}` → MDP 함수 (재사용 가능한 관찰/보상/종료 함수)

**예시:**
```python
# Configuration
ArticulationCfg          → 로봇 설정
ActionsCfg              → 액션 설정
ObservationsCfg         → 관찰값 설정

# Runtime
Articulation            → 실제 로봇 객체
ActionManager           → 액션 처리 매니저
ObservationManager      → 관찰값 처리 매니저

# MDP Functions
mdp.joint_pos_rel       → 관절 위치 관찰 함수
mdp.is_terminated       → 종료 조건 함수
```

#### 5. 새 기능 추가 시 체크리스트

```markdown
- [ ] Cfg 클래스 정의 완료
- [ ] @configclass 데코레이터 적용
- [ ] 기본값 또는 MISSING 지정
- [ ] 타입 힌팅 추가 (선택사항이지만 권장)
- [ ] 상위 EnvCfg에 통합
- [ ] Runtime 클래스가 Cfg를 받도록 구현
- [ ] to_dict(), replace() 등 configclass 메서드 활용 가능 확인
- [ ] 재사용성을 위해 상속 구조 고려
```

#### 6. 디버깅 시 우선순위

**문제 발생 시 다음 순서로 확인:**

1. **Cfg 검증**: `env_cfg.validate()` - MISSING 값 확인
2. **Cfg 내용 확인**: `env_cfg.to_dict()` - 전체 설정 출력
3. **Manager 초기화 로그**: 각 Manager가 어떤 Cfg를 받았는지
4. **MDP 함수 실행**: 관찰/보상 함수가 올바른 값을 반환하는지

#### 7. 실전 경험: 주요 함정과 해결책

**함정 1: ManagerBasedEnv vs DirectRLEnv 혼동**

❌ **잘못된 접근:**
```python
# DirectRLEnv 패턴 사용 (원본이 ManagerBasedEnv인데)
class MyEnv(DirectRLEnv):
    def _setup_scene(self):
        spawn_ground_plane(...)
        # 직접 구현...
```

✅ **올바른 접근:**
```python
# ManagerBasedEnv 패턴 사용 (원본과 동일)
@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(...)
    light = AssetBaseCfg(...)

@configclass
class MyEnvCfg(ManagerBasedEnvCfg):
    scene: MySceneCfg = MySceneCfg(...)
```

**교훈:** 원본 코드의 아키텍처를 먼저 파악하고 동일한 패턴을 따라야 함.

---

**함정 2: ObservationManager 빈 그룹 에러**

❌ **잘못된 코드:**
```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pass  # 빈 그룹!
    policy: PolicyCfg = PolicyCfg()
```

**에러:**
```
RuntimeError: Unable to concatenate observation terms in group 'policy'.
The shapes of the terms are: [].
```

✅ **해결:**
```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # 최소 1개의 term 필요
        dummy_obs = ObservationTermCfg(func=dummy_observation)
    policy: PolicyCfg = PolicyCfg()

def dummy_observation(env) -> torch.Tensor:
    """반드시 (num_envs, feature_dim) 텐서 반환"""
    return torch.zeros(env.num_envs, 1, device=env.device)
```

**교훈:** ObservationManager는 최소 1개의 term 필요. ActionManager/EventManager는 0개 허용.

---

**함정 3: ManagerBasedEnv의 step() 반환값 혼동**

❌ **잘못된 코드:**
```python
# RL 환경처럼 5개 값 기대
obs, reward, terminated, truncated, info = env.step(action)
```

**에러:**
```
ValueError: not enough values to unpack (expected 5, got 2)
```

✅ **해결:**
```python
# ManagerBasedEnv는 2개만 반환 (obs, info)
obs, info = env.step(action)

# ManagerBasedRLEnv는 5개 반환
obs, reward, terminated, truncated, info = rl_env.step(action)
```

**교훈:**
- `ManagerBasedEnv`: 일반 환경용, `step()` → `(obs, info)`
- `ManagerBasedRLEnv`: RL 환경용, `step()` → `(obs, reward, terminated, truncated, info)`

---

**함정 4: 관찰 함수 시그니처**

❌ **잘못된 함수:**
```python
def my_observation() -> torch.Tensor:  # env 인자 없음
    return torch.zeros(2, 3)
```

✅ **올바른 함수:**
```python
def my_observation(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """관찰 함수는 env를 첫 번째 인자로 받음"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins
```

**교훈:** MDP 함수는 항상 `env`를 첫 번째 인자로 받음.

---

**함정 5: prim_path 패턴 혼동**

```python
# 전역 경로 (모든 환경이 공유)
terrain = TerrainImporterCfg(prim_path="/World/ground")
light = AssetBaseCfg(prim_path="/World/light")

# 환경별 경로 (각 환경마다 복제)
cube = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/cube")
# → /World/envs/env_0/cube, /World/envs/env_1/cube, ...
```

**교훈:** `{ENV_REGEX_NS}`는 환경별로 다른 객체가 필요할 때 사용.

---

**함정 6: print문 남발로 코드 흐름 가독성 저하**

❌ **비효율적:**
```python
print("=" * 80)
print("[정보] Environment 생성 중...")
print("=" * 80)
print(f"환경 개수: {num_envs}")
print(f"환경 간격: {spacing}m")
# ... 수십 줄의 print문
```

✅ **간결하게:**
```python
print(f"Environment 생성 완료 | Scene entities: {list(env.scene.keys())}")
print(f"Reset 완료 | Observation groups: {list(obs.keys())}\n")
```

**교훈:** 핵심 정보만 출력하여 코드 흐름 명확하게 유지.

### Import 정렬 순서 (isort)

```python
# 1. FUTURE
from __future__ import annotations

# 2. STDLIB
import os
import sys

# 3. THIRDPARTY (numpy, torch, gymnasium, Isaac Sim 등)
import torch
import numpy as np
from omni.isaac.core.api import World

# 4. ASSETS_FIRSTPARTY
import isaaclab_assets

# 5. FIRSTPARTY
import isaaclab
from isaaclab.envs import ManagerBasedEnv

# 6. EXTRA_FIRSTPARTY
import isaaclab_rl
import isaaclab_mimic

# 7. TASK_FIRSTPARTY
import isaaclab_tasks

# 8. LOCALFOLDER
from config import MyConfig
```

### 타입 체크 (Pyright)

- 모드: `basic`
- Python 버전: 3.10
- `reportMissingImports = "none"` (CI 환경에서 모듈 미설치)
- `reportGeneralTypeIssues = "none"` (dataclass MISSING 리터럴)

### 코드 스타일

- **라인 길이:** 120자
- **포매터:** Black
- **린터:** Flake8 + simplify + return
- **타입 힌팅:** 가능한 한 사용 (특히 공개 API)

### 라이선스 헤더

모든 새 파일에 라이선스 헤더 추가 (pre-commit hook 자동 삽입):
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

**isaaclab_mimic의 경우:**
```python
# SPDX-License-Identifier: Apache-2.0
```

## 고급 사용법

### 새 환경 생성

```bash
# 템플릿 사용
cp -r tools/template/template_env/ my_project/
# 환경 설정 클래스 정의
# Gymnasium에 환경 등록
# 학습 스크립트 작성
```

### Docker 사용

```bash
# Docker 이미지 빌드
docker/container.py start

# ROS2 통합 이미지
docker compose -f docker/docker-compose.yaml up ros2
```

### 클러스터 배포

Ray RLlib을 사용한 분산 학습 지원 (docs/deployment/ 참조).

## 주요 의존성

| 패키지 | 버전 | 용도 | 비고 |
|--------|------|------|------|
| PyTorch | 2.7.0+cu128 | 신경망, 텐서 연산 (CUDA 12.8) | |
| Gymnasium | >=1.0 | RL 환경 API | |
| NumPy | 1.26.0 | 수치 계산 | **<2.0 제약** (Isaac Sim 호환성) |
| SciPy | 1.11.4 | 과학 계산 | **1.15.3에서 다운그레이드 필수** |
| ONNX | 1.16.1 | 모델 변환 | |
| Warp | warp-lang | GPU 커널 최적화 | |
| libstdcxx-ng | >=12 (15.2.0) | C++ 표준 라이브러리 | GLIBCXX_3.4.30+ 지원 |

## 문제 해결

### ⚠️ SciPy/NumPy 호환성 에러 (Critical)

**증상:**
```
ValueError: All ufuncs must have type `numpy.ufunc`.
Received (<ufunc 'sph_legendre_p'>, ...)
```

**영향을 받는 모듈:**
- `isaacsim.replicator.grasping`
- `isaacsim.util.camera_inspector`
- `isaacsim.ros2.bridge`

**원인:**
- Isaac Sim 5.0/5.1이 SciPy 1.15.3을 설치하지만, 이 버전은 NumPy 2.x 전환용
- NumPy 1.26.0과 호환되지 않아 ufunc 타입 검사 실패

**해결:**
```bash
pip install "scipy==1.11.4"
```

**검증:**
```bash
python -c "import scipy; print(scipy.__version__)"  # 1.11.4 출력되어야 함
```

**참고:** `isaacsim-core`가 scipy==1.15.3을 요구한다는 경고가 나타나지만 무시해도 됨. 1.11.4가 실제로 더 안정적.

---

### ⚠️ 의존성 충돌 경고

설치 중 다음과 같은 의존성 충돌 경고가 나타날 수 있습니다:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
isaacsim-kernel 5.0.0.0 requires sentry-sdk==1.43.0, but you have sentry-sdk 2.42.1
```

**영향:**
- 대부분의 경우 실제 동작에는 문제가 없습니다
- Isaac Lab의 RL 라이브러리들이 더 최신 버전의 sentry-sdk를 요구합니다
- Isaac Sim의 에러 리포팅 기능에만 영향을 줄 수 있습니다

**해결 (선택사항):**
```bash
# 1. Isaac Sim 기능 우선 (에러 리포팅 필요 시)
pip install "sentry-sdk==1.43.0"

# 2. Isaac Lab 기능 우선 (RL 학습 안정성 우선)
# 현재 상태 유지 (권장)
```

**권장:** 실제 학습/시뮬레이션에 문제가 없다면 현재 상태 유지를 권장합니다.

---

### ⚠️ GLIBCXX_3.4.30 에러 (Critical)

**증상:**
```
OSError: version 'GLIBCXX_3.4.30' not found
(required by /path/to/omni/libcarb.so)
```

**원인:**
- Isaac Sim 네이티브 라이브러리(`libcarb.so`)가 GLIBCXX_3.4.30 (GCC 12+) 필요
- Conda 환경의 기본 libstdc++는 GLIBCXX_3.4.29 (GCC 11)까지만 지원
- torch 또는 tensorboard를 `AppLauncher` 전에 import하면 발생

**해결 방법 1: C++ 라이브러리 업그레이드 (권장)**

```bash
conda config --env --set channel_priority strict
conda config --env --add channels conda-forge
conda install -y -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"
```

**검증:**
```bash
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep "^GLIBCXX" | sort -V | tail -1
# GLIBCXX_3.4.34 (또는 3.4.30 이상) 출력되어야 함
```

**해결 방법 2: Import 순서 변경**

```python
# ❌ 잘못된 순서
import torch
from isaaclab.app import AppLauncher

# ✅ 올바른 순서
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
# AppLauncher 생성 후 torch import
import torch
```

**공식 문서 참조:**
- `docs/source/refs/issues.rst` (96-104줄): Known Issues
- `docs/source/overview/imitation-learning/skillgen.rst` (113-119줄): SkillGen 설치

---

### ⚠️ Python 경로 문제 (환경 이름)

**증상:**
```
[ERROR] Unable to find any Python executable at path: '/path/to/conda/envs/CUSTOM_NAME/bin/python'
```

**원인:**
- Isaac Lab의 `isaaclab.sh` 스크립트는 특정 환경 이름을 우선 순위로 검색합니다
- 기본 우선순위: `env_isaaclab` > `isaaclab` > 기타

**해결:**
```bash
# 방법 1: 권장 환경 이름 사용 (권장)
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

# 방법 2: 심볼릭 링크 생성 (기존 환경 유지하려는 경우)
cd $HOME/miniconda3/envs
ln -s your_custom_env env_isaaclab

# 방법 3: Python 경로 직접 지정
export PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
./isaaclab.sh --install
```

**검증:**
```bash
conda activate env_isaaclab
./isaaclab.sh -p --version  # Python 버전 확인
```

---

### Isaac Sim 경로 오류
```bash
# _isaac_sim 심볼릭 링크 확인
ls -la _isaac_sim
# Conda 환경 활성화 확인
conda activate env_isaaclab
```

### 메모리 부족
```bash
# 환경 수 줄이기
--num_envs 256  # 기본값 4096에서 감소
```

### GPU 메모리 오류
```bash
# 헤드리스 모드로 렌더링 비활성화
--headless
```

### 설치 검증

Isaac Lab이 정상적으로 설치되었는지 확인:

```bash
# 1. Conda 환경 활성화 확인
conda activate env_isaaclab
which python  # /home/USERNAME/miniconda3/envs/env_isaaclab/bin/python

# 2. 패키지 버전 확인
pip list | grep -E "(isaaclab|scipy|numpy|torch)"
# NumPy: 1.26.0
# SciPy: 1.11.4 (중요!)
# torch: 2.7.0+cu128
# isaacsim: 5.1.0.0
# isaaclab: 0.47.4
# isaaclab-assets: 0.2.3
# isaaclab-mimic: 1.0.15
# isaaclab-rl: 0.4.4
# isaaclab-tasks: 0.11.6

# 3. C++ 라이브러리 확인 (GLIBCXX 3.4.30+ 필요)
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep "^GLIBCXX" | sort -V | tail -1
# GLIBCXX_3.4.30 이상 출력되어야 함

# 4. 환경 목록 실행 (가벼운 테스트)
./isaaclab.sh -p scripts/environments/list_envs.py
# 200+ 환경이 에러 없이 표시되어야 함

# 5. 간단한 시뮬레이션 실행 (헤드리스 모드)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# 성공하면 "Simulation App Startup Complete" 메시지 출력
```

**주의사항:**
- SciPy 버전이 1.15.3이면 반드시 1.11.4로 다운그레이드 필요
- GLIBCXX 버전이 3.4.30 미만이면 libstdcxx-ng 업그레이드 필요
- 첫 실행 시 shader 컴파일로 인해 시간이 걸릴 수 있음

## 리소스

- **문서:** https://isaac-sim.github.io/IsaacLab
- **GitHub Issues:** 실행 가능한 작업, 버그 리포트
- **GitHub Discussions:** 아이디어, 질문
- **Discord:** Omniverse Discord (NVIDIA)
- **기여 가이드:** docs/refs/contributing.html

## 인용

Isaac Lab은 원래 Orbit 프레임워크에서 시작되었습니다:
```bibtex
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and ...},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   journal={IEEE Robotics and Automation Letters},
   year={2023}
}
```

---

## 변경 이력

- **2025-11-03 (업데이트 4)**: Phase 2 학습 경험 반영
  - 개발 가이드라인에 "실전 경험: 주요 함정과 해결책" 섹션 추가
  - ManagerBasedEnv vs DirectRLEnv 혼동 문제와 해결책
  - ObservationManager 빈 그룹 에러 해결 방법
  - ManagerBasedEnv의 step() 반환값 차이 명확화
  - 관찰 함수 시그니처, prim_path 패턴, 코드 가독성 개선 팁
  - `scripts/tutorials/03_envs/my_clone/` TDD 방식 클론 코딩 학습 진행 중

- **2025-10-31 (업데이트 3)**: Isaac Sim 5.1.0 실제 설치 경험 반영
  - Isaac Sim 5.0.0 → 5.1.0으로 권장 버전 변경
  - 실제 설치된 패키지 버전 업데이트 (Isaac Lab 0.47.4 기준)
  - SciPy 1.15.3 호환성 문제가 5.1에도 동일하게 적용됨 확인
  - libstdcxx-ng 15.2.0 (GLIBCXX_3.4.34) 검증 완료

- **2025-10-28 (업데이트 2)**: 환경 설정 실전 경험 반영
  - 권장 Conda 환경 이름을 `env_isaaclab`로 명시
  - Python 경로 문제 해결 방법 추가 (3가지 방법)
  - 설치 검증 섹션 강화 (5단계 체크리스트)
  - Isaac Sim 5.0 설치 소요 시간 정보 추가
  - 패키지 버전 목록 업데이트 (실제 설치 버전 기준)

- **2025-10-28 (업데이트 1)**: SciPy/GLIBCXX 호환성 문제 해결 방법 추가
  - 초기 설정에 호환성 문제 해결 단계 추가 (5단계)
  - 주요 의존성 테이블에 SciPy 1.11.4 및 libstdcxx-ng 추가
  - 문제 해결 섹션에 SciPy/NumPy 호환성 에러 상세 설명
  - GLIBCXX_3.4.30 에러에 대한 2가지 해결 방법 추가
  - 설치 검증 절차 추가

*이 문서는 Isaac Lab v2.3.0 + Isaac Sim 5.1 기준으로 작성되었습니다.*
*실제 설치 및 학습 경험을 바탕으로 작성되어 실전 환경에서 검증되었습니다.*
