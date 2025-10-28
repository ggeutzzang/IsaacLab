# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**Isaac Lab**은 NVIDIA Isaac Sim 기반의 GPU 가속 로봇 학습 프레임워크입니다. 강화학습(RL), 모방학습(IL), 모션 플래닝 등의 워크플로우를 통합하며, 벡터화된 환경과 정확한 물리/센서 시뮬레이션을 제공합니다.

- **버전:** 2.1.0 (Isaac Sim 4.5 호환)
- **Python:** 3.10+
- **라이선스:** BSD-3-Clause (메인), Apache-2.0 (isaaclab_mimic)

## 환경 설정 및 빌드

### 초기 설정
```bash
# 1. Conda 환경 생성 및 활성화
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab

# 2. PyTorch 설치 (CUDA 11.8 또는 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# 3. Isaac Sim 설치
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# 4. Isaac Lab 설치
./isaaclab.sh --install  # Linux
# or
isaaclab.bat --install   # Windows
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

### 2. 설정 기반 아키텍처

모든 환경은 `@configclass` 데코레이터로 정의된 설정 클래스 사용:
```python
@configclass
class MyEnvCfg:
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # ... 기타 매니저 설정
```

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
```

### 5. 관찰값 구조

다중 관찰값 그룹 지원 (비대칭 액터-크리틱, 다중 에이전트):
```python
VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
# 예: {"policy": tensor([...]), "critic": tensor([...])}
```

## 개발 가이드라인

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

| 패키지 | 버전 | 용도 |
|--------|------|------|
| PyTorch | 2.5.1 | 신경망, 텐서 연산 |
| Gymnasium | >=1.0 | RL 환경 API |
| NumPy | <2 | 수치 계산 |
| ONNX | 1.16.1 | 모델 변환 |
| Warp | warp-lang | GPU 커널 최적화 |

## 문제 해결

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
