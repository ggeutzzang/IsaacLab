# Isaac Lab 튜토리얼 Phase 별 학습 가이드

이 디렉토리는 Isaac Lab의 Manager-Based Environment를 단계적으로 학습하기 위한 클론 코딩 프로젝트입니다.

## 📚 Phase 개요

| Phase | 파일명 | 학습 목표 | 실행 명령 |
|-------|--------|----------|-----------|
| Phase 2 | `phase2_scene.py` | Scene 설정 (지형 + 조명) | `./isaaclab.sh -p scripts/tutorials/03_envs/my_clone/phase2_scene.py --num_envs 2` |
| Phase 3 | `phase3_managers.py` | ObservationManager 활용 | `./isaaclab.sh -p scripts/tutorials/03_envs/my_clone/phase3_managers.py --num_envs 2` |

---

## Phase 2: Scene 설정 기초

### 학습 내용
- ✅ **InteractiveSceneCfg** 상속하여 Scene 구성
- ✅ **TerrainImporterCfg**로 지형 추가
- ✅ **AssetBaseCfg + DomeLightCfg**로 조명 추가
- ✅ **ManagerBasedEnv** 기본 구조 학습

### 주요 코드
```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 지형 추가
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    # 조명 추가
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )
```

### Scene Entities
- `terrain`: 평평한 바닥
- `light`: 돔 라이트 (전역 조명)

### Observations
- **더미 관찰값**: `dummy_observation()` → 상수 0.0 반환
- **Shape**: `(num_envs, 1)`

---

## Phase 3: Manager 시스템 학습

### Phase 2 대비 추가/변경 사항

#### 1️⃣ Scene에 RigidObject 추가 ✨

```python
# Phase 2: 지형 + 조명만
scene: MySceneCfg = MySceneCfg(...)

# Phase 3: 지형 + 조명 + 큐브
@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(...)  # 동일
    light = AssetBaseCfg(...)          # 동일

    # ★ 신규: 큐브 객체 추가
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",  # 환경별 인스턴스 생성
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
```

**핵심 개념:**
- `{ENV_REGEX_NS}`: 각 환경마다 별도 큐브 생성 (`/World/envs/env_0/cube`, `/World/envs/env_1/cube`, ...)
- `RigidObjectCfg`: 물리 시뮬레이션이 적용되는 강체 객체
- `init_state`: 초기 위치 z=1.0m (중력으로 낙하 예정)

#### 2️⃣ 실제 MDP 함수 구현 ✨

```python
# Phase 2: 더미 함수
def dummy_observation(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, 1, device=env.device)

# Phase 3: 실제 큐브 상태 추적
def cube_position(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """큐브의 위치를 환경 원점 기준 상대 좌표로 반환"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins

def cube_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """큐브의 선속도 반환"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w
```

**핵심 개념:**
- **MDP 함수**: `env`와 `asset_cfg`를 인자로 받아 텐서 반환
- **SceneEntityCfg**: Scene의 객체를 이름으로 참조
- **재사용성**: 다른 객체에도 동일한 함수 적용 가능

#### 3️⃣ ObservationManager 실전 활용 ✨

```python
# Phase 2: 더미 관찰값
@configclass
class PolicyCfg(ObservationGroupCfg):
    dummy_obs = ObservationTermCfg(func=dummy_observation)

# Phase 3: 실제 큐브 상태
@configclass
class PolicyCfg(ObservationGroupCfg):
    # 큐브 위치 (3차원)
    cube_pos = ObservationTermCfg(
        func=cube_position,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )

    # 큐브 속도 (3차원)
    cube_vel = ObservationTermCfg(
        func=cube_velocity,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
```

**핵심 개념:**
- **ObservationTermCfg**: MDP 함수와 파라미터를 연결
- **SceneEntityCfg("cube")**: Scene의 "cube" 객체를 함수에 전달
- **자동 연결**: ObservationManager가 매 step마다 자동으로 함수 호출

#### 4️⃣ 관찰값 Shape 변화 ✨

```python
# Phase 2:
# obs['policy'].shape = (num_envs, 1)
# 값: [0.0] (더미)

# Phase 3:
# obs['policy'].shape = (num_envs, 6)
# 값: [x, y, z, vx, vy, vz] (큐브 위치 + 속도)
```

#### 5️⃣ 실행 결과 비교 ✨

**Phase 2:**
```
Environment 생성 완료 | Scene entities: ['terrain', 'light']
Reset 완료 | Observation groups: ['policy']
(200 스텝 실행, 관찰값 변화 없음)
```

**Phase 3:**
```
Environment 생성 완료 | Scene entities: ['terrain', 'cube', 'light']
Reset 완료 | Observation groups: ['policy']

Step   0 | Env 0 큐브 위치 z=0.973m, 속도 vz=-0.654m/s  ← 중력 낙하 시작
Step  10 | Env 0 큐브 위치 z=0.182m, 속도 vz=-3.924m/s  ← 가속 중
Step  20 | Env 0 큐브 위치 z=0.100m, 속도 vz=-0.000m/s  ← 바닥 충돌
Step  30 | Env 0 큐브 위치 z=0.100m, 속도 vz=-0.000m/s  ← 정지
```

---

## 💡 핵심 학습 포인트

### Configuration-Driven Architecture
```
사용자: Cfg 클래스 정의 (선언적)
   ↓
프레임워크: 자동으로 객체 생성 및 초기화
   ↓
Runtime: Manager가 자동으로 함수 호출 및 상태 업데이트
```

### Manager 시스템 계층 구조
```
EnvCfg (ManagerBasedEnvCfg)
├── scene: MySceneCfg (InteractiveSceneCfg)
│   ├── terrain: TerrainImporterCfg
│   ├── light: AssetBaseCfg
│   └── cube: RigidObjectCfg  ← Phase 3 추가
├── observations: ObservationsCfg
│   └── policy: PolicyCfg
│       ├── cube_pos: ObservationTermCfg  ← Phase 3 추가
│       └── cube_vel: ObservationTermCfg  ← Phase 3 추가
├── actions: ActionsCfg (빈 클래스)
└── events: EventCfg (빈 클래스)
```

### MDP 함수의 재사용성
```python
# 동일한 함수를 다른 객체에 적용 가능
cube_pos = ObservationTermCfg(
    func=cube_position,  # 재사용 가능한 함수
    params={"asset_cfg": SceneEntityCfg("cube")},  # 큐브에 적용
)

robot_pos = ObservationTermCfg(
    func=cube_position,  # 동일한 함수
    params={"asset_cfg": SceneEntityCfg("robot")},  # 로봇에 적용
)
```

---

## 🚀 다음 단계

**Phase 4 (예정)**: ManagerBasedRLEnv로 확장
- RewardManager: 보상 함수 추가
- TerminationManager: Episode 종료 조건
- RL 환경으로 완전한 전환

**Phase 5 (예정)**: ActionManager 활용
- 큐브에 힘 적용하는 ActionTerm 구현
- 액션-보상 루프 완성

---

## 🔍 디버깅 팁

### Scene entities 확인
```python
print(list(env.scene.keys()))
# Phase 2: ['terrain', 'light']
# Phase 3: ['terrain', 'cube', 'light']
```

### Observation shape 확인
```python
print(obs['policy'].shape)
# Phase 2: torch.Size([2, 1])
# Phase 3: torch.Size([2, 6])
```

### 큐브 데이터 직접 접근
```python
cube = env.scene["cube"]
print(cube.data.root_pos_w)  # 월드 좌표계 위치
print(cube.data.root_lin_vel_w)  # 선속도
```

---

## 📖 참고 자료

- **Isaac Lab 공식 문서**: https://isaac-sim.github.io/IsaacLab
- **CLAUDE.md**: 프로젝트 루트의 상세 설정 가이드
- **원본 튜토리얼**: `source/standalone/tutorials/03_envs/`
