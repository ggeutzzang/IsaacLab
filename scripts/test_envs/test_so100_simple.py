#!/usr/bin/env python3
"""
간단한 SO-100 환경 테스트 스크립트
"""

from isaaclab.app import AppLauncher

# Isaac Sim 런타임 초기화 (헤드리스 모드)
app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg


def test_so100_simple():
    """간단한 SO-100 환경 테스트"""
    
    print("=== 간단한 SO-100 환경 테스트 시작 ===")
    
    # 카메라가 없는 환경 사용
    env_id = "Isaac-Stack-Cube-SO100-v0"
    print(f"환경 ID: {env_id}")
    
    try:
        # 환경 설정 파싱
        env_cfg = parse_env_cfg(env_id)
        
        # 환경 수를 1개로 설정
        env_cfg.scene.num_envs = 1
        
        # 환경 생성
        env = gym.make(env_id, cfg=env_cfg)
        print("✅ 환경 생성 성공")
        
        # 환경 정보 출력
        print(f"관찰 공간: {env.observation_space}")
        print(f"행동 공간: {env.action_space}")
        
        # 환경 리셋
        obs, info = env.reset()
        print(f"초기 관찰 키: {list(obs.keys())}")
        
        # 간단한 행동 테스트
        action = env.action_space.sample()
        # numpy 배열을 torch 텐서로 변환
        import torch
        action_tensor = torch.from_numpy(action).float()
        obs, reward, terminated, truncated, info = env.step(action_tensor)
        print(f"보상: {reward}")
        
        # 환경 종료
        env.close()
        print("✅ 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_so100_simple()
    simulation_app.close() 