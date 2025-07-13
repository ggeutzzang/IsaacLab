#!/usr/bin/env python3
"""
SO-100 Mimic 환경 테스트 스크립트
Isaac-Stack-SO100-Abs-Mimic-v0 환경을 테스트합니다.
"""

from isaaclab.app import AppLauncher

# Isaac Sim 런타임 초기화 (카메라 활성화)
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from isaaclab_mimic.envs.pinocchio_envs import StackSO100MimicEnv
from isaaclab_tasks.utils import parse_env_cfg


def test_so100_mimic_env():
    """SO-100 Mimic 환경을 테스트합니다."""
    
    print("=== SO-100 Mimic 환경 테스트 시작 ===")
    
    # 환경 생성
    env_id = "Isaac-Stack-SO100-Abs-Mimic-v0"
    print(f"환경 ID: {env_id}")
    
    try:
        # 환경 설정 파싱
        env_cfg = parse_env_cfg(env_id)
        
        # 환경 수를 1개로 설정
        env_cfg.scene.num_envs = 1
        
        # 환경 생성 (cfg 인자 전달)
        env = gym.make(env_id, cfg=env_cfg)
        print("✅ 환경 생성 성공")
        
        # 환경 정보 출력
        print(f"\n=== 환경 정보 ===")
        print(f"관찰 공간: {env.observation_space}")
        print(f"행동 공간: {env.action_space}")
        print(f"환경 ID: {env.unwrapped.spec.id}")
        
        # 환경 리셋
        print(f"\n=== 환경 리셋 ===")
        obs, info = env.reset()
        print(f"초기 관찰 키: {list(obs.keys())}")
        print(f"초기 정보 키: {list(info.keys())}")
        
        # 관찰 데이터 구조 확인
        print(f"\n=== 관찰 데이터 구조 ===")
        for key, value in obs.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}: {sub_value.shape} ({sub_value.dtype})")
                    else:
                        print(f"    {sub_key}: {type(sub_value)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 행동 공간 확인
        print(f"\n=== 행동 공간 상세 정보 ===")
        action_space = env.action_space
        print(f"행동 차원: {action_space.shape}")
        print(f"행동 타입: {action_space.dtype}")
        print(f"행동 공간 타입: {type(action_space)}")
        
        # 랜덤 행동으로 몇 스텝 실행
        print(f"\n=== 랜덤 행동 테스트 ===")
        for step in range(5):
            # 랜덤 행동 생성 (numpy array를 tensor로 변환)
            action_numpy = env.action_space.sample()
            action = torch.tensor(action_numpy, dtype=torch.float32)
            print(f"스텝 {step + 1}: 행동 shape = {action.shape}")
            
            # 환경 스텝 실행
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  보상: {reward}")
            print(f"  종료: {terminated}")
            print(f"  잘림: {truncated}")
            
            if terminated or truncated:
                print("  환경이 종료되었습니다.")
                obs, info = env.reset()
                break
        
        # 환경 종료
        env.close()
        print("\n✅ 환경 테스트 완료")
        
    except Exception as e:
        print(f"❌ 환경 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Mimic 환경 테스트만 실행
    test_so100_mimic_env()
    
    print("\n=== 테스트 완료 ===")
    simulation_app.close() 