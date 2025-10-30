"""
Phase 1: Basic Setup and Empty Environment

This script demonstrates:
1. How to launch Isaac Sim using AppLauncher
2. Basic import structure
3. Creating an empty simulation that runs

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/my_clone/phase1_basic_setup.py --num_envs 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher 

# argparse 인자 추가 
parser = argparse.ArgumentParser(description="Phase1: 기초 설정 및 빈 환경 테스트")
parser.add_argument("--num_envs", type=int, default=2, help="생성할 환경의 개수")

# AppLauncher의 CLI 인자 추가 
AppLauncher.add_app_launcher_args(parser)

# 인자 파싱 
args_cli = parser.parse_args()

# Omniverse 기동 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

  # ★ 학습 포인트 ────────────────────────────────────────────────
  # 왜 torch를 AppLauncher 이후에 import 하는가?
  #
  # Isaac Sim은 내부적으로 CUDA 라이브러리를 사용합니다.
  # torch도 CUDA를 사용하므로, 두 라이브러리의 로딩 순서가 중요합니다.
  #
  # AppLauncher가 먼저 실행되면:
  # 1. Isaac Sim의 C++ 라이브러리 경로가 설정됩니다
  # 2. GLIBCXX_3.4.30 같은 최신 C++ 표준 라이브러리가 로드됩니다
  # 3. 이후 torch를 import해도 충돌이 발생하지 않습니다
  #
  # 만약 torch를 먼저 import하면:
  # - torch가 시스템의 오래된 libstdc++를 먼저 로드
  # - Isaac Sim이 최신 버전을 요구하면 충돌 발생
  # - "GLIBCXX_3.4.30 not found" 에러 발생
  # ────────────────────────────────────────────────────────────

import torch 
 
def main():
    print("-" * 80)
    print("[정보] Phase1: 기초 설정 테스트")
    print(f"[정보] PyTorch 버전 : {torch.__version__}")
    print(f"[정보] CUDA 사용 가능 : {torch.cuda.is_available()}")
    print(f"[정보] 환경 개수 : {args_cli}")