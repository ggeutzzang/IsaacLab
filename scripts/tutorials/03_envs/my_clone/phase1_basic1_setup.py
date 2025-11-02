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
# 1. Isaac Sim의 C++ 라이브러리 경로가 설정됩니다P
# 2. GLIBCXX_3.4.30 같은 최신 C++ 표준 라이브러리가 로드됩니다
# 3. 이후 torch를 import해도 충돌이 발생하지 않습니다
#
# 만약 torch를 먼저 import하면:
# - torch가 시스템의 오래된 libstdc++를 먼저 로드
# - Isaac Sim이 최신 버전을 요구하면 충돌 발생
# - "GLIBCXX_3.4.30 not found" 에러 `발생
# ────────────────────────────────────────────────────────────


import torch
import time 


def main():

	print("-" * 80)
	
	# 시뮬레이션 루프
	print("\n[시작] 시뮬레이션 루프 시작 (100 스텝 실행)")
	print("Ctrl+C를 누르면 종료됩니다.\n")
	
	count = 0
	max_steps = 100
	
	# 성능 측정 변수
	start_time = time.time()
	step_times = []

	while simulation_app.is_running() and count < max_steps:
		# 스텝 시작 시간 기록
		step_start = time.time()

		# 시뮬레이션 스텝 실행
		simulation_app.update()

		# # (테스트) 매 스텝마다 0.1초 대기 (느리게 보기)
		# time.sleep(0.1)

		# 스텝 종료 시간 기록
		step_end = time.time()
		step_duration = step_end - step_start
		step_times.append(step_duration)
		
		# 10 스텝마다 진행 상황 출력
		if count % 10 == 0:
			print(f"[스텝 {count:03d}] 시뮬레이션 진행 중... (최근 스텝: {step_duration*1000:.2f}ms)")
		
		count += 1
	
	# 전체 실행 시간
	end_time = time.time()
	total_time = end_time - start_time

	print("\n" + "=" * 80)
	print("[완료] 시뮬레이션 성능 통계")
	print("=" * 80)
	
	# 기본 정보
	print(f"총 스텝 수: {count}개")
	print(f"총 실행 시간: {total_time:.3f}초")
	
	# FPS 계산
	avg_fps = count / total_time
	print(f"평균 FPS: {avg_fps:.2f} frames/sec")
	
	# 스텝당 시간 통계
	import statistics
	avg_step_time = statistics.mean(step_times) * 1000  # ms 단위
	min_step_time = min(step_times) * 1000
	max_step_time = max(step_times) * 1000
	std_step_time = statistics.stdev(step_times) * 1000
	
	print(f"\n[스텝당 시간 통계 (ms)]")
	print(f"  평균: {avg_step_time:.2f}ms")
	print(f"  최소: {min_step_time:.2f}ms")
	print(f"  최대: {max_step_time:.2f}ms")
	print(f"  표준편차: {std_step_time:.2f}ms")
	
	# 실시간 여부 판단 (60Hz 기준)
	target_step_time = 1000 / 60  # 16.67ms
	if avg_step_time <= target_step_time:
		print(f"\n✅ 실시간 시뮬레이션 가능! (목표: {target_step_time:.2f}ms)")
	else:
		slowdown = avg_step_time / target_step_time
		print(f"\n⚠️ 실시간보다 {slowdown:.2f}배 느림 (목표: {target_step_time:.2f}ms)")
	
	print("=" * 80)
	print("[정보] Phase 1 테스트 성공! ✓")

if __name__ == "__main__":
	main()
	simulation_app.close()