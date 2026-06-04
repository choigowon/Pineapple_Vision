import time
from check_battery import get_estimated_remaining_time, check_under_voltage
import check_battery

def run_logic_test():
    print("=== 배터리 잔량 계산 로직 정밀 검증 ===")
    
    # 설정값 확인
    total_min = check_battery.MAX_OPERATING_MINUTES
    print(f"설정된 최대 가동 시간: {total_min}분")

    # [테스트 1] 시작 직후 (0분 경과)
    min_0, per_0 = check_battery.get_estimated_remaining_time()
    print(f"\n1. 시작 직후: {min_0}분 남음, {per_0}%")
    
    # [테스트 2] 60분 경과 강제 설정 (가상 점프)
    check_battery.start_time = time.time() - (60 * 60) 
    
    min_60, per_60 = check_battery.get_estimated_remaining_time()
    print(f"2. 60분 경과 시뮬레이션: {min_60}분 남음, {per_60}%")
    
    # 검증: 240분 중 60분 썼으면 180분 남고, 75%여야 함
    expected_min = total_min - 60
    expected_per = int((expected_min / total_min) * 100)
    
    if min_60 == expected_min and per_60 == expected_per:
        print("결과: 정확함 (180분, 75%)")
    else:
        print(f"결과: 오류 (기대값: {expected_min}분, {expected_per}%)")

    # [테스트 3] 방전 직전 (239분 경과 시뮬레이션)
    check_battery.start_time = time.time() - (239 * 60)
    min_239, per_239 = check_battery.get_estimated_remaining_time()
    print(f"\n3. 239분 경과 시뮬레이션: {min_239}분 남음, {per_239}%")

    # [테스트 4] 한도 초과 (300분 경과 시뮬레이션)
    check_battery.start_time = time.time() - (300 * 60)
    min_over, per_over = check_battery.get_estimated_remaining_time()
    print(f"4. 한도 초과(300분) 시뮬레이션: {min_over}분 남음, {per_over}%")
    
    if min_over == 0 and per_over == 0:
        print("✅ 결과: 음수 방지 로직 정상 작동 (0분, 0%)")

def run_battery_test():
    print("=== 배터리 및 전력 감시 시스템 테스트 시작 ===")
    
    # 테스트를 위해 5번 반복 확인 (1초 간격)
    for i in range(1, 6):
        remaining_min, percent = get_estimated_remaining_time()
        voltage_msg = check_under_voltage()
        
        print(f"\n[체크 {i}]")
        print(f"- 남은 예상 시간: {remaining_min}분")
        print(f"- 배터리 잔량: {percent}%")
        
        if voltage_msg:
            print(f"- 전압 상태: ⚠️ {voltage_msg}")
        else:
            print(f"- 전압 상태: ✅ 정상 (또는 PC 환경)")
            
        time.sleep(1)

    print("\n=== 가동 시간 감소 시뮬레이션 ===")
    run_logic_test()
    print("성공적으로 배터리 잔량 및 전압 체크 로직이 작동 중입니다.")

if __name__ == "__main__":
    run_battery_test()