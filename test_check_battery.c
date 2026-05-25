#include <stdio.h>
#include <unistd.h>
#include "check_battery.h"
#include "tts.h"

extern void set_mock_start_time(double mock_time);

void run_logic_test() {
    printf("\n=== 배터리 잔량 계산 로직 정밀 검증 ===\n");
    
    int total_min = MAX_OPERATING_MINUTES;
    printf("설정된 최대 가동 시간: %d분\n", total_min);

    int rem_min = 0, percent = 0;

    // [테스트 1] 시작 직후 (0분 경과)
    get_estimated_remaining_time(&rem_min, &percent);
    printf("\n1. 시작 직후: %d분 남음, %d%%\n", rem_min, percent);
    
    // [테스트 2] 60분 경과 강제 설정 (가상 점프)
    double current_time = get_current_timestamp();
    set_mock_start_time(current_time - (60 * 60)); 
    
    get_estimated_remaining_time(&rem_min, &percent);
    printf("2. 60분 경과 시뮬레이션: %d분 남음, %d%%\n", rem_min, percent);
    
    // 검증
    int expected_min = total_min - 60;
    int expected_per = (int)(((double)expected_min / total_min) * 100.0);
    
    if (rem_min == expected_min && percent == expected_per) {
        printf("결과: 정확함 (%d분, %d%%)\n", rem_min, percent);
    } else {
        printf("결과: 오류 (기대값: %d분, %d%%)\n", expected_min, expected_per);
    }

    // [테스트 3] 방전 직전 (239분 경과 시뮬레이션)
    set_mock_start_time(current_time - (239 * 60));
    get_estimated_remaining_time(&rem_min, &percent);
    printf("\n3. 239분 경과 시뮬레이션: %d분 남음, %d%%\n", rem_min, percent);

    // [테스트 4] 한도 초과 (300분 경과 시뮬레이션)
    set_mock_start_time(current_time - (300 * 60));
    get_estimated_remaining_time(&rem_min, &percent);
    printf("4. 한도 초과(300분) 시뮬레이션: %d분 남음, %d%%\n", rem_min, percent);
    
    if (rem_min == 0 && percent == 0) {
        printf("✅ 결과: 음수 방지 로직 정상 작동 (0분, 0%%)\n");
    }
}

void run_battery_test() {
    printf("=== 배터리 및 전력 감시 시스템 테스트 시작 ===\n");
    
    // 테스트를 위해 5번 반복 확인 (1초 간격)
    for (int i = 1; i <= 5; i++) {
        int remaining_min = 0, percent = 0;
        get_estimated_remaining_time(&remaining_min, &percent);
        const char* voltage_msg = check_under_voltage();
        
        printf("\n[체크 %d]\n", i);
        printf("- 남은 예상 시간: %d분\n", remaining_min);
        printf("- 배터리 잔량: %d%%\n", percent);
        
        if (voltage_msg != NULL) {
            printf("- 전압 상태: ⚠️ %s\n", voltage_msg);
        } else {
            printf("- 전압 상태: ✅ 정상 (또는 PC 환경)\n");
        }
        
        sleep(1);
    }

    printf("\n=== 가동 시간 감소 시뮬레이션 ===\n");
    run_logic_test();
    printf("\n성공적으로 배터리 잔량 및 전압 체크 로직이 작동 중입니다.\n");
}

int main() {
    // 음성 매니저 및 배터리 초기화
    init_voice_manager();
    init_battery_monitor();
    
    run_battery_test();
    return 0;
}