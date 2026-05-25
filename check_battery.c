#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "check_battery.h"
#include "tts.h" // 음성 매니저(speak) 헤더 연결

// 프로그램 시작 시각을 기록할 전역 변수 (이 파일 내부에서만 접근하도록 static 설정)
static double start_time = 0.0;

// 배터리 모니터 초기화 (시작 시간 마킹)
void init_battery_monitor() {
    start_time = get_current_timestamp();
}

// 현재 시간(초)을 구하는 헬퍼 함수
double get_current_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + (ts.tv_nsec / 1000000000.0);
}

// 1. 가동 시간 기반 카운트다운 함수
void get_estimated_remaining_time(int *remaining_minutes, int *percent) {
    double current_time = get_current_timestamp();
    double elapsed_seconds = current_time - start_time;
    int elapsed_minutes = (int)(elapsed_seconds / 60.0);
    
    *remaining_minutes = MAX_OPERATING_MINUTES - elapsed_minutes;
    
    if (*remaining_minutes < 0) {
        *remaining_minutes = 0;
    }
    
    *percent = (int)(((double)*remaining_minutes / MAX_OPERATING_MINUTES) * 100.0);
}

// 2. 라즈베리 파이 전용 저전압 플래그 감지 함수
const char* check_under_voltage() {
    static char warning_msg[100];
    FILE *fp;
    char result[64];
    unsigned int status = 0;

    fp = popen("vcgencmd get_throttled 2>/dev/null", "r");
    if (fp == NULL) {
        return NULL; 
    }

    if (fgets(result, sizeof(result), fp) != NULL) {
        char *hex_str = strchr(result, '=');
        if (hex_str != NULL) {
            hex_str++; 
            status = (unsigned int)strtol(hex_str, NULL, 16); 
        }
    }
    pclose(fp);

    if (status & 0x1) {
        strcpy(warning_msg, "긴급. 전압이 낮습니다. 보조배터리를 점검하세요.");
        return warning_msg;
    }

    return NULL;
}

// 3. 시스템 전원 관리 서브 커널 루프
void monitor_power_system(double *last_check_time) {
    double current_time = get_current_timestamp();
    
    // 10분(600초) 주기 타이머 트리거
    if (current_time - *last_check_time > 600.0) {
        const char* uv_msg = check_under_voltage();
        
        if (uv_msg != NULL) {
            // 긴급 상황 (우선순위 1, 가로채기 발화)
            speak(uv_msg, 1, 0.0);
        } else {
            // 정상 상황 시 시간 역산 안내 (우선순위 3, 디바운스 2초)
            int rem_min = 0, percent = 0;
            get_estimated_remaining_time(&rem_min, &percent);
            
            char status_msg[128];
            sprintf(status_msg, "남은 예상 시간은 %d분, 잔량은 %d퍼센트입니다.", rem_min, percent);
            speak(status_msg, 3, 2.0);
        }
        
        *last_check_time = current_time;
    }
}