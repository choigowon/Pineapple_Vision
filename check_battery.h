#ifndef CHECK_BATTERY_H
#define CHECK_BATTERY_H

// 설계 시 정한 보조배터리 완충 시 최대 가동 시간 (예: 4시간 = 240분)
#define MAX_OPERATING_MINUTES 240

// 외부에서 호출할 핵심 인터페이스 함수 선언
void init_battery_monitor();
double get_current_timestamp();
void get_estimated_remaining_time(int *remaining_minutes, int *percent);
const char* check_under_voltage();
void monitor_power_system(double *last_check_time);

#endif // BATTERY_MONITOR_H