import time
import subprocess

# 설계 시 정한 보조배터리 완충 시 최대 가동 시간 (예: 4시간 = 240분)
MAX_OPERATING_MINUTES = 240 
start_time = time.time() # 프로그램 시작 시각 기록

def get_estimated_remaining_time():
    elapsed_seconds = time.time() - start_time
    elapsed_minutes = int(elapsed_seconds // 60)
    
    remaining_minutes = MAX_OPERATING_MINUTES - elapsed_minutes
    
    # 0분 이하로 떨어지지 않게 조절
    remaining_minutes = max(0, remaining_minutes)
    
    percent = int((remaining_minutes / MAX_OPERATING_MINUTES) * 100)
    
    return remaining_minutes, percent

def check_under_voltage():
    try:
        # 라즈베리 파이 전용 명령어 실행
        result = subprocess.check_output(['vcgencmd', 'get_throttled']).decode()
        status = int(result.split('=')[1], 16)
        
        # 0x1: 현재 저전압 발생 중, 0x10000: 최근에 저전압 발생했었음
        if status & 0x1:
            return "긴급. 전압이 낮습니다. 보조배터리를 점검하세요."
        return None
    except:
        # PC(Windows) 환경에서는 에러가 나므로 무시
        return None