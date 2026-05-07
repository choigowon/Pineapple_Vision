import time

# 설계 시 정한 보조배터리 완충 시 최대 가동 시간 (예: 4시간 = 240분)
MAX_OPERATING_MINUTES = 240 
start_time = time.time() # 프로그램 시작 시각 기록

def get_estimated_remaining_time():
    elapsed_seconds = time.time() - start_time
    elapsed_minutes = int(elapsed_seconds // 60)
    
    remaining_minutes = MAX_OPERATING_MINUTES - elapsed_minutes
    
    # 0분 이하로 떨어지지 않게 조절
    remaining_minutes = max(0, remaining_minutes)
    
    # 남은 시간을 %로 환산 (교수님께 보여주기용 수치)
    percent = int((remaining_minutes / MAX_OPERATING_MINUTES) * 100)
    
    return remaining_minutes, percent