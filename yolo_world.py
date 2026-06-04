import cv2
import numpy as np
import threading
import time
from ultralytics import YOLOWorld
import torch
import os
import signal  
import sys

from tts import VoiceManager
from check_battery import get_estimated_remaining_time, check_under_voltage

# 라즈베리파이 CPU 최적화
torch.set_num_threads(2)

latest_frame = None
frame_lock = threading.Lock()
is_running = True

current_zone = "UNKNOWN"  
zone_lock = threading.Lock()

voice_engine = VoiceManager()

def speak_pure_audio(text):
    def _say():
        try:
            os.system(f"espeak -v ko+f3 -s 160 -a 200 \"{text}\" > /dev/null 2>&1")
        except Exception as e:
            pass
    threading.Thread(target=_say, daemon=True).start()

# [추가] 배터리 및 전압 상태를 주기적으로 모니터링하는 스레드 함수
def battery_monitor_thread_func():
    global is_running
    
    # 처음 실행 후 안정화를 위해 잠시 대기
    time.sleep(5)
    
    # 알림 주기 설정을 위한 변수 (초 단위)
    UNDER_VOLTAGE_CHECK_INTERVAL = 10  # 저전압은 10초마다 체크
    BATTERY_STATUS_LOG_INTERVAL = 300   # 배터리 잔량 로그는 5분(300초)마다 출력
    
    last_voltage_check = 0
    last_status_log = 0

    print("[시스템 내부 모니터링] 배터리 및 전압 체크 스레드 가동 시작")

    while is_running:
        current_time = time.time()

        # 1. 저전압 긴급 상황 체크 (vcgencmd 이용)
        if current_time - last_voltage_check > UNDER_VOLTAGE_CHECK_INTERVAL:
            voltage_warning = check_under_voltage()
            if voltage_warning:
                print(f"⚠️ [⚠️경고] {voltage_warning}")
                # 음성 엔진을 통해 사용자에게 즉시 위험 경고 (우선순위 높임)
                voice_engine.speak("경고. 전압이 낮습니다. 배터리를 점검하세요.", priority=5, debounce_time=10)
            last_voltage_check = current_time

        # 2. 잔여 시간 및 퍼센트 계산 및 로그 출력
        if current_time - last_status_log > BATTERY_STATUS_LOG_INTERVAL:
            remaining_min, percent = get_estimated_remaining_time()
            print(f"[배터리 상태] 잔여 시간: 약 {remaining_min}분 ({percent}%)")
            
            # 배터리가 15% 이하로 떨어지면 경고 음성 출력
            if percent <= 15 and percent > 0:
                voice_engine.speak("배터리가 부족합니다.", priority=4, debounce_time=60)
                
            last_status_log = current_time

        # CPU 과점유 방지를 위한 주기적 휴지
        time.sleep(1)

def inference_thread_func(model, clean_labels, korean_names, crop_config):
    global latest_frame, is_running, current_zone

    last_spoken_state = ""
    last_spoken_time = 0
    AUDIO_COOLDOWN = 1.3  
    
    start_x, end_x = crop_config
    crop_width = end_x - start_x

    # 순정 컷오프 타이트 조율
    class_conf_thresholds = {
        'stairs': 0.27, 'handrail': 0.18, 'door': 0.18, 'handle': 0.18, 
        'refrigerator': 0.55, 'scooter': 0.20, 'bicycle': 0.22, 'motorcycle': 0.25,     
        'bench': 0.50, 'table': 0.35, 'box': 0.35, 'window': 0.30,
        'wardrobe': 0.40, 'shelf': 0.45, 'chair': 0.35, 'person': 0.30, 
        'plant': 0.30, 'tv': 0.35, 'laptop': 0.35, 'phone': 0.30, 
        'microwave': 0.40, 'bollard': 0.35, 'cone': 0.35, 'pole': 0.35, 
        'puddle': 0.30, 'tree': 0.30, 'car': 0.35, 'bus': 0.35, 
        'truck': 0.35, 'bed': 0.40, 'hanger': 0.40, 'stair': 0.30
    }

    indoor_hints = ['refrigerator', 'bed', 'wardrobe', 'hanger', 'shelf', 'tv', 'laptop', 'microwave']
    outdoor_hints = ['scooter', 'bollard', 'cone', 'pole', 'puddle', 'tree', 'car', 'bus', 'truck']

    print(f"🧠 [라즈베리파이 AI 엔진] 가중치 및 한글 변환 매트릭스 구동 완료")

    while is_running:
        with frame_lock:
            local_frame = latest_frame
            latest_frame = None

        if local_frame is None:
            time.sleep(0.01)
            continue

        cropped_frame = local_frame[:, start_x:end_x]

        try:
            results = model.predict(cropped_frame, imgsz=320, conf=0.15, iou=0.40, verbose=False)
            
            boxes = []
            for result in results:
                if result.boxes:
                    boxes.extend(result.boxes)
            
            boxes.sort(key=lambda b: float(b.conf[0]), reverse=True)
            final_selected_obj = None

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                raw_name = model.names[cls_id]
                
                # 1단계: 프롬프트 원문을 시스템 내부 표준 토큰명으로 1차 변환
                class_name = clean_labels.get(raw_name, raw_name)

                # 공간 인지 필터
                with zone_lock:
                    if class_name in indoor_hints and conf > 0.45:
                        if current_zone != "INDOOR": current_zone = "INDOOR"
                    elif class_name in outdoor_hints and conf > 0.45:
                        if current_zone != "OUTDOOR": current_zone = "OUTDOOR"
                    if current_zone == "INDOOR" and class_name in outdoor_hints: conf -= 0.40  
                    elif current_zone == "OUTDOOR" and class_name in indoor_hints: conf -= 0.40  

                # 가중치 튜닝
                if class_name in ['door', 'handle', 'handrail']: 
                    conf = min(conf + 0.35, 1.0)  
                elif class_name in ['wardrobe', 'shelf']:
                    conf = conf - 0.20

                if class_name in class_conf_thresholds and conf < class_conf_thresholds[class_name]:
                    continue  

                obj_center_x = (x1 + x2) / 2.0
                if obj_center_x < (crop_width * 0.33): direction = "좌측"
                elif obj_center_x > (crop_width * 0.66): direction = "우측"
                else: direction = "정면"

                # 2단계: 표준 토큰명을 최종 한글 단어로 강제 치환 (영어 유출 절대 차단)
                ko_name = korean_names.get(class_name, class_name)
                
                final_selected_obj = f"{direction} {ko_name}"
                break  

            if final_selected_obj:
                if "wall" not in final_selected_obj and "window" not in final_selected_obj:
                    current_time = time.time()
                    if (final_selected_obj != last_spoken_state) or (current_time - last_spoken_time > AUDIO_COOLDOWN):
                        print(f"[순정 즉시출력] 🎯 {final_selected_obj}")
                        
                        voice_engine.speak(final_selected_obj, priority=3, debounce_time=AUDIO_COOLDOWN)

                        last_spoken_state = final_selected_obj
                        last_spoken_time = current_time
            else:
                last_spoken_state = ""

        except Exception as e:
            pass

        time.sleep(0.01)


def run_pi_system():
    global latest_frame, is_running

    print("⏳ [초기화] AI 모델 로딩 중...")
    model = YOLOWorld('fixed_model.pt')

    # 초기 웜업(Warm-up) 연산 수행하여 실시간 스트리밍 시 버벅임 차단
    print("⚡ [초기화] 하드웨어 가속 및 웜업 연산 가동...")
    dummy_img = np.zeros((240, 160, 3), dtype=np.uint8)
    model.predict(dummy_img, imgsz=320, verbose=False)

    # 💡 [오타 교정 완료] 중복 및 기호 누락 코드 깔끔하게 수정됨
    clean_labels = {
        "automatic glass sliding door with silver metal frame": "door",
        "framed glass door panel for entrance": "door", 
        "room door with a handle": "door", 
        "door handle or door knob": "handle", 
        
        "pedestrian stairs with multiple continuous vertical steps, not a single road curb or flat ramp": "stairs", 
        "pedestrian stairs leading upwards with sequential levels": "stairs", 
        
        "electric kick scooter with a vertical handlebar and a flat board to stand on": "scooter", 
        "person": "person", 
        "riding bicycle": "bicycle", 
        "motorcycle with heavy engine": "motorcycle", 
        
        "bench": "bench", 
        "furniture chair with backrest": "chair", 
        "potted plant": "plant", "tv": "tv", "laptop": "laptop", "cell phone": "phone", "microwave": "microwave", "bollard": "bollard", 
        "traffic cone": "cone", "utility pole": "pole", "water puddle": "puddle", 
        
        "vertical storage shelf with multiple grid racks for holding items, not a flat table": "shelf", 
        "stair": "stair", "tree": "tree", 
        "passenger car on the road": "car", 
        "large passenger bus": "bus", 
        "cargo truck": "truck", 
        
        "kitchen refrigerator appliance": "refrigerator", 
        "bed": "bed", 
        "flat dining table or desk supported by legs with a single flat surface for working, NO multiple shelves": "table", 
        "cardboard box": "box", 
        "wall": "wall", 
        "window fixed in a wall": "window", 
        "wooden wardrobe": "wardrobe", 
        "clothes hanger rack": "hanger", 
        
        "safety handrail or metallic grab bar mounted along stairs": "handrail"
    }

    korean_names = {
        'door': '문', 'handle': '문 손잡이', 'shelf': '선반', 'wardrobe': '옷장', 'hanger': '행거',
        'table': '테이블', 'bench': '벤치', 'refrigerator': '냉장고', 'chair': '의자',
        'person': '사람', 'stairs': '계단', 'stair': '계단', 'tv': '티비', 'laptop': '노트북',
        'scooter': '킥보드', 'bicycle': '자전거', 'motorcycle': '오토바이', 
        'plant': '화분', 'phone': '핸드폰', 'microwave': '전자레인지', 'bollard': '볼라드', 
        'cone': '라바콘', 'pole': '전신주', 'puddle': '물웅덩이', 'tree': '나무', 
        'car': '자동차', 'bus': '버스', 'truck': '트럭', 'bed': '침대', 'box': '상자', 
        'handrail': '난간', 'window': '창문'
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    W, H = 320, 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    crop_w = 160
    start_x = (W - crop_w) // 2
    end_x = start_x + crop_w

    ai_thread = threading.Thread(
        target=inference_thread_func, 
        args=(model, clean_labels, korean_names, (start_x, end_x)), 
        daemon=True
    )
    ai_thread.start()

    battery_thread = threading.Thread(target=battery_monitor_thread_func, daemon=True)
    battery_thread.start()

    # Ctrl+C 수신 시 자원 해제 후 프로세스를 즉시 완전 종료시키는 핸들러
    def signal_handler(sig, frame):
        print("\n👋 [시스템 즉시 종료] Ctrl+C 즉시 해제 요청 승인. 자원을 해제합니다.")
        global is_running
        is_running = False
        cap.release()
        os._exit(0)  

    signal.signal(signal.SIGINT, signal_handler)

    print("🚀 [라즈베리파이] 모든 패치가 완료되었습니다. 탐지를 시작합니다...")

    while is_running:
        for _ in range(2): 
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret: 
            break
        with frame_lock: 
            latest_frame = frame
        time.sleep(0.04)


if __name__ == "__main__":
    run_pi_system()
