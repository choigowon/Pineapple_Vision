import cv2
import numpy as np
import threading
import time
from ultralytics import YOLOWorld

# --- 1. 버그 없는 일회성 비동기 TTS 함수 정의 ---
def say_english_thread(text):
    """호출될 때마다 독립된 스레드에서 pyttsx3를 초기화하여 말하고 종료 (먹통 버그 원천 차단)"""
    import pyttsx3
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'en' in voice.id.lower() or 'english' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 170)  # 속도 살짝 빠르게 설정
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS 발화 오류: {e}")

def speak_async(text):
    """메인 영상 루프를 방해하지 않도록 백그라운드로 음성 실행"""
    t = threading.Thread(target=say_english_thread, args=(text,), daemon=True)
    t.start()


# --- 2. 메인 프로세스 함수 ---
def run_velog_optimized_camera():
    print("YOLOv8s-worldv2 모델 로드 중...")
    model = YOLOWorld('yolov8s-worldv2.pt')

    custom_classes = [
        "a room door with a handle", "a partially visible entrance door", 
        "a closed wooden door frame", "a glass door with a metal handle",
        "concrete curb on the street", "stairs leading downwards", "stairs leading upwards", 
        "wheelchair ramp or slope", "electric kick scooter", 
        "person", "bicycle", "motorcycle", "sink", "stop sign", "bench", "chair", 
        "potted plant", "tv", "laptop", "cell phone", "microwave", "bollard", 
        "traffic cone", "utility pole", "water puddle", "shelf", "stair", "tree",
        "a car or a part of a car", "a bus or a part of a bus", "a truck or a part of a truck", 
        "a large refrigerator", "a bed in a bedroom", "a dining table", "a cardboard box on the floor", 
        "wall", "window", "wardrobe", "clothes hanger rack"
    ]
    
    model.set_classes(custom_classes)
    print("정밀 프롬프트 지정 완료!")

    door_prompts = [
        "a room door with a handle", "a partially visible entrance door", 
        "a closed wooden door frame", "a glass door with a metal handle"
    ]
    large_vehicle_prompts = {
        "a car or a part of a car": "car", "a bus or a part of a bus": "bus", "a truck or a part of a truck": "truck",
        "a large refrigerator": "refrigerator", "a bed in a bedroom": "bed", "a cardboard box on the floor": "box"
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 입력 해상도 세팅
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 실시간 음성 제어용 변수
    last_spoken_state = ""    # 마지막으로 소리 낸 사물 고유 상태 (예: "door_close")
    last_spoken_time = 0      # 마지막으로 소리 낸 타임스탬프
    AUDIO_COOLDOWN = 0.8       # 동일한 사물일 때 재안내 대기 시간 (0.8초로 단축하여 실시간성 극대화)

    print("실시간 가까운 사물 안내를 시작합니다. 종료하려면 'q'를 누세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 요청하신 대로 연산량이 큰 왜곡 보정(undistort)은 전면 제외 (라즈베리파이 최적화)

        # 광각 카메라 거치 방향에 맞춰 90도 회전
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h_frame, w_frame = frame.shape[:2]

        results = model.predict(frame, conf=0.24, iou=0.45, stream=True, verbose=False)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                
                class_name = custom_classes[cls_id]
                
                # 이름 세탁 매핑
                if class_name in door_prompts:
                    class_name = "door"
                elif class_name == "concrete curb on the street":
                    class_name = "curb"
                elif class_name == "stairs leading downwards":
                    class_name = "stairs down"
                elif class_name == "stairs leading upwards":
                    class_name = "stairs up"
                elif class_name == "wheelchair ramp or slope":
                    class_name = "ramp"
                elif class_name == "electric kick scooter":
                    class_name = "scooter"
                elif class_name in large_vehicle_prompts:
                    class_name = large_vehicle_prompts[class_name]
                elif class_name == "clothes hanger rack":
                    class_name = "hanger"

                # 화면 면적 대비 박스 크기로 거리 상태 판단
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / (w_frame * h_frame)

                if area_ratio > 0.12:
                    distance_status = "close"
                elif area_ratio > 0.03:
                    distance_status = "medium"
                else:
                    distance_status = "far"

                # 모든 인식 결과를 일단 수집 (가장 가까운 것을 선별하기 위함)
                detected_objects.append({
                    'name': class_name, 
                    'distance': distance_status,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 
                    'y2_coord': y2  # 바닥에 얼마나 가까운지 판단하는 지표
                })

                # 바운딩 박스 그리기
                if class_name in ["curb", "stairs down", "stairs up", "ramp"]:
                    color = (0, 0, 255)
                elif class_name == "door":
                    color = (255, 100, 0)
                elif class_name in ["wall", "window", "wardrobe", "hanger"]:
                    color = (200, 0, 200)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_y = y1 + 15 if y1 < 15 else y1 - 10
                
                label = f"{class_name} ({distance_status}) {conf_score:.2f}"
                cv2.putText(frame, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 3. [핵심] 가장 가까운(앞에 있는) 사물 딱 1개만 골라서 실시간 음성 출력 ---
        if detected_objects:
            # y2 좌표(박스의 맨 아래쪽 선)가 클수록 카메라 기준 내 발 앞에 가장 가까운 물체입니다.
            detected_objects.sort(key=lambda o: o['y2_coord'], reverse=True)
            closest_target = detected_objects[0]
            
            obj_name = closest_target['name']
            obj_dist = closest_target['distance']
            
            # 벽이나 창문 같은 배경 요소를 제외한 실질적 방해물만 말하기
            if obj_name not in ["wall", "window"]:
                current_time = time.time()
                current_state_string = f"{obj_name}_{obj_dist}"
                
                # 상황 A: 완전히 새로운 물체나 거리 단계가 바뀌었으면 즉시 출력
                # 상황 B: 똑같은 물체가 계속 눈앞에 있다면 0.8초마다 지속적으로 안내
                if (current_state_string != last_spoken_state) or (current_time - last_spoken_time > AUDIO_COOLDOWN):
                    speech_text = f"{obj_name}, {obj_dist}"
                    
                    # 비동기로 즉시 영어 발화 실행 (렉 없음)
                    speak_async(speech_text)
                    
                    # 상태 업데이트
                    last_spoken_state = current_state_string
                    last_spoken_time = current_time
        else:
            # 화면에 아무것도 감지되지 않으면 최근 기억 상태를 리셋하여 다시 사물 진입 시 즉시 안내되도록 함
            last_spoken_state = ""

        cv2.imshow("YOLOv8s-World (Bug-Free Audio Fixed)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_velog_optimized_camera()
