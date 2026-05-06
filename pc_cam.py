import cv2
from ultralytics import YOLO
import easyocr
from tts import VoiceManager, check_under_voltage
import time
# =====================================
# 초기 설정
# =====================================
voice_assistant = VoiceManager()
# YOLO Nano 모델
model = YOLO('yolov8n.pt')

# OCR
reader = easyocr.Reader(['ko', 'en'])

# 모드
# 1 = 사물 인식 + 거리 경고
# 2 = 글자 인식
mode = 1

# 마지막 인식 값
last_object = ""
last_text = ""

# 시간 체크
last_object_detection_time = 0
last_ocr_time = 0
last_warning_time = 0
last_battery_check_time = 0
 
# =====================================
# 거리 측정 설정
# =====================================

REAL_WIDTH = 40      # 사람 평균 어깨 너비(cm)
FOCAL_LENGTH = 550   # 테스트용 값
WARNING_DISTANCE = 100  # 100cm 이하 경고

# =====================================
# 웹캠 연결
# =====================================

cap = cv2.VideoCapture(0)

# 라즈베리파이용 해상도 최적화
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    display_frame = frame.copy()

    # ==================================================
    # 1번 모드 : 사물 인식 + 거리 경고
    # ==================================================

    if mode == 1:

        # 0.7초마다만 객체 인식 실행
        if current_time - last_object_detection_time > 0.7:

            results = model(frame, conf=0.5)

            display_frame = results[0].plot()

            frame_center = frame.shape[1] / 2

            for result in results[0].boxes:

                class_id = int(result.cls[0])

                label = model.names[class_id]

                # 사람만 거리 측정
                if label != "person":
                    continue

                x1, y1, x2, y2 = result.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # 화면 중앙 객체만 사용
                center_x = (x1 + x2) / 2

                if abs(center_x - frame_center) > 150:
                    continue

                pixel_width = x2 - x1

                if pixel_width <= 0:
                    continue

                # 거리 계산
                distance = (
                    REAL_WIDTH * FOCAL_LENGTH
                ) / pixel_width

                # 거리 표시
                cv2.putText(
                    display_frame,
                    f"{distance:.1f} cm",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # 가까우면 경고
                if distance < WARNING_DISTANCE:

                    # 빨간 박스
                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        display_frame,
                        "WARNING",
                        (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    # 3초마다 경고 음성
                    if current_time - last_warning_time > 3:

                        voice_assistant.speak(
                            "경고. 사람이 가까이 있습니다", priority=1
                        )

                        last_warning_time = current_time

            last_object_detection_time = current_time

        cv2.putText(
            display_frame,
            "MODE 1 : OBJECT WARNING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ==================================================
    # 2번 모드 : 글자 인식
    # ==================================================

    elif mode == 2:

        h, w = frame.shape[:2]

        # 중앙 영역만 OCR
        center_crop = frame[
            h // 4: 3 * h // 4,
            w // 4: 3 * w // 4
        ]

        # OCR 영역 표시
        cv2.rectangle(
            display_frame,
            (w // 4, h // 4),
            (3 * w // 4, 3 * h // 4),
            (0, 255, 0),
            2
        )

        # 3초마다 OCR 실행
        if current_time - last_ocr_time > 3:

            gray = cv2.cvtColor(
                center_crop,
                cv2.COLOR_BGR2GRAY
            )

            ocr_results = reader.readtext(gray)

            detected_text = " ".join(
                [r[1] for r in ocr_results]
            ).strip()

            if detected_text:

                if detected_text != last_text:

                    print("글자:", detected_text)

                    voice_assistant.speak(detected_text)

                    last_text = detected_text

            else:

                last_text = ""

            last_ocr_time = current_time

        cv2.putText(
            display_frame,
            "MODE 2 : TEXT READING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ==================================================
    # 공통 UI
    # ==================================================

    cv2.putText(
        display_frame,
        "1:Object  2:OCR  Q:Quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if current_time - last_battery_check_time > 120:
        
        # vcgencmd를 통해 하드웨어 전압 상태 확인
        low_volt_msg = check_under_voltage()
        
        # 저전압 신호가 감지된 경우에만 음성 안내
        if low_volt_msg:
            print("[시스템 경고]", low_volt_msg) # 터미널 출력
            voice_assistant.speak(low_volt_msg, priority=1)         # 음성 출력
            
        last_battery_check_time = current_time

    cv2.imshow(
        "Vision AI Assistant",
        display_frame
    )

    # ==================================================
    # 키 입력
    # ==================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('1'):

        mode = 1

        voice_assistant.speak(
            "사물 인식 모드", priority=2
        )

    elif key == ord('2'):

        mode = 2

        voice_assistant.speak(
            "글자 인식 모드", priority=2
        )

# =====================================
# 종료
# =====================================

cap.release()
cv2.destroyAllWindows()
