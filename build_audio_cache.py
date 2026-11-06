# build_audio_cache.py
import os
import wave
import hashlib
from pathlib import Path
from piper import PiperVoice

PIPER_MODEL = os.getenv("WALKING_ASSISTANT_PIPER_MODEL", "./voices/ko_KR-kss-medium.onnx")
PIPER_SPEED_FACTOR = 1.12
CACHE_DIR = Path("./audio_cache")

# test.py의 guidance_for_detection 및 상황 제어에서 실제로 출력되는 모든 완성형 문구 목록
PHRASES = [
    # 1. 횡단보도 및 신호 제어
    "정면 횡단보도", "왼쪽 횡단보도", "오른쪽 횡단보도",
    "멈추세요", "건너세요",

    # 2. 문/손잡이/도어락 상세 안내 (위치: 정면, 왼쪽, 오른쪽)
    *[f"{pos} 자동문, 버튼이나 센서를 확인하세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 문 버튼, 누르면 문이 열립니다" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 미닫이문, 옆으로 미세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 여닫이문, 밀거나 당기세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 문, 손잡이를 잡고 밀거나 당기세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 원형 문고리, 돌려서 당기세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 푸시바, 앞으로 밀어서 여세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 레버 손잡이, 아래로 누르고 밀거나 당기세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 문손잡이, 잡고 밀거나 당기세요" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 도어락" for pos in ["정면", "왼쪽", "오른쪽"]],

    # 3. 기타 시설물
    *[f"{pos} 계단" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 엘리베이터" for pos in ["정면", "왼쪽", "오른쪽"]],

    # 4. KOREAN_LABELS 사물 목록 (위치 + KOREAN_LABELS[label])
    *[f"{pos} {name}" for pos in ["정면", "왼쪽", "오른쪽"] for name in [
        "상자", "책상", "행거", "의자", "침대", "선반", "옷장", "냉장고", "창문", "철문",
        "나무", "차량", "킥보드", "오토바이", "스쿠터", "연석", "난간", "가드레일",
        "전동 킥보드", "볼라드", "전봇대", "자전거", "사람", "원형 문고리", "문손잡이",
        "레버 손잡이", "도어락", "푸시바", "문 버튼", "매립 손잡이", "출입문", "실내 문",
        "나무 문", "닫힌 문", "열린 문", "자동문", "자동 유리문", "유리 미닫이문",
        "여닫이 유리문", "손잡이형 유리문", "유리문"
    ]]
]

# 중복 문구 제거
PHRASES = sorted(list(set(PHRASES)))

def get_cache_path(text):
    cache_key = f"ko_KR-kss-medium|{PIPER_SPEED_FACTOR:.3f}\0{text}"
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.wav"

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Piper 모델 로딩 중: {PIPER_MODEL}")
    voice = PiperVoice.load(PIPER_MODEL)
    
    print(f"음성 파일 사전 생성을 시작합니다... (총 {len(PHRASES)}개 멘트)")
    for text in PHRASES:
        out_path = get_cache_path(text)
        if out_path.is_file() and out_path.stat().st_size > 44:
            continue
        
        chunks = list(voice.synthesize(text))
        if not chunks:
            continue
            
        rate = round(chunks[0].sample_rate * PIPER_SPEED_FACTOR)
        channels = chunks[0].sample_channels
        
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # S16_LE
            wf.setframerate(rate)
            for chunk in chunks:
                wf.writeframes(chunk.audio_int16_bytes)
                
        print(f"생성 완료: '{text}' -> {out_path.name}")

    print("모든 음성 캐시 생성 완료!")

if __name__ == "__main__":
    main()
