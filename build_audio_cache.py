# build_audio_cache.py

import os
import wave
import hashlib
from pathlib import Path
from piper import PiperVoice

PIPER_MODEL = os.getenv("WALKING_ASSISTANT_PIPER_MODEL", "./voices/ko_KR-kss-medium.onnx")
PIPER_SPEED_FACTOR = 1.12
CACHE_DIR = Path("./audio_cache")

PHRASES = [
    # 1. 횡단보도
    "정면 횡단보도", "왼쪽 횡단보도", "오른쪽 횡단보도",
    "멈추세요", "건너세요",

    # 1-1. OCR 문 표지판 안내
    "당기세요", "미세요",
    "문 당기세요", "문 미세요",

    # 2. 독립 부속품 및 손잡이 단독 멘트
    *[f"{pos} {name}" for pos in ["정면", "왼쪽", "오른쪽"] for name in [
        "원형 문고리", "푸시바, 밀으세요", "레버 손잡이, 내리세요",
        "문손잡이", "문 버튼, 누르세요", "도어락"
    ]],

    # 3. 문 본체 + 지침 조합 멘트 (미닫이문 및 자동문 조합)
    *[f"{pos} {name}" for pos in ["정면", "왼쪽", "오른쪽"] for name in [
        "자동문, 버튼을 누르세요", "자동문", "미닫이문, 옆으로 미세요", "미닫이문",
        
        "유리문, 레버 손잡이를 내리세요", "문, 레버 손잡이를 내리세요", "나무문, 레버 손잡이를 내리세요", "철문, 레버 손잡이를 내리세요",
        "유리문, 문고리를 돌리세요", "문, 문고리를 돌리세요", "나무문, 문고리를 돌리세요", "철문, 문고리를 돌리세요",
        "유리문, 푸시바를 밀으세요", "문, 푸시바를 밀으세요", "나무문, 푸시바를 밀으세요", "철문, 푸시바를 밀으세요",
        "유리문, 손잡이 잡고 당기세요", "문, 손잡이 잡고 당기세요", "나무문, 손잡이 잡고 당기세요", "철문, 손잡이 잡고 당기세요",
        "유리문, 도어락", "문, 도어락", "나무문, 도어락", "철문, 도어락",

        "유리문", "문", "나무문", "철문"
    ]],

    # 4. 기타 장애물 및 시설물
    *[f"{pos} 계단" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} 엘리베이터" for pos in ["정면", "왼쪽", "오른쪽"]],
    *[f"{pos} {name}" for pos in ["정면", "왼쪽", "오른쪽"] for name in [
        "상자", "책상", "행거", "의자", "침대", "선반", "옷장", "냉장고", "창문",
        "나무", "차량", "킥보드", "오토바이", "스쿠터", "연석", "난간",
        "전동 킥보드", "볼라드", "전봇대", "자전거", "사람",
        "캐리어", "선풍기", "가방"
    ]]
]

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
            wf.setsampwidth(2)
            wf.setframerate(rate)
            for chunk in chunks:
                wf.writeframes(chunk.audio_int16_bytes)
                
        print(f"생성 완료: '{text}' -> {out_path.name}")

    print("모든 음성 캐시 생성 완료!")

if __name__ == "__main__":
    main()
