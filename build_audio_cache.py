# build_audio_cache.py
import os
import hashlib
from pathlib import Path
from piper import PiperVoice

# a.py에 설정된 모델 경로 및 속도와 동일하게 설정
PIPER_MODEL = os.getenv("WALKING_ASSISTANT_PIPER_MODEL", "./voices/ko_KR-kss-medium.onnx")
PIPER_SPEED_FACTOR = 1.12
CACHE_DIR = Path("./audio_cache")

# a.py에서 사용되는 모든 한국어 안내 멘트 모음
PHRASES = [
    # 횡단보도 및 신호
    "정면 횡단보도", "왼쪽 횡단보도", "오른쪽 횡단보도",
    "멈추세요", "건너세요",
    
    # 방향별 이동 장애물
    "정면 장애물", "왼쪽 장애물", "오른쪽 장애물",
    "정면 계단", "왼쪽 계단", "오른쪽 계단",
    "정면 엘리베이터", "왼쪽 엘리베이터", "오른쪽 엘리베이터",
    "정면 자동 유리문", "왼쪽 자동 유리문", "오른쪽 자동 유리문",
    "정면 유리 미닫이문", "왼쪽 유리 미닫이문", "오른쪽 유리 미닫이문",
    "정면 여닫이 유리문", "왼쪽 여닫이 유리문", "오른쪽 여닫이 유리문",
    "정면 손잡이형 유리문", "왼쪽 손잡이형 유리문", "오른쪽 손잡이형 유리문",
    "정면 유리문", "왼쪽 유리문", "오른쪽 유리문",
    "정면 원형 문고리", "왼쪽 원형 문고리", "오른쪽 원형 문고리",
    "정면 푸시바", "왼쪽 푸시바", "오른쪽 푸시바",
    "정면 레버 손잡이", "왼쪽 레버 손잡이", "오른쪽 레버 손잡이",
    "정면 문손잡이", "왼쪽 문손잡이", "오른쪽 문손잡이",
    "정면 문 버튼", "왼쪽 문 버튼", "오른쪽 문 버튼",
    "정면 도어락", "왼쪽 도어락", "오른쪽 도어락",
    "정면 철문", "왼쪽 철문", "오른쪽 철문",
    "정면 나무문", "왼쪽 나무문", "오른쪽 나무문",
    "정면 미닫이문", "왼쪽 미닫이문", "오른쪽 미닫이문",
    "정면 여닫이문", "왼쪽 여닫이문", "오른쪽 여닫이문",
    "정면 열린 문", "왼쪽 열린 문", "오른쪽 열린 문",
    "정면 닫힌 문", "왼쪽 닫힌 문", "오른쪽 닫힌 문",
    "정면 문", "왼쪽 문", "오른쪽 문",
    
    # 사물 명칭 (KOREAN_LABELS 조합)
    "정면 상자", "왼쪽 상자", "오른쪽 상자",
    "정면 책상", "왼쪽 책상", "오른쪽 책상",
    "정면 행거", "왼쪽 행거", "오른쪽 행거",
    "정면 의자", "왼쪽 의자", "오른쪽 의자",
    "정면 침대", "왼쪽 침대", "오른쪽 침대",
    "정면 선반", "왼쪽 선반", "오른쪽 선반",
    "정면 옷장", "왼쪽 옷장", "오른쪽 옷장",
    "정면 냉장고", "왼쪽 냉장고", "오른쪽 냉장고",
    "정면 창문", "왼쪽 창문", "오른쪽 창문",
    "정면 나무", "왼쪽 나무", "오른쪽 나무",
    "정면 차량", "왼쪽 차량", "오른쪽 차량",
    "정면 킥보드", "왼쪽 킥보드", "오른쪽 킥보드",
    "정면 전동 킥보드", "왼쪽 전동 킥보드", "오른쪽 전동 킥보드",
    "정면 오토바이", "왼쪽 오토바이", "오른쪽 오토바이",
    "정면 스쿠터", "왼쪽 스쿠터", "오른쪽 스쿠터",
    "정면 연석", "왼쪽 연석", "오른쪽 연석",
    "정면 난간", "왼쪽 난간", "오른쪽 난간",
    "정면 가드레일", "왼쪽 가드레일", "오른쪽 가드레일",
    "정면 볼라드", "왼쪽 볼라드", "오른쪽 볼라드",
    "정면 전봇대", "왼쪽 전봇대", "오른쪽 전봇대",
    "정면 자전거", "왼쪽 자전거", "오른쪽 자전거",
    "정면 사람", "왼쪽 사람", "오른쪽 사람",
]

def get_cache_path(text):
    cache_key = f"ko_KR-kss-medium|{PIPER_SPEED_FACTOR:.3f}\0{text}"
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.wav"

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Piper 모델 로딩 중: {PIPER_MODEL}")
    voice = PiperVoice.load(PIPER_MODEL)
    
    print("음성 파일 사전 생성을 시작합니다...")
    for text in PHRASES:
        out_path = get_cache_path(text)
        if out_path.is_file():
            continue
        
        # WAV 생성
        import wave
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
