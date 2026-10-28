#!/usr/bin/env python3
"""YOLOE-26n segmentation 가중치와 텍스트 인코더를 한 번 받아 준비합니다.

실행 예:
  source ~/yolo-env/bin/activate
  python download_yoloe_model.py
"""
from pathlib import Path
import os
import shutil
import sys

MODEL_NAME = "yoloe-26n-seg.pt"
PROMPT_WARMUP = ["person"]
HERE = Path(__file__).resolve().parent


def main():
    # YOLOE text encoder는 현재 작업 폴더에 저장되므로 스크립트 위치를 사용합니다.
    os.chdir(HERE)
    try:
        from ultralytics import YOLOE
    except ImportError:
        print("Ultralytics가 없습니다. 활성화된 가상환경에 설치하세요: pip install -U 'ultralytics>=8.4.0'", file=sys.stderr)
        return 2

    target = HERE / MODEL_NAME
    print(f"모델 확인/다운로드: {MODEL_NAME}", flush=True)
    try:
        model = YOLOE(MODEL_NAME)  # 없으면 Ultralytics가 공식 배포 위치에서 자동 다운로드
        source = Path(getattr(model, "ckpt_path", MODEL_NAME)).expanduser().resolve()
        if source.is_file() and source != target.resolve():
            shutil.copy2(source, target)
        elif not target.is_file() and source.is_file():
            shutil.copy2(source, target)
        if not target.is_file():
            print(f"모델이 준비되지 않았습니다. 확인 경로: {source}", file=sys.stderr)
            return 1
        print(f"모델 준비 완료: {target} ({target.stat().st_size / (1024 * 1024):.1f} MiB)", flush=True)

        print("YOLOE 텍스트 인코더 준비 중 (최초 실행 시 약 254 MB 다운로드될 수 있습니다)...", flush=True)
        model.set_classes(PROMPT_WARMUP)
        encoder = HERE / "mobileclip2_b.ts"
        if encoder.is_file():
            print(f"텍스트 인코더 준비 완료: {encoder} ({encoder.stat().st_size / (1024 * 1024):.1f} MiB)", flush=True)
        else:
            print("텍스트 인코더가 준비됐는지 확인하세요. YOLOE 버전에 따라 저장 위치/이름이 다를 수 있습니다.", flush=True)
        print("완료. 보행 보조 프로그램에서도 이 폴더를 작업 디렉터리로 사용하세요.", flush=True)
        return 0
    except Exception as exc:
        print(f"다운로드/초기화 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
