#!/bin/bash
cd /home/pav

# 볼륨 확보 및 즉시 시작 멘트
amixer set Master 100% > /dev/null 2>&1
espeak -v ko+f3 -s 160 -a 200 "시작합니다." > /dev/null 2>&1

# 메인 코드 실행 (이 안에서 웜업이 진행됨)
/home/pav/yolo_env/bin/python3 /home/pav/test.py >> /home/pav/ai_output.log 2>&1
