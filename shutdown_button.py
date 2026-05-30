from gpiozero import Button
from subprocess import check_call
from signal import pause

button = Button(3, hold_time=2)

def shutdown():
    # 이제 앞에 'sudo'를 붙여도 비밀번호를 묻지 않습니다.
    check_call(['sudo', 'poweroff'])

button.when_held = shutdown
pause()
