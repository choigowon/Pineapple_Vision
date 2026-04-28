import  pyttsx3

engine = pyttsx3.init()

text = "전방에 장애물"

engine.say(text)
engine.runAndWait()