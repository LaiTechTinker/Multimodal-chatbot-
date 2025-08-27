import os
import sys
import pyttsx3
import speech_recognition as sr
import pyautogui
import webbrowser
import datetime 
# from evaluate import evaluate


def initialize_speak():
    engine=pyttsx3.init('sapi5') #this initoalize the google voice engine
    voices=engine.getProperty('voices')# this get the voices 
    engine.setProperty('voice',voices[1].id)#this get the female voice
    rate=engine.getProperty('rate') #this get the rate of voices
    engine.setProperty('rate',rate-50)
    volume=engine.getProperty('volume')
    engine.setProperty('volume',volume+0.50)
    return engine
def speak_action(engine,text):
    engine.say(text)
    engine.runAndWait()
def get_my_voice():
    recognizer=sr.Recognizer()#this is used for recognizing voice
    with sr.Microphone() as src:
        recognizer.adjust_for_ambient_noise(src, duration=0.5)
        print("Listening.......", end="", flush=True)
        recognizer.pause_threshold=1.0
        recognizer.phrase_threshold=0.3
        recognizer.sample_rate = 48000
        recognizer.dynamic_energy_threshold=True
        recognizer.operation_timeout=5
        recognizer.non_speaking_duration=0.5
        recognizer.dynamic_energy_adjustment=2
        recognizer.energy_threshold=4000
        recognizer.phrase_time_limit = 10
        # print(sr.Microphone.list_microphone_names())
        audio = recognizer.listen(src)
# speak_action(initialize_speak(),"hello my genius master Lai")