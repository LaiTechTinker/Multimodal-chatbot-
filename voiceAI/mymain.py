import os
import datetime
import sys
import speech_recognition as sr
import pyttsx3
import time
import webbrowser
from evaluate import evaluate

def initialize_engine():
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-50)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume+0.25)
    return engine
def speak(text):
    engine = initialize_engine()
    engine.say(text)
    engine.runAndWait()
speak('hello master lai how are you doing')
def command1():
    r = sr.Recognizer()

    with sr.Microphone(device_index=2) as source:
        
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening.......", end="", flush=True)
        r.pause_threshold=3.0
        r.phrase_threshold=0.3
        r.sample_rate = 48000
        r.dynamic_energy_threshold=True
        r.operation_timeout=5
        r.non_speaking_duration=0.5
        r.dynamic_energy_adjustment=2
        r.energy_threshold=4000
        r.phrase_time_limit = 40
        # print(sr.Microphone.list_microphone_names())
        audio = r.listen(source)
    try:
        print("\r" ,end="", flush=True)
        print("Recognizing......", end="", flush=True)
        query = r.recognize_google(audio, language='en-in')
        print("\r" ,end="", flush=True)
        print(f"User said : {query}\n")
        print(query)
    except Exception as e:
        print("Say that again please")
        return "None"
    return query
# def cal_day():
#     day = datetime.datetime.today().weekday() + 1
#     day_dict={
#         1:"Monday",
#         2:"Tuesday",
#         3:"Wednesday",
#         4:"Thursday",
#         5:"Friday",
#         6:"Saturday",
#         7:"Sunday"
#     }
#     if day in day_dict.keys():
#         day_of_week = day_dict[day]
#         print(day_of_week)
#     return day_of_week
# def wishMe():
#     hour = int(datetime.datetime.now().hour)
#     t = time.strftime("%I:%M:%p")
#     day = cal_day()

#     if(hour>=0) and (hour<=12) and ('AM' in t):
#         speak(f"Good morning Lai, it's {day} and the time is {t}")
#     elif(hour>=12)  and (hour<=16) and ('PM' in t):
#         speak(f"Good afternoon Lai, it's {day} and the time is {t}")
#     else:
#         speak(f"Good evening Lai, it's {day} and the time is {t}")
def social_media(command):
    if 'facebook' in command:
        speak("opening your facebook")
        webbrowser.open("https://www.facebook.com/")
    elif 'whatsapp' in command:
        speak("opening your whatsapp")
        webbrowser.open("https://web.whatsapp.com/")
    elif 'discord' in command:
        speak("opening your discord server")
        webbrowser.open("https://discord.com/")
    elif 'switch' in command:
        speak("switching to conversation")
        print("switching to conversation")
        
        while True:
         user_message = command1().lower()
         if user_message is None:
            continue
         print(user_message)
         response = evaluate(user_message)
         speak(' '.join(response))
         print(' '.join(response))

         user_message_lower = user_message.lower()
         if 'exit' in user_message_lower or 'stop' in user_message_lower or 'quit' in user_message_lower:
            speak("Exiting conversation mode.")
            break
    elif 'instagram' in command:
        speak("opening your instagram")
        webbrowser.open("https://www.instagram.com/")
   
    else:
        speak("No result found")




if __name__=="__main__":

    while True:
        query = command1().lower()
        # query  = input("Enter your command-> ")
        # query=command().lower()
        if ('facebook' in query) or ('discord' in query) or ('whatsapp' in query) or ('instagram' in query) or ('switch' in query):
            social_media(query)
        # elif ("university time table" in query) or ("schedule" in query):
        #     schedule()
        # elif ("open calculator" in query) or ("open notepad" in query) or ("open paint" in query):
        #     openApp(query)
        # elif ("close calculator" in query) or ("close notepad" in query) or ("close paint" in query):
        #     closeApp(query)
    # query=command().lower()
    # print(query)