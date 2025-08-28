import os
import pyttsx3
import speech_recognition as sr
import webbrowser
import datetime
from evaluate import evaluate
import time


class VoiceCommand:
    def __init__(self):
        self.engine = self.initialize_speak()
        self.query = None

    def initialize_speak(self):
        engine = pyttsx3.init('sapi5')  # initialize speech engine
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # female voice
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 50)
        volume = engine.getProperty('volume')
        engine.setProperty('volume', min(volume + 0.50, 1.0))
        return engine

    def speak_action(self, text):
        """Speaks the given text and adds a pause so it is audible."""
        self.engine.say(text)
        self.engine.runAndWait()
        print(f"[SPEAK]: {text}")
        time.sleep(0.7)  # short pause so speech finishes before listening again

    def get_my_voice(self):
        """Listen to microphone and return recognized text."""
        r = sr.Recognizer()
        with sr.Microphone(device_index=2) as src:
            r.adjust_for_ambient_noise(src, duration=0.5)
            print("Listening.......", end="", flush=True)
            r.pause_threshold = 1.0
            r.phrase_threshold = 0.3
            r.sample_rate = 48000
            r.dynamic_energy_threshold = True
            r.operation_timeout = 5
            r.non_speaking_duration = 0.5
            r.dynamic_energy_adjustment = 2
            r.energy_threshold = 4000
            r.phrase_time_limit = 10

            audio = r.listen(src)

            try:
                print("\r", end="", flush=True)
                print("recognizing...", flush=True)
                self.query = r.recognize_google(audio, language='en-US')
                print(f"You said: {self.query}")
                return self.query
            except Exception as e:
                print("say that again")
                print(e)
                return None

    def open_app(self, command):
        """Handle app opening or conversation mode."""
        try:
            if 'open notepad' in command.lower():
                self.speak_action("Opening Notepad")
                os.startfile('C:\\Windows\\System32\\notepad.exe')

            elif 'open calculator' in command.lower():
                self.speak_action("Opening Calculator")
                os.startfile('C:\\Windows\\System32\\calc.exe')

            elif 'open whatsapp' in command.lower():
                self.speak_action("Opening WhatsApp")
                webbrowser.open("https://www.whatsapp.com/")

            elif 'open instagram' in command.lower():
                self.speak_action("Opening Instagram")
                webbrowser.open("https://www.instagram.com/")

            elif 'open x' in command.lower():
                self.speak_action("Opening X")
                webbrowser.open("https://www.x.com/")

            elif 'switch to conversation' in command.lower():
                self.speak_action("Switching to conversation mode")
                while True:
                    actual_message = self.get_my_voice()
                    if actual_message is None:
                        print("no message")
                        continue
                    if 'exit' in actual_message.lower():
                        self.speak_action("Exiting conversation mode")
                        break
                    response = evaluate(actual_message)
                    if response:
                        decoded = ' '.join(response)
                        print(f"decoded sentence: {decoded}")
                        self.speak_action(decoded)
                    else:
                        self.speak_action("I can't produce a response right now.")

        except Exception as e:
            print(e)
            print("unable to open app")


# -------- Main program loop --------
cmd = VoiceCommand()
cmd.speak_action("Hello my master Lai, I'm your virtual assistant. How can I help you today?")

while True:
    query = cmd.get_my_voice()
    if query:
        if "exit" in query.lower():
            cmd.speak_action("Goodbye, master Lai.")
            break
        cmd.open_app(query)
