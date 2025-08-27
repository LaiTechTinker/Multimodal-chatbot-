import os
import pyttsx3
import speech_recognition as sr
import webbrowser
from evaluate import evaluate

# ------------------ Text-to-Speech ------------------
def initialize_speak():
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # female voice
    engine.setProperty('rate', engine.getProperty('rate') - 50)
    engine.setProperty('volume', engine.getProperty('volume') + 0.5)
    return engine

def speak_action(engine, text):
    print(f"Assistant says: {text}")  # debug print
    engine.say(text)
    engine.runAndWait()

engine = initialize_speak()

# ------------------ Voice Input ------------------
def listen_once():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(device_index=2) as src:
            recognizer.adjust_for_ambient_noise(src, duration=0.5)
            print("Listening...", end="", flush=True)
            audio = recognizer.listen(src)
        query = recognizer.recognize_google(audio, language='en-US')
        print(f"\nYou said: {query}")
        return query
    except Exception as e:
        print("\nCould not recognize:", e)
        return None

# ------------------ Command Handling ------------------
def open_app(command):
    command = command.lower()
    try:
        if 'open notepad' in command:
            speak_action(engine, 'Opening Notepad')
            os.startfile('C:\\Windows\\System32\\notepad.exe')

        elif 'open calculator' in command:
            speak_action(engine, 'Opening Calculator')
            os.startfile('C:\\Windows\\System32\\calculator.exe')

        elif 'open whatsapp' in command:
            speak_action(engine, 'Opening WhatsApp')
            webbrowser.open("https://www.whatsapp.com/")

        elif 'open instagram' in command:
            speak_action(engine, 'Opening Instagram')
            webbrowser.open("https://www.instagram.com/")

        elif 'open x' in command:
            speak_action(engine, 'Opening X')
            webbrowser.open("https://www.X.com/")

        elif 'switch to conversation' in command:
            speak_action(engine, "Switching to conversation mode. Say 'exit' to stop.")
            while True:
                user_message = listen_once()
                if not user_message:
                    continue
                if any(word in user_message.lower() for word in ['exit', 'stop', 'quit']):
                    speak_action(engine, "Exiting conversation mode.")
                    break
                try:
                    response = evaluate(user_message)
                    speak_action(engine, ' '.join(response))
                except Exception as e:
                    print("Error in evaluate():", e)
                    speak_action(engine, "Sorry, I could not process that.")

        else:
            speak_action(engine, "Command not recognized. Please try again.")

    except Exception as e:
        print("Error in open_app:", e)
        speak_action(engine, "Sorry, I couldn't process that.")

# ------------------ Main Loop ------------------
if __name__ == "__main__":
    speak_action(engine, "Hello! I am your assistant. How can I help you today?")
    while True:
        command = listen_once()
        if command:
            open_app(command)
