import cv2
import json
from geopy.geocoders import Nominatim
from detection import AccidentDetectionModel
import numpy as np
import os
import winsound
import threading
import time
import tkinter as tk
from twilio.rest import Client
from PIL import Image, ImageTk  # Import PIL modules for image handling
from dotenv import load_dotenv

# This line reads your .env file and loads the secrets
load_dotenv()

emergency_timer = None
alarm_triggered = False  # Flag to track if an alarm has been triggered

model = AccidentDetectionModel("model.json", "model_weights.keras")
font = cv2.FONT_HERSHEY_SIMPLEX

def save_accident_photo(frame):
    try:
        current_date_time = time.strftime("%Y-%m-%d-%H%M%S")
        directory = "accident_photos"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"{directory}/{current_date_time}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Accident photo saved as {filename}")
    except Exception as e:
        print(f"Error saving accident photo: {e}")

def get_location_from_config():
    """Reads location data from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config['latitude'], config['longitude']
    except (FileNotFoundError, KeyError):
        print("ERROR: config.json or location data is missing.")
        return "0.0", "0.0" # Fallback location
def get_address_from_coords(lat, lon):
    """Performs reverse geocoding to get a readable address."""
    try:
        geolocator = Nominatim(
            user_agent="accident_detection_app",
            timeout=10
        )

        location = geolocator.reverse(
            f"{lat}, {lon}",
            exactly_one=True
        )

        if location:
            return location.address

        return f"Latitude: {lat}, Longitude: {lon}"

    except Exception as e:
        print(f"Reverse geocoding failed: {e}")

        return f"Latitude: {lat}, Longitude: {lon}"    

def call_ambulance():
    # 1. Get location (this part is the same)
    cam_latitude, cam_longitude = get_location_from_config()
    address = get_address_from_coords(cam_latitude, cam_longitude)
    print(f"Found Address: {address}")

    # 2. Load ALL secrets from the .env file
    try:
        account_sid = os.getenv("TWILIO_SID")
        auth_token = os.getenv("TWILIO_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")
        to_number = os.getenv("TWILIO_TO_NUMBER")
        twiml_url = os.getenv("TWILIO_TWIML_URL")
        
        # Check if any key is missing
        if not all([account_sid, auth_token, from_number, to_number, twiml_url]):
            print("ERROR: One or more Twilio environment variables are missing.")
            print("Please check your .env file.")
            return

    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return

    # 3. Build the final URL 
    final_url = (
        f"{twiml_url}?Address={address.replace(' ', '%20')}"
        f"&Latitude={cam_latitude}"
        f"&Longitude={cam_longitude}"
    )

    # 4. Make the call using the new variables
    try:
        client = Client(account_sid, auth_token)
        
        call = client.calls.create(
            url=final_url,
            to=to_number,
            from_=from_number
        )
        print(f"Call initiated with SID: {call.sid}")
    
    except Exception as e:
        print(f"Error calling ambulance: {e}")

def show_alert_message():
    def on_call_ambulance():
        call_ambulance()
        alert_window.destroy()

    # Play the beep sound
    frequency = 2500  
    duration = 2000  
    winsound.Beep(frequency, duration)

    alert_window = tk.Tk()
    alert_window.title("Alert")
    alert_window.geometry("500x250")  # Adjust window size to fit the GIF and message box
    alert_label = tk.Label(alert_window, text="Alert: Accident Detected!\n\nIs the Accident Critical?", fg="black", font=("Helvetica", 16))
    alert_label.pack() 
    call_ambulance_button = tk.Button(alert_window, text="Call Ambulance", command=on_call_ambulance)
    call_ambulance_button.pack()
    cancel_button = tk.Button(alert_window, text="Cancel", command=alert_window.destroy)
    cancel_button.pack()
    alert_window.mainloop()
    
def start_alert_thread():
    alert_thread = threading.Thread(target=show_alert_message)
    alert_thread.daemon = True  # Set the thread as daemon so it doesn't block the main thread
    alert_thread.start()

def startapplication():
    global alarm_triggered  # Use global variable for tracking alarm status
    video = cv2.VideoCapture("evaluation_video.mp4") 
    while True:
        ret, frame = video.read()
        if not ret:
            print("No more frames to read")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))


        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

      

        if pred == "Accident" and not alarm_triggered:
            prob = round(prob[0][0] * 100, 2)
            
            if prob > 80:
                # frequency = 2500  
                # duration = 2000  
                # winsound.Beep(frequency, duration)
                save_accident_photo(frame)
                alarm_triggered = True  # Set the alarm_triggered flag to True
                start_alert_thread()  # Start the alert message thread

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)  

if __name__ == '__main__':
    startapplication()

