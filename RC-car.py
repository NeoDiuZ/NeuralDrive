import time
import asyncio
import numpy as np
import websockets
from sklearn.svm import OneClassSVM
from neuropy3.neuropy3 import MindWave
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread
import subprocess
import psutil
import keyboard  # Add this import at the top

app = Flask(__name__)
CORS(app)

# WebSocket URI for ESP32 (replace with your ESP32's IP address)
ESP32_URI = "ws://172.20.10.2:81"  # Replace with the actual IP address

# Define EEG features (can be adjusted based on your data)
FEATURES = ['delta', 'theta', 'alpha_l', 'alpha_h', 'beta_l', 'beta_h']

# Global variables
model = None
last_function_call_time = time.time()
ranges = {
    'high': 75,
    'medium': 50,                           
    'low': 25
}

latest_attention_value = 0  # This variable will hold the latest attention value

# Function to extract features from EEG data
def extract_features(eeg_data): 
    return np.array([eeg_data[f] for f in FEATURES]).reshape(1, -1)

async def send_command_via_websocket(command):
    """Send command to ESP32 via WebSocket."""
    try:        
        async with websockets.connect(ESP32_URI) as websocket:
            await websocket.send(command)
            print(f"Sent command via WebSocket: {command}")
    except Exception as e:
        print(f"Failed to send command via WebSocket: {e}")

def A():
    print(f"Function A: Attention is high (more than {ranges['high']})")
    asyncio.run(send_command_via_websocket('A'))

def B():
    print(f"Function B: Attention is medium-high (more than {ranges['medium']} but less than or equal to {ranges['high']})")
    asyncio.run(send_command_via_websocket('B'))

def C():
    print(f"Function C: Attention is medium-low (more than {ranges['low']} but less than or equal to {ranges['medium']})")
    asyncio.run(send_command_via_websocket('C'))

def D():
    print(f"Function D: Attention is low (more than 0 but less than or equal to {ranges['low']})")
    asyncio.run(send_command_via_websocket('D'))                                                                                                                                                           

# Callback function to handle EEG data
def eeg_callback(data):
    global model

    # Extract features from EEG data
    features = extract_features(data)

    # If model is not trained, train it on initial data (you should collect enough data points first)
    if model is None:
        # Initialize an empty list for initial data
        initial_data = []

        # Collect initial data for training
        initial_data.append(features.flatten())

        # Train the model once we have enough data points
        if len(initial_data) >= 10:  # Example threshold; adjust as needed
            model = OneClassSVM(nu=0.1)
            model.fit(initial_data)
            print("Model trained on initial data")
    else:
        # Predict if the model is trained
        prediction = model.predict(features)

        # Only process data classified as normal
        if prediction[0] == 1:
            print(f"EEG: {data}")
        else:
            print("EEG data classified as erratic, ignoring.")

# Callback function to handle meditation data
def meditation_callback(value):
    print("Meditation: ", value)

# Callback function to handle attention data
def attention_callback(value):
    global last_function_call_time, ranges, latest_attention_value  # Include latest_attention_value

    print("Attention: ", value)
    latest_attention_value = value  # Update the latest attention value

    current_time = time.time()
    if current_time - last_function_call_time >= 1:
        if value > ranges['high']:
            A()
        elif value > ranges['medium']:
            B()
        elif value > ranges['low']:
            C()
        else:
            D()
        last_function_call_time = current_time

@app.route('/get_attention', methods=['GET'])
def get_attention():
    return jsonify({"attention": latest_attention_value}), 200

# Flask routes
@app.route('/update_ranges', methods=['POST'])
def update_ranges():
    global ranges
    new_ranges = request.json
    ranges['high'] = new_ranges.get('high', ranges['high'])
    ranges['medium'] = new_ranges.get('medium', ranges['medium'])
    ranges['low'] = new_ranges.get('low', ranges['low'])
    print(f"Ranges updated: {ranges}")
    return jsonify({"message": "Ranges updated successfully"}), 200

@app.route('/get_ranges', methods=['GET'])
def get_ranges():
    return jsonify(ranges), 200

def run_flask():
    app.run(debug=False, port=5000)

def main():
    global last_function_call_time

    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Initialize the MindWave device
    mw = MindWave(address='A4:DA:32:70:03:4E', autostart=False, verbose=3)

    # Set up callbacks
    mw.set_callback('eeg', eeg_callback)
    mw.set_callback('meditation', meditation_callback)
    mw.set_callback('attention', attention_callback)

    # Start the device and collect data
    mw.start()

    # Keep the script running
    try:
        while True:
            time.sleep(0.1)  # Add a small delay to prevent high CPU usage
    except KeyboardInterrupt:
        print("Script interrupted and stopped.")
    finally:
        mw.stop()
        # Optionally, you can add a way to stop the Flask thread here

if __name__ == "__main__":
    main()