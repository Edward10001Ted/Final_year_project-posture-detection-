"""Simple reader example for prediction_store.

Run this alongside `face_ver.py` (in a separate terminal/process) to see realtime predictions.
It prints only when the prediction changes.
"""
import time
import signal
import sys
import prediction_store

running = True

def _sigint(_signum, _frame):
    global running
    running = False

signal.signal(signal.SIGINT, _sigint)

last = None
print("Starting prediction reader. Press Ctrl+C to exit.")
try:
    while running:
        pred = prediction_store.read()
        if pred != last and pred is not None:
            print(f"Prediction changed: {pred}")
            last = pred
        time.sleep(0.1)
finally:
    prediction_store.close()
    print("Reader exiting.")
