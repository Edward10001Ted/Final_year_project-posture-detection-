import tensorflow as tf
import numpy as np
import time
import cv2

# IP camera URL (edit here)
ip_url = "http://10.70.35.132:8080/video"

# 1. Load model
model = tf.keras.models.load_model(r'face\face_detection_model.keras')

# 2. Per-frame prediction helper ---------------------------------------------
preprocess_input = tf.keras.applications.resnet.preprocess_input


def predict_frame(frame):
	"""Preprocess an OpenCV BGR frame and run the model. Returns (label, score)."""
	# Convert BGR to RGB and resize to model input
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	resized = cv2.resize(rgb, (160, 160))

	# Convert to array, expand dims and preprocess
	arr = tf.keras.utils.img_to_array(resized)
	arr = np.expand_dims(arr, 0)
	arr = preprocess_input(arr)

	pred = model.predict(arr)
	score = float(pred[0][0])
	label = "Sena" if score > 0.5 else "Edward"
	return label, score


def run_stream_and_predict(url: str):
	"""Use the IP camera URL as input frames, run the model on each frame,
	and display the annotated video. Press 'q' to quit."""
	cap = cv2.VideoCapture(url)
	if not cap.isOpened():
		print(f"Unable to open stream: {url}")
		return

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				print("No frame received, stopping stream")
				break

			# Run model on frame
			try:
				label, score = predict_frame(frame)
				text = f"{label}: {score:.2f}"
				# Print prediction to terminal
				print(text)
				cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
							0.9, (0, 255, 0), 2)
			except Exception as e:
				# If prediction fails, show error on frame but continue
				cv2.putText(frame, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
							0.6, (0, 0, 255), 2)

			cv2.imshow("Phone Camera", frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	run_stream_and_predict(ip_url)
