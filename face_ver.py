import tensorflow as tf
import numpy as np
import time
import cv2
import csv
import os
from datetime import datetime
import prediction_store
try:
	from ultralytics import YOLO
	_yolo_available = True
except Exception:
	YOLO = None
	_yolo_available = False

# Try to load MediaPipe pose landmarker (task file) or fall back to mediapipe.solutions.pose
_pose_task_available = False
_pose_sol_available = False
pose_landmarker = None
pose_detector = None
try:
	from mediapipe.tasks.python import vision as mp_vision
	from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, VisionRunningMode
	# try to create landmarker from task file if present
	try:
		opts = PoseLandmarkerOptions(model_asset_path='pose_landmarker_full.task', running_mode=VisionRunningMode.IMAGE)
		pose_landmarker = PoseLandmarker.create_from_options(opts)
		_pose_task_available = True
		print('Loaded MediaPipe Pose Landmarker from task file')
	except Exception as _e:
		pose_landmarker = None
except Exception:
	# Try fallback to mediapipe.solutions.pose
	try:
		import mediapipe as mp
		pose_detector = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
		_pose_sol_available = True
		print('Using mediapipe.solutions.pose for pose gating')
	except Exception:
		pose_detector = None

# IP camera URL (edit here)
ip_url = "http://10.70.35.132:8080/video"

# 1. Compatibility: provide a `GetItem` custom layer if the saved model used it.
class  GetItem(tf.keras.layers.Layer):
	def __init__(self, index=None, **kwargs):
		super().__init__(**kwargs)
		self.index = index

	def call(self, inputs):
		try:
			if self.index is None:
				return inputs
			# support single int or tuple/list
			return inputs[self.index]
		except Exception:
			# fallback to tensor gather on axis 0 for numeric index
			try:
				return tf.gather(inputs, tf.constant(self.index))
			except Exception:
				return inputs

	def get_config(self):
		cfg = super().get_config()
		cfg.update({'index': self.index})
		return cfg

# Compatibility: provide a simple `Stack` custom layer if the saved model used it.
class Stack(tf.keras.layers.Layer):
	def __init__(self, axis=0, **kwargs):
		super().__init__(**kwargs)
		self.axis = axis

	def call(self, inputs):
		try:
			# If inputs is a list/tuple of tensors, stack them
			if isinstance(inputs, (list, tuple)):
				return tf.stack(inputs, axis=self.axis)
			# If already a tensor, return as-is
			return inputs
		except Exception:
			try:
				return tf.stack(inputs, axis=self.axis)
			except Exception:
				return inputs

	def get_config(self):
		cfg = super().get_config()
		cfg.update({'axis': self.axis})
		return cfg

# Register an Ellipsis placeholder so Keras deserializer can construct an Ellipsis object
import builtins as _builtins

class EllipsisPlaceholder:
	@classmethod
	def from_config(cls, config):
		return _builtins.Ellipsis

	def get_config(self):
		return {}

# 1. Lazy-load model to avoid import-time execution when Keras imports this module
model = None
def load_model_if_needed():
	global model
	if model is None:
		try:
			# Try loading with custom_objects - map 'Ellipsis' to our placeholder
			custom_objs = {
				'GetItem': GetItem,
				'Stack': Stack,
				'Ellipsis': EllipsisPlaceholder
			}
			# Try the .keras format first (newer format)
			if os.path.exists('face_detection_model.keras'):
				try:
					model = tf.keras.models.load_model('face_detection_model.keras', custom_objects=custom_objs, safe_mode=False)
					print("Loaded model from face_detection_model.keras")
				except Exception as e1:
					print(f"Failed to load .keras model: {e1}, trying .h5...")
					model = tf.keras.models.load_model('face_detection_model.h5', custom_objects=custom_objs, safe_mode=False)
					print("Loaded model from face_detection_model.h5")
			else:
				model = tf.keras.models.load_model('face_detection_model.h5', custom_objects=custom_objs, safe_mode=False)
				print("Loaded model from face_detection_model.h5")
		except Exception as e:
			print(f"Error loading model: {e}")
			raise
	return model

# Load YOLOv8 detection model (for person gating)
det_model = None
if _yolo_available:
	try:
		det_model = YOLO('yolov8m.pt')
	except Exception as _e:
		print(f"Failed loading YOLOv8 model: {_e}")
		det_model = None
else:
	print("ultralytics YOLO not available; person detection gating disabled")

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

	# ensure model loaded
	mdl = load_model_if_needed()
	pred = mdl.predict(arr)
	score = float(pred[0][0])
	label = "Sena" if score > 0.5 else "Edward"
	return label, score


def check_pose(frame, min_landmarks=5, min_conf=0.5, require_face_landmark=True):
	"""Return True if a human pose with sufficient visible landmarks is detected.

	Uses MediaPipe Tasks PoseLandmarker when available, otherwise falls back to
	mediapipe.solutions.pose. If no pose detector is available, returns True
	(do not block verification).
	"""
	if not (_pose_task_available or _pose_sol_available):
		return True

	# Helper for task-based landmarker
	if _pose_task_available and pose_landmarker is not None:
		try:
			img = mp_vision.TensorImage.create_from_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			res = pose_landmarker.detect(img)
			if not res or res.pose_landmarks is None:
				return False
			# count landmarks with sufficient score/presence
			count = 0
			for lm in res.pose_landmarks:
				s = getattr(lm, 'score', None) or getattr(lm, 'presence', None)
				if s is None or s >= min_conf:
					count += 1
			if count < min_landmarks:
				return False
			if require_face_landmark:
				try:
					nose = res.pose_landmarks[0]
					left_eye = res.pose_landmarks[1]
					right_eye = res.pose_landmarks[2]
					nose_ok = (getattr(nose, 'score', 1.0) or 1.0) >= min_conf
					left_ok = (getattr(left_eye, 'score', 1.0) or 1.0) >= min_conf
					right_ok = (getattr(right_eye, 'score', 1.0) or 1.0) >= min_conf
					return nose_ok or (left_ok and right_ok)
				except Exception:
					return True
		except Exception as e:
			print(f"Pose task-landmarker error: {e}")

	# Fallback: mediapipe.solutions.pose
	if _pose_sol_available and pose_detector is not None:
		try:
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			res = pose_detector.process(rgb)
			if not res or res.pose_landmarks is None:
				return False
			count = 0
			for lm in res.pose_landmarks.landmark:
				vis = getattr(lm, 'visibility', None)
				if vis is None or vis >= min_conf:
					count += 1
			if count < min_landmarks:
				return False
			if require_face_landmark:
				try:
					nose = res.pose_landmarks.landmark[0]
					left_eye = res.pose_landmarks.landmark[1]
					right_eye = res.pose_landmarks.landmark[2]
					nose_ok = getattr(nose, 'visibility', 1.0) >= min_conf
					left_ok = getattr(left_eye, 'visibility', 1.0) >= min_conf
					right_ok = getattr(right_eye, 'visibility', 1.0) >= min_conf
					return nose_ok or (left_ok and right_ok)
				except Exception:
					return True
		except Exception as e:
			print(f"Pose detector error: {e}")

	return True



def run_stream_and_predict(url: str):
	"""Use the IP camera URL as input frames, run the model on each frame,
	and display the annotated video. Press 'q' to quit.

	Workflow:
	- 'detect' state: run YOLO person detection periodically. If person found -> 'verify'
	- 'verify' state: run face model on every frame for VERIFICATION_SECONDS, then return to 'detect'
	"""
	cap = cv2.VideoCapture(url)
	if not cap.isOpened():
		print(f"Unable to open stream: {url}")
		return

	# State machine
	VERIFICATION_SECONDS = 30
	state = 'detect'
	verify_end = 0
	detect_frame_skip = 3
	frame_count = 0
	
	# Prediction collection for bias application
	prediction_counts = {'Sena': 0, 'Edward': 0}
	prediction_scores = []
	EDWARD_BIAS = 0.15  # Bias factor for Edward (increase his count by this % to compensate for model bias)

	# Setup prediction logging to CSV
	log_file = 'predictions_log.csv'
	log_exists = os.path.exists(log_file)
	csv_writer = None
	log_f = open(log_file, 'a', newline='')
	csv_writer = csv.writer(log_f)
	if not log_exists:
		csv_writer.writerow(['timestamp', 'label', 'score'])
		log_f.flush()
	print(f"Logging predictions to {log_file}")

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				print("No frame received, stopping stream")
				break

			frame_count += 1
			
			# Debug: Show state transitions
			if state == 'verify' and frame_count == 1:
				print(f"[DEBUG] Entered verify state. verify_end={verify_end}, current_time={time.time()}, remaining={verify_end - time.time():.2f}s")

			if state == 'detect':
				person_found = False
				if det_model is not None and (frame_count % detect_frame_skip) == 0:
					try:
						results = det_model(frame, conf=0.3, verbose=False)
						for r in results:
							if hasattr(r, 'boxes') and r.boxes is not None:
								for b in r.boxes:
									# b.cls may be array-like or scalar
									cls_val = None
									try:
										cls_val = int(b.cls[0])
									except Exception:
										try:
											cls_val = int(getattr(b, 'cls'))
										except Exception:
											cls_val = None
									if cls_val == 0:
										person_found = True
										break
							if person_found:
								break
					except Exception as _e:
						print(f"YOLO detection failed: {_e}")

				if person_found:
					# if pose gating is available, require pose check to pass
					try:
						if (_pose_task_available or _pose_sol_available):
							pose_ok = check_pose(frame)
							if not pose_ok:
								person_found = False
					except Exception as _e:
						print(f"Pose gating error: {_e}")

				if person_found:
					state = 'verify'
					verify_end = time.time() + VERIFICATION_SECONDS
					print(f"Person detected — entering verification for {VERIFICATION_SECONDS} seconds")

				cv2.putText(frame, "State: DETECTING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
							0.8, (255, 255, 0), 2)

			elif state == 'verify':
				try:
					label, score = predict_frame(frame)
					# Collect prediction
					prediction_counts[label] += 1
					prediction_scores.append((label, score))
					
					text = f"{label}: {score:.2f}"
					print(text)
					try:
						prediction_store.write(label, score)
						# Log prediction to file
						if csv_writer:
							csv_writer.writerow([datetime.now().isoformat(), label, f"{score:.4f}"])
							log_f.flush()
					except Exception as _e:
						print(f"prediction_store write failed: {_e}")
					cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
								0.9, (0, 255, 0), 2)
				except Exception as e:
					print(f"[ERROR during prediction] {e}")
					cv2.putText(frame, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
								0.6, (0, 0, 255), 2)

				if time.time() >= verify_end:
					# Verification window ended - apply bias and get final prediction
					biased_counts = prediction_counts.copy()
					biased_counts['Edward'] *= (1 + EDWARD_BIAS)  # Apply Edward bias
					
					final_label = max(biased_counts, key=biased_counts.get)
					avg_score = sum(s for l, s in prediction_scores if l == final_label) / max(1, sum(1 for l, s in prediction_scores if l == final_label))
					
					print(f"\n===== VERIFICATION COMPLETE =====")
					print(f"Total predictions collected: {len(prediction_scores)}")
					print(f"Raw counts - Sena: {prediction_counts['Sena']}, Edward: {prediction_counts['Edward']}")
					print(f"With Edward Bias ({EDWARD_BIAS*100:.0f}%) - Sena: {biased_counts['Sena']:.1f}, Edward: {biased_counts['Edward']:.1f}")
					print(f"FINAL RESULT: {final_label} (avg score: {avg_score:.4f})")
					print(f"================================\n")
					
					# Reset for next verification window
					prediction_counts = {'Sena': 0, 'Edward': 0}
					prediction_scores = []
					state = 'detect'
					frame_count = 0
					print('Verification window ended — returning to person detection')

				cv2.putText(frame, f"State: VERIFY ({int(max(0, verify_end - time.time()))}s)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
							0.7, (0, 255, 255), 2)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()
		if log_f:
			log_f.close()
			print(f"Prediction log saved to {log_file}")


if __name__ == '__main__':
	run_stream_and_predict(ip_url)