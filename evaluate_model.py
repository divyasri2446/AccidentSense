import cv2
from matplotlib import cm
import numpy as np
from detection import AccidentDetectionModel
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_JSON_FILE = "model.json"
MODEL_WEIGHTS_FILE = "model_weights.keras"
VIDEO_PATH = "test_video.mp4" # IMPORTANT: Use a video the model has NOT seen before
CONFIDENCE_THRESHOLD = 65
ACCIDENT_FRAMES = [(170,235)] # Example: Accident happens between frame 550 and 650


def is_frame_an_accident(frame_number, accident_ranges):
    """Checks if a frame number falls within any of the true accident ranges."""
    for start, end in accident_ranges:
        if start <= frame_number <= end:
            return True
    return False

def main():
    # Load the model
    model = AccidentDetectionModel(MODEL_JSON_FILE, MODEL_WEIGHTS_FILE)
    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    y_true = []  # List to store the true labels (1 for Accident, 0 for No Accident)
    y_pred = []  # List to store the model's predictions

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("Finished processing video.")
            break

        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        # Get model prediction
        pred_class, prob = model.predict_accident(roi[np.newaxis, :, :])
        prob_percent = prob[0][0] * 100

        
        model_says_accident = (pred_class == "Accident" and prob_percent > CONFIDENCE_THRESHOLD)

        # --- LOGIC TO DETERMINE GROUND TRUTH ---
        truth_is_accident = is_frame_an_accident(frame_count, ACCIDENT_FRAMES)

        # Append results to our lists
        y_true.append(1 if truth_is_accident else 0)
        y_pred.append(1 if model_says_accident else 0)

        frame_count += 1

    video.release()

    # --- CALCULATE AND PRINT METRICS ---
    if not y_true:
        print("No frames were processed. Cannot calculate metrics.")
        return
        
    print("\n--- Model Evaluation Report ---")
    # Use scikit-learn's classification_report for a detailed summary
    # target_names tells the report what to call class 0 and class 1
    # report = classification_report(y_true, y_pred, target_names=["No Accident", "Accident"])
    # This is the fix:
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=["No Accident", "Accident"], 
        labels=[0, 1]  # <--- ADD THIS PART
    )
    print(report)

    # Generate and display a Confusion Matrix for a visual report
        # Generate and display a Confusion Matrix for a visual report
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xticks([0, 1], ["No Accident", "Accident"])
    plt.yticks([0, 1], ["No Accident", "Accident"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    


if __name__ == '__main__':
    main() 

















    
# import cv2
# import numpy as np
# from detection import AccidentDetectionModel
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # --- CONFIGURATION ---
# MODEL_JSON_FILE = "model.json"
# MODEL_WEIGHTS_FILE = "model_weights.keras"
# VIDEO_PATH = "test_video.mp4" 

# # CRITICAL 1: Adjust this value (e.g., 0.30 to 0.50) to optimize Recall for the 'Accident' class.
# CONFIDENCE_THRESHOLD = 0.25
# ACCIDENT_FRAMES = [(170,235)] 

# # CRITICAL 2: Adjust this value (e.g., 5, 10, or 20) to balance the evaluation dataset.
# # Higher value = fewer 'No Accident' frames are processed, leading to more meaningful metrics.
# SKIP_NON_ACCIDENT_FRAMES = 20


# def is_frame_an_accident(frame_number, accident_ranges):
#     """Checks if a frame number falls within any of the true accident ranges."""
#     for start, end in accident_ranges:
#         if start <= frame_number <= end:
#             return True
#     return False

# def main():
#     print("--- Accident Detection Model Evaluation ---")
#     print(f"Video: {VIDEO_PATH}")
#     print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
#     print(f"Non-Accident Frame Skip Rate (N): {SKIP_NON_ACCIDENT_FRAMES}")
    
#     # Load the model
#     try:
#         model = AccidentDetectionModel(MODEL_JSON_FILE, MODEL_WEIGHTS_FILE) 
#     except Exception as e:
#         print(f"Error loading model files: {e}")
#         return

#     video = cv2.VideoCapture(VIDEO_PATH)
#     if not video.isOpened():
#         print(f"Error: Could not open video file at {VIDEO_PATH}")
#         return

#     y_true = []  # True labels (1 for Accident, 0 for No Accident)
#     y_pred = []  # Model's predictions
#     frame_count = 0
#     processed_count = 0
    
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             print("\nFinished processing video.")
#             break

#         truth_is_accident = is_frame_an_accident(frame_count, ACCIDENT_FRAMES)
        
#         # --- SAMPLING LOGIC (CRITICAL FIX) ---
#         # If it's NOT an accident, skip if the frame index is not a multiple of the skip rate.
#         if not truth_is_accident and (frame_count % SKIP_NON_ACCIDENT_FRAMES != 0):
#             frame_count += 1
#             continue

#         # Preprocess the frame
#         try:
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             roi = cv2.resize(gray_frame, (250, 250))
#         except cv2.error as e:
#             print(f"Skipping frame {frame_count} due to preprocessing error: {e}")
#             frame_count += 1
#             continue
            
#         # Get model prediction
#         _, prob = model.predict_accident(roi[np.newaxis, :, :])
        
#         # --- LOGIC TO DETERMINE PREDICTION (THRESHOLD) ---
#         # Assuming prob[0][0] is the probability for the POSITIVE ('Accident') class
#         prob_accident = prob[0][0]
#         model_says_accident = (prob_accident > CONFIDENCE_THRESHOLD)

#         # Append results
#         y_true.append(1 if truth_is_accident else 0)
#         y_pred.append(1 if model_says_accident else 0)

#         frame_count += 1
#         processed_count += 1

#     video.release()

#     # --- CALCULATE AND PRINT METRICS ---
#     if processed_count == 0:
#         print("No frames were processed. Cannot calculate metrics.")
#         return
        
#     print(f"Total Frames Processed (Sampled): {processed_count}")
#     print("\n--- Model Evaluation Report ---")
    
#     # Use scikit-learn's classification_report for a detailed summary
#     report = classification_report(
#         y_true, 
#         y_pred, 
#         target_names=["No Accident", "Accident"],
#         zero_division=0 
#     )
#     print(report)

#     # Generate and display a Confusion Matrix for a visual report
#     print("\n--- Confusion Matrix ---")
#     cm = confusion_matrix(y_true, y_pred)
#     print(cm)
    
#     # Optional: Plot the confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(
#         cm, 
#         annot=True, 
#         fmt='d', 
#         cmap='Reds', 
#         xticklabels=["No Accident", "Accident"], 
#         yticklabels=["No Accident", "Accident"]
#     )
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title(f'Confusion Matrix (Sampled, Threshold={CONFIDENCE_THRESHOLD})')
#     plt.show()


# if __name__ == '__main__':
#     main()