import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow as tf

# Replace these paths with your actual file locations
TRAIN_VIDEO_DIR = "traindata"  # Folder containing training videos
TEST_VIDEO_DIR = "testdata"  # Folder containing test videos
RESULTS_CSV = "Results.csv"  # Output file for recognized gestures

# Function to generate training features using a pre-trained model (replace if needed)

@tf.function(reduce_retracing=True)
def extract_hand_shape_features(frame):
    """
    Extracts hand shape features from a given frame.

    Args:
        frame: A NumPy array representing the input frame.

    Returns:
        A NumPy array containing the extracted hand shape features.
    """

    # Load the VGG16 model using Keras
    model = VGG16(weights='imagenet')

    # Preprocess the frame
    # Resize to 224x224 to match model input expectations
    frame_resized = cv2.resize(frame, (224, 224))
    # Reshape to BGR format for VGG16
    frame_processed = preprocess_input(frame_resized[:, :, ::-1])
    # Expand dimensions to create a batch of 1 image
    frame_processed = np.expand_dims(frame_processed, axis=0)

    # Extract features using model's predict method
    feature_vector = model.predict(frame_processed)

    # Flatten the feature vector
    return feature_vector.flatten()



# Function to process a video and extract its penultimate layer (feature vector)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize variables to store features and video length
    features = []
    num_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features from the middle frame (adjust as needed)
        if num_frames == cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2:
            features.append(extract_hand_shape_features(frame))

        num_frames += 1

    cap.release()

    # Ensure at least one frame was processed
    if not features:
        raise ValueError("No frames processed from video:", video_path)

    # Return the feature vector as a NumPy array
    return np.array(features).flatten()

# Function to perform gesture recognition using cosine similarity
def recognize_gestures(test_features, train_features):
    # Calculate cosine similarities between test features and each training feature
    similarities = cosine_similarity(test_features.reshape(1, -1), train_features)

    # Find the index of the most similar training feature
    most_similar_idx = np.argmax(similarities)

    # Return the gesture label (replace with your actual mapping)
    return most_similar_idx

def generate_train_features():
    train_features = []
    for video_name in os.listdir(TRAIN_VIDEO_DIR):
        video_path = os.path.join(TRAIN_VIDEO_DIR, video_name)
        features = process_video(video_path)
        train_features.append(features)

    train_features = np.array(train_features)
    np.save("train_features.npy", train_features)

# Check if training features file exists, generate if not
if not os.path.exists("train_features.npy"):
    generate_train_features()

# Load training features
train_features = np.load("train_features.npy")

# Create the results CSV file (delete existing file if it exists)
if os.path.exists(RESULTS_CSV):
    os.remove(RESULTS_CSV)
with open(RESULTS_CSV, "w", newline="") as f:
    f.write("Video Name,Gesture Label\n")

# Process each test video, recognize gestures, and save results
for video_name in os.listdir(TEST_VIDEO_DIR):
    video_path = os.path.join(TEST_VIDEO_DIR, video_name)
    test_features = process_video(video_path)

    try:
        gesture_label = recognize_gestures(test_features, train_features)
    except ValueError as e:
        print(f"Error processing video: {video_name}, {e}")
        gesture_label = -1  # Indicate error

    with open(RESULTS_CSV, "a", newline="") as f:
        f.write(f"{video_name},{gesture_label}\n")

print("Gesture recognition completed. Results saved to:", RESULTS_CSV)
