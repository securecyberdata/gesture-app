import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load pre-trained VGG16 model, handling potential GPU issues
try:
    model = VGG16(weights="imagenet", include_top=False)
except OSError as e:
    if "Could not find CUDA drivers" in str(e):
        print("Warning: Could not find CUDA drivers, GPU will not be used.")
    else:
        raise e  # Re-raise other errors

def extract_features(video_path):
    """
    Extracts features from the middle frame of a video using VGG16 and OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        np.ndarray: Flattened feature vector, or None if an error occurs.
    """

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, middle_frame = cap.read()

    if not ret:
        print(f"Error reading frame from video {video_path}. Skipping.")
        return None

    # Handle empty frames
    if middle_frame is None:
        print(f"Empty frame captured from video {video_path}. Skipping.")
        return None

    # Preprocess using OpenCV
    middle_frame = cv2.resize(middle_frame, (224, 224))  # Resize to match VGG16 input size
    middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert to NumPy array and add batch dimension
    x = np.expand_dims(middle_frame, axis=0)

    # Preprocess the image (e.g., normalization, mean subtraction) as needed for the model
    # ... (replace with your specific preprocessing steps based on your model) ...

    # Extract features
    features = model.predict(x)

    return features.flatten()  # Flatten the feature vector


def predict_gesture(video_path, reference_features, gesture_names):
    """
    Predicts the gesture in a video based on the nearest neighbor approach with cosine similarity.

    Args:
        video_path (str): Path to the video file.
        reference_features (list): List of pre-computed features for reference videos.
        gesture_names (list): List of corresponding gesture names for reference videos.

    Returns:
        str: Predicted gesture name, or None if an error occurs.
    """

    try:
        # Extract features from the test video
        test_features = extract_features(video_path)
        if test_features is None:
            return None

        # Calculate cosine similarities with each reference feature
        similarities = cosine_similarity(test_features.reshape(1, -1), reference_features)

        # Find the index of the most similar reference video
        most_similar_index = np.argmax(similarities)

        # Predict the gesture based on the index
        return gesture_names[most_similar_index]

    except Exception as e:
        print(f"Error predicting gesture for video {video_path}: {e}")
        return None


if __name__ == "__main__":
    # Replace these placeholders with your actual implementations
    # - Load reference features and gesture names from training data
    reference_features = []
    gesture_names = []

    # Process test videos
    predicted_gestures = []
    for filename in os.listdir("traindata"):
        video_path = os.path.join("traindata", filename)
        predicted_gesture = predict_gesture(video_path, reference_features, gesture_names)
        predicted_gestures.append(predicted_gesture)

    # Write results to CSV, using newline='' to avoid extra line
    with open("Results.csv", "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(predicted_gestures)

    print("Results written to Results.csv")
