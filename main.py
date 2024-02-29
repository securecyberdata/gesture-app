import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load pre-trained VGG16 model
model = VGG16(weights="imagenet", include_top=False)  # Load without the top classification layers

def extract_features(video_path):
    """
    Extracts features from the middle frame of a video using a pre-trained VGG16 model and OpenCV.

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

    # Preprocess the image (resize, normalization) as needed for the model
    # ... (replace with your specific preprocessing steps) ...

    # Extract features
    features = model.predict(x)

    return features.flatten()  # Flatten the feature vector



def main():
    # Define paths to training and test video folders
    train_video_folder = "traindata/train"
    test_video_folder = "traindata/test"

    # Create empty lists for training features and labels
    training_features = []
    training_labels = []

    # Extract features and labels for training videos
    for video_path in os.listdir(train_video_folder):
        # Assuming labels are the filenames (without extension)
        label = os.path.splitext(video_path)[0]

        features = extract_features(os.path.join(train_video_folder, video_path))
        if features is not None:
            training_features.append(features)
            training_labels.append(label)

    # Extract features for test videos
    testing_features = []
    testing_video_ids = []

    for video_path in os.listdir(test_video_folder):
        features = extract_features(os.path.join(test_video_folder, video_path))
        if features is not None:
            testing_features.append(features)
            testing_video_ids.append(video_path)

    # Calculate cosine similarity and find most similar training videos
    similarities = cosine_similarity(testing_features, training_features)
    predicted_labels = [training_labels[i] for i in np.argmax(similarities, axis=1)]

    # Prepare and save results
    results = zip(testing_video_ids, predicted_labels)
    with open("Results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Video ID", "Predicted Gesture"])
        writer.writerows(results)

    print("Gesture recognition completed. Results saved to Results.csv.")


if __name__ == "__main__":
    import os
    import csv

    main()
