import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained VGG16 model
model = VGG16(weights="imagenet", include_top=False)  # Load without the top classification layers

def extract_features(video_path):
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

    # Convert BGR frame to RGB
    middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)

    # Use PIL for temporary image handling
    from PIL import Image
    temp_image_path = "temp_frame.jpg"
    Image.fromarray(middle_frame).save(temp_image_path)

    # Load the temporary image
    img = image.load_img(temp_image_path, target_size=(224, 224))

    # Convert image to array and ensure it has 3 channels (RGB)
    x = image.img_to_array(img)
    if x.ndim < 3:  # Check if it has fewer than 3 dimensions (e.g., grayscale)
        x = np.expand_dims(x, axis=-1)  # Add a channel dimension if needed
    x = np.expand_dims(x, axis=0)  # Add a batch dimension

    # Preprocess the image (resize, normalization) as needed for the model
    # ... (replace with your specific preprocessing steps) ...

    # Extract features
    features = model.predict(x)

    # Remove temporary image file
    os.remove(temp_image_path)

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
