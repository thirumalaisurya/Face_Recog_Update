import streamlit as st
import os
import cv2
import face_recognition
import matplotlib.pyplot as plt

# Function to recognize faces in an image
def recognize_face(image_path, dataset_path):
    # Load test image
    imgTest = face_recognition.load_image_file(image_path)
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

    # Encode test image
    encodeTest = face_recognition.face_encodings(imgTest)[0]

    # Initialize variables for similarity distance
    similarity_results = []
    face_distances = []

    # Iterate through images in the dataset
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                # Load and encode image from dataset
                imgDataset = face_recognition.load_image_file(image_path)
                encodeDataset = face_recognition.face_encodings(imgDataset)[0]

                # Compare faces
                results = face_recognition.compare_faces([encodeDataset], encodeTest)
                if results[0]:
                    # Calculate face similarity distance
                    face_distance = face_recognition.face_distance([encodeDataset], encodeTest)
                    similarity_results.append(results[0])
                    face_distances.append(face_distance[0])
                    return person_name, imgDataset, similarity_results, face_distances  # Return the name, image, similarity results, and face distances if a match is found

    return "Unknown", None, [], []  # Return "Unknown" if no match is found

def main():
    st.title("Face Recognition App")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary location
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform face recognition
        recognized_person, recognized_image, similarity_results, face_distances = recognize_face("temp_image.jpg", "dataset")

        # Display the recognized image with rectangle boxes and labels
        if recognized_image is not None:
            recognized_image = cv2.cvtColor(recognized_image, cv2.COLOR_BGR2RGB)
            st.image(recognized_image, caption=f"Recognized Person: {recognized_person}")
            if recognized_person != "Unknown":
                st.write(f"Face similarity with {recognized_person}: {similarity_results} | Distance: {face_distances[0]}")
        else:
            st.write("No matching person found in the dataset.")

if __name__ == "__main__":
    main()
