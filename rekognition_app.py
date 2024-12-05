import streamlit as st
import boto3
from PIL import Image, ImageDraw

# AWS Rekognition Client Setup
rekognition_client = boto3.client('rekognition')


def detect_faces(image_bytes):
    response = rekognition_client.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )
    return response['FaceDetails']

def draw_boxes(image, face_details):
    draw = ImageDraw.Draw(image)
    for face in face_details:
        box = face['BoundingBox']
        width, height = image.size
        left = box['Left'] * width
        top = box['Top'] * height
        right = left + (box['Width'] * width)
        bottom = top + (box['Height'] * height)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
    return image

st.title("Amazon Rekognition: Face Detection")
st.write("Upload an image to detect faces using Amazon Rekognition.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    image_bytes = uploaded_file.read()

    with st.spinner("Analyzing..."):
        face_details = detect_faces(image_bytes)

    if face_details:
        st.success(f"Detected {len(face_details)} face(s).")

        # Draw bounding boxes
        boxed_image = draw_boxes(image, face_details)
        st.image(boxed_image, caption="Image with Detected Faces", use_column_width=True)

        # Display detailed results
        st.write("Face Details:")
        for i, face in enumerate(face_details):
            st.write(f"Face {i + 1}:")
            st.write(f" - Confidence: {face['Confidence']:.2f}%")
            st.write(" - Emotions:")
            for emotion in face['Emotions']:
                st.write(f"   - {emotion['Type']}: {emotion['Confidence']:.2f}%")
    else:
        st.warning("No faces detected.")
