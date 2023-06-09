import streamlit as st
import tensorflow as tf
import os

def main():
    st.title("Baggage Detection App")

    # Load the trained model
    #model_version = max([int(i) for i in os.listdir("models") + [0]])
    model_path = f"models/1"
    model = tf.keras.models.load_model(model_path)

    # Upload and classify images
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg"])
    if uploaded_image is not None:
        # Perform image classification using the uploaded image and the loaded model
        image = tf.io.decode_image(uploaded_image.read(), channels=3)
        image = tf.image.resize(image, [640, 640])
        image = image / 255.0  # Normalize the image
        image = tf.expand_dims(image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image)
        labels = ['Gun', 'Knife', 'Pliers', 'Scissors', 'Wrench']  # Replace with your actual labels
        

        # Display the results
        st.subheader("Prediction Results:")
        for i, label in enumerate(labels):
            probability = predictions[0][i]
            st.write(f"{label}: {probability}")

        # Display the uploaded image
        st.image(image.numpy().squeeze(), channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()
