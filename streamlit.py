import streamlit as st
from keras.preprocessing import image as keras_image
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the quantized model
quantized_model_path = 'humans_small_model_quantized.tflite'
interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
interpreter.allocate_tensors()

class_names = ['Mirha', 'Saad', 'Eman']

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Resize the image to the target size
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_class(img_path, class_names):
    x = load_and_preprocess_image(img_path)
    
    # Quantize the input image
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_tensor_index, x.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_tensor_index = interpreter.get_output_details()[0]['index']
    predictions = interpreter.get_tensor(output_tensor_index)

    # Find the predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class, predictions

# Rest of your Streamlit app code remains unchanged...
  # Return predictions as well

# Streamlit app
st.title("Human Classification App")
# Show example images for each class
st.subheader("Example Images:")
example_images = [
    'Mirha.53.jpg',
    'Saad.39.jpg',
    'Eman.51.jpg'
]
for i, img_path in enumerate(example_images):
    st.image(Image.open(img_path), caption=f"Example {class_names[i]} Image.", use_column_width=True)

# Upload image through Streamlit
uploaded_file = st.file_uploader("Upload Image to make predictions...", type="jpg")

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the class
    predicted_class, predictions = predict_class(uploaded_file, class_names)

    # Display prediction result
    st.write(f"Predicted class: {predicted_class}")

    # Display the probabilities for each class
    st.write("Class Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]:.4f}")
