# Human Classification using CNN and Streamlit

This project involves training a Convolutional Neural Network (CNN) to classify images of three individuals - Mirha, Saad, and Eman. The trained model is then deployed in a Streamlit web application for real-time image classification.

## Overview

The project is divided into two main parts:

1. **Model Training (`3 images.ipynb`):**
   - The `3 images.ipynb` notebook contains the code for training a CNN using a dataset of human images.
   - The dataset is structured into training, validation, and test sets, each containing images of Mirha, Saad, and Eman.
   - The CNN architecture consists of convolutional layers with max pooling, followed by fully connected layers.
   - The model is trained using binary crossentropy loss and RMSprop optimizer.
   - After training, the model is saved as `humans_small_model.h5`.

2. **Streamlit Web Application (`streamlit.py`):**
   - The `streamlit.py` script deploys the trained model in a Streamlit web application for easy image classification.
   - Users can upload their own images to the app and receive real-time predictions for the class of the individual in the image.
   - The Streamlit app includes a simple user interface, class prediction display, and example images for each class.

## Project Structure

- `3 images.ipynb`: Jupyter notebook for model training.
- `streamlit.py`: Script for the Streamlit web application.
- `humans_small_model.h5`: Pre-trained CNN model for human classification.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/mahad-iqbal4443/humanclassification.git
    ```

2. Navigate to the project directory:

    ```bash
    cd humanclassification
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Model Training

- Open and run the `3 images.ipynb` notebook to train the CNN model.
- The trained model (`humans_small_model.h5`) will be saved in the project directory.

## Usage

- Run the Streamlit app to use the pre-trained model for real-time image classification:

    ```bash
    streamlit run streamlit.py
    ```

- Visit the provided local URL (usually http://localhost:8501) in your web browser to use the app.
- Upload images to receive predictions and explore example images.

Feel free to modify the code and adapt it to your specific needs.
