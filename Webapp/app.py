import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

IMG_SIZE = 150

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

# Set up model
# Build model: 4 conv layer of 64 nodes and 1 dense layer
def load_model(input_shape):
    model = Sequential()
    for i in range(4):
        model.add(Conv2D(64, (3,3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # Load weight
    model.load_weights('cnn-bal-aug-D1-C4-N64.h5')

    return model


def main():
    st.title("Pneumonia X-Ray Image Detection")
    uploaded_file = st.file_uploader("Please upload the x-ray image", type=['jpg','png','jpeg'])

    if uploaded_file:
        img = load_image(uploaded_file)
        # Show image
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        #st.write("Loading Image")

        img = np.array(img.convert('RGB'))
        img = cv2.cvtColor(img,1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X = []
        X.append(img)
        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        X = X/255

        #st.write("Predicting")

        model = load_model(X.shape[1:])

        # Predict and output result
        prediction_prob = model.predict(X)[0][0]
        if prediction_prob > 0.5:
            st.write("Prediction: Positive")
        else:
            st.write("Prediction: Negative")
        st.write("Confidence: " + str(prediction_prob))

main()