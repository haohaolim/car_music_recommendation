#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade pip


# In[2]:


get_ipython().system('pip install streamlit')



# In[10]:


get_ipython().system('pip show streamlit')


# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st
from keras.utils import to_categorical


# In[13]:


get_ipython().system('pip install streamlit')
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# Read the data (assuming you have it in a CSV file)
# Replace 'your_data.csv' with the actual file path or URL of your dataset
final = pd.read_csv('/FYP/preprocessed_data.csv')

# Preprocess the data (similar to what you have done)
X = final.drop('title', axis=1)  # Features
y = final['title']  # Target variable

categorical_vars = ['DrivingStyle', 'landscape', 'mood', 'naturalphenomena', 'RoadType', 'sleepiness', 'trafficConditions', 'weather', 'artist', 'category_name']
label_encoder = LabelEncoder()

for var in categorical_vars:
    X[var] = label_encoder.fit_transform(X[var])

label_encoder_y = LabelEncoder()
encoded_y = label_encoder_y.fit_transform(y)
num_classes = len(label_encoder_y.classes_)
y_encoded = to_categorical(encoded_y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

# Create a function for the ANN model and training
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Input layer
    model.add(Dense(64, activation='relu'))  # Hidden layer
    model.add(Dense(64, activation='relu'))  # Additional hidden layer
    model.add(Dense(64, activation='relu'))  # Additional hidden layer
    model.add(Dense(num_classes, activation='softmax'))  # Output layer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
def train_model():
    model = create_model()
    model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=0)
    return model

# Evaluate the model
def evaluate_model(model):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy

# Streamlit App
def main():
    st.title("Simple ANN GUI with Streamlit")

    # Optionally, you can add some introductory text or explanation here
    st.write("Welcome to the Simple ANN GUI! This app demonstrates how to create an ANN model using Streamlit.")

    # Training and evaluating the model
    st.write("Training the model...")
    model = train_model()
    st.write("Model trained successfully!")

    st.write("Evaluating the model...")
    loss, accuracy = evaluate_model(model)
    st.write(f'Test Loss: {loss:.4f}')
    st.write(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()


# In[ ]:




