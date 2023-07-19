#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd


# In[3]:


# App title
st.title('My Streamlit App')

# Load data (replace 'data.csv' with your data file)
data = pd.read_csv('/FYP/preprocessed_data.csv')

# Display DataFrame
st.dataframe(data)


# In[ ]:




