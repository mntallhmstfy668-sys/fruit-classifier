import os
import streamlit as st 
import numpy as np
import tensorflow as tf    
from tensorflow import keras     
from tensorflow.keras.models import load_model 
from PIL import Image
import pickle

base_path=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(base_path,"model_1.h5")
classes_path=os.path.join(base_path,"classes.pkl")

classes_indicies=pickle.load(open(classes_path,'rb'))
model=load_model(model_path)


st.set_page_config(
    page_title="Fresh vs Rotten Fruits",
    page_icon='🍎',
    layout="centered"
)

st.title("Fresh Vs Rotten Fruits 🍌",)
img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

def preprocessing_image(img):  
    img=Image.open(img)
    img=img.convert('RGB')
    resized_image=img.resize((128,128))
    img_array=np.asarray(resized_image)
    img_array=np.expand_dims(img_array,axis=0)
    rescaled_image=img_array.astype('float32')/255.
    return rescaled_image
  
  
  
  
if st.button("Predict"):
   image=preprocessing_image(img)
   prediction=model.predict(image)
   class_index=np.argmax(prediction,axis=1)[0]
   class_name=classes_indicies[class_index]
    
    
   st.success(class_name) 

    