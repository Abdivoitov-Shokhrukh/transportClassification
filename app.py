import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



#Title
st.title("Transport klassifikatsiya qiluvchi model")

#Image uploading

file = st.file_uploader("Rasm yuklash", type = ['png','jpeg','gif','svg'])
st.image(file)
if file:

    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner("transport_model.pkl")

    #Prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    #Plotting
    fig = px.bar(x=probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)

