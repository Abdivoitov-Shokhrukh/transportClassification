import streamlit as st
from fastai.vision.all import *



#Title
st.title("Transport klassifikatsiya qiluvchi model")

#Image uploading

file = st.file_uploader("Rasm yuklash", type = ['png','jpeg','gif','svg','jpg'])
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





