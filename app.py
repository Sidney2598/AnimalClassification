from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform
from fastai.learner import load_learner

plt=platform.system()
if plt=='Linux':pathlib.WindowsPath=pathlib.PosixPath
# temp=pathlib.PosixPath
# pathlib.PosixPath=pathlib.WindowsPath


st.title('Mening ikkinchi loyiham')
file=st.file_uploader('rasm yuklash',type=['png','jpeg','gif','svg'])
if file:
    st.image(file)
    img=PILImage.create(file)
    #model
    model=load_learner('transport_2_loyiha.pkl',cpu=True)
    predict, pred_id, probs=model.predict(img)
    st.success(f'Bashorat:{predict}')
    st.info(f'Ehtimollik:{probs[pred_id]*100:.1f}%')    
    #plotly
    fig=px.bar(x=probs,y=model.dls.vocab)
    st.plotly_chart(fig)
    

