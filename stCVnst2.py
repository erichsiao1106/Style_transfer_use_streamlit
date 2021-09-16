import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import cv2 as cv
import tempfile

f = st.file_uploader("Upload file")

tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(f.read())


vf = cv.VideoCapture(tfile.name)

stframe = st.empty()

while vf.isOpened():
    ret, frame = vf.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    stframe.image(gray)
# pip install streamlit
# streamlit run stnst.py

st.title('Streamlit + Arbitrary Image Stylization')
source_name = st.sidebar.selectbox(
    '請選擇原始相片', ('1.jpg', '2.jpg', '3.jpg', '7.jpg', 'test1.jpg'))
style_name = st.sidebar.selectbox('請選擇風格圖片', ("andy.jpg", "candy.jpg", "composition.jpg", "la_muse.jpg",
                                              "mosaic.jpg", "starry_night.jpg", "the_wave.jpg", "ss.jpg", "ss2.jpg", "ss3.jpg"))

source_image = cv2.imread("model/" + source_name)
style_image = cv2.imread("model/" + style_name)

col1, col2 = st.beta_columns(2)
with col1:
   st.header("Source image:")
   st.image(source_image, channels='BGR')
with col2:
   st.header('Style image:')
   st.image(style_image, channels='BGR')

clicked = st.button('風格轉移')

if clicked:
    source_image = source_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))
    model = hub.load('model')
    out_image = model(tf.constant(source_image), tf.constant(style_image))
    st.write('Output image:')
    st.image(np.squeeze(out_image), width=400, channels= 'BGR')
