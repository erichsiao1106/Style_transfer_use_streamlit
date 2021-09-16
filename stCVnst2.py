import cv2
import streamlit as st


import streamlit as st
import cv2 as cv
import tempfile

f = st.file_uploader("Upload file")
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(f.read())
vf = cv.VideoCapture(tfile.name)



cap = cv2.VideoCapture(0)
st.title('Streamlit + CV2')
run = st.checkbox('請打勾來執行')
FRAME_WINDOW = st.image([])
while run:
    success, frame = cap.read()
    FRAME_WINDOW.image(frame, channels= 'BGR')

cap.release()
