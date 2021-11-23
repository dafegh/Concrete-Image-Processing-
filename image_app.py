# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:18:33 2021

@author: David
"""
# importing necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import streamlit as st 
from PIL import Image
fig = plt.figure()


st.title('Concrete Classifier using image processing')

   
def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    if file_uploaded is not None:    
        img = Image.open(file_uploaded)
        st.text('Orginal Image')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        

    img_new = np.array (img.convert('RGB'))
    gray = cv2.cvtColor(img_new,cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(3,3))
    img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
    img_log = np.array(img_log,dtype=np.uint8)
    bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)
    edges = cv2.Canny(bilateral,100,200)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(closing, None)
    featuredImg = cv2.drawKeypoints(closing, keypoints, None)
    plt.imshow(featuredImg,cmap='gray')
    st.text('Predicted Image')
    st.pyplot(fig)
    
    
   

    

if __name__ == "__main__":
    main()
