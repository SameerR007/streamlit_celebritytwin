import streamlit as st
from PIL import Image
import os,cv2
import numpy as np
from mtcnn import MTCNN
detector=MTCNN()
import pickle
from keras_vggface.utils import preprocess_input
st.title("Which bollywood celebrity are you?")
uploaded_image=st.file_uploader("Upload image")
if(uploaded_image!=None):
    display_image=Image.open(uploaded_image)
    st.image(display_image)
    if st.button("Predict"):
        st.text("Finding your twin. This may take a minute or two.")
        with open(os.path.join(uploaded_image.name),"wb") as f:
            f.write(uploaded_image.getbuffer())

        img=cv2.imread(os.path.join(uploaded_image.name))
        
        results = detector.detect_faces(img)
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image)
        face_array = face_array.astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        model=pickle.load(open("model.pkl",'rb'))
        result = model.predict(preprocessed_img).flatten()
        from sklearn.metrics.pairwise import cosine_similarity
        feature_list=pickle.load(open("embedding.pkl",'rb'))
        similarity = [] 
        for i in range(len(feature_list)):
            similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
        index=sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
        col1,col2 = st.columns(2)
        filenames = pickle.load(open('filenames.pkl','rb')) 

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            predicted_actor =" ".join(filenames[index].split('\\')[1].split('_'))
            st.header("Seems like "+ predicted_actor)
            st.image(filenames[index],width=300)
        st.text("Thank you for your time and patience.")