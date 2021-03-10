import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pickle

# 여기서 모델 불러오자 

def run_ml_app() :
    st.subheader('Machine Learning')

    model =  tensorflow.keras.models.load_model('data/my_model.h5')

    # 다른데서 예측하기 위해 가져가서 쓸려면  y_pred 값만 본다

    # 잘 만든것인지는 코랩에서 가져온 파일결과와 같은지 확인하면 된다 

    new_data = np.array( [ 0, 38, 90000, 2000, 50000 ] )

    new_data = new_data.reshape(1,-1)

    new_data = sc.transform(new_data)

    sc_X = joblib.load('data/sc_X.pk1')

    new_data = sc_X.transform(new_data)  # 이게 어떤 스케일러인지 모름 >> 오류남

    y_pred =  model.predict(new_data)

    st.write( y_pred[0][0] )
    # pip install scikit-learn==0.23.2 다운 받아야 실행 가능 

    sc_y =  joblib.load('data/sc_y.pk1')

    y_pred_orginal = sc_y.inverse_transform(y_pred)

    st.write(y_pred_orginal)



