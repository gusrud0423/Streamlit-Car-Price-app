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

def run_eda_app():
    st.subheader('EDA 화면입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding= 'ISO-8859-1') 

    radio_menu = ['데이터 프레임', '통계치', '상관관계 분석']
    selectbox_radio =  st.radio('선택하세요', radio_menu)

    if selectbox_radio == '데이터 프레임' :
        st.dataframe(car_df)
    elif selectbox_radio == '통계치' :
        st.dataframe(car_df.describe() )
    elif selectbox_radio == '상관관계 분석' :
        st.dataframe( car_df.corr() )


    