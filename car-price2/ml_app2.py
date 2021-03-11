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
import joblib 

# 여기서 모델 불러오자 

def run_ml_app() :
    st.subheader('Machine Learning')

    #1. 유저한테 입력을 받는다
    # 성별
    # 남자 버튼 누르면 남자데이터 , 여자 버튼 누르면 여자데이터가 나오도록
    gender =  st.radio('성별을 선택하세요', ['남자', '여자'])
    if gender == '남자' : 
        gender = 1

    # 나이                             # 최소 몇~~ 부터 최대 몇 ~ 까지 범위지정
    age =  st.number_input('나이 입력', min_value=0, max_value= 100)

    # salary                              
    salary =  st.number_input('연봉 입력', min_value= 0)

    # debt
    debt = st.number_input('카드빚 입력', min_value=0)

    # 자산
    worth = st.number_input('자산입력', min_value=0)

    # 2. 예측한다
    # 2-1. 모델 불러오기
    model =  tensorflow.keras.models.load_model('data2/car_ai.h5')

    # 2-2. 넘파이 어레이 만든다 
    new_data = np.array( [ gender, age, salary, debt, worth ] )

    # 2-3. 피처 스케일링 하자
    new_data = new_data.reshape(1,-1)
    sc_X = joblib.load('data2/sc_X.pk1')
    new_data = sc_X.transform(new_data)

    # 2-4. 예측한다
    y_pred =  model.predict(new_data)

    # 예측 결과는 스케일링 된 결과이므로 다시 돌려야 한다
    # st.write( y_pred[0][0] )
    sc_y =  joblib.load('data2/sc_y.pk1')

    y_pred_orginal = sc_y.inverse_transform(y_pred)

    #st.write(y_pred_orginal)


    #3. 결과를 화면에 보여준다
    button =  st.button('결과보기')
    if button :
        st.write('예측 결과입니다 {:,.2f} 달러의 차를 살 수 있습니다'.format(y_pred_orginal[0,0]))
                              # 천단위마다 콤마 찍어서 나타내라
                              # 소수점 밑으로 2자리까지만 나타내라


    # 처음 설정했던것 
    # new_data = np.array( [ 0, 38, 90000, 2000, 50000 ] )

    # new_data = new_data.reshape(1,-1)

    # # 코랩이랑 vscode 는 서로 다른 환경이므로 변수를 다시 지정해 줘야한다 
    # sc_X = joblib.load('data2/sc_X.pk1')

    # new_data = sc_X.transform(new_data)  # 이게 어떤 스케일러인지 모름 >> 오류남

    # y_pred =  model.predict(new_data)

    # st.write( y_pred[0][0] )
    # # pip install scikit-learn==0.23.2 다운 받아야 실행 가능 

    # # 코랩이랑 vscode 는 서로 다른 환경이므로 변수를 다시 지정해 줘야한다 
    # sc_y =  joblib.load('data2/sc_y.pk1')

    # y_pred_orginal = sc_y.inverse_transform(y_pred)

    # st.write(y_pred_orginal)


    # # 다른데서 예측하기 위해 가져가서 쓸려면  y_pred 값만 본다

# 잘 만든것인지는 코랩에서 가져온 파일결과와 같은지 확인하면 된다 
