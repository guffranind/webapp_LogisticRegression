# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:11:11 2021

@author: user
"""
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
#st.write('Hello')

#message_text = st.text_input("Enter a message for spam evaluation")

import pickle

with open('ccfraud_lg.pickle','rb') as f:
    classifier=pickle.load(f)


#input_dict={'V4':V4,'V10':V10,'V12':V12,'V14':V14,'V16':V16}
#input_df=pd.DataFrame([input_dict])


def main():
    st.sidebar.subheader("Enter the values")
    V4=st.sidebar.number_input('V4')
    V10=st.sidebar.number_input('V10')
    V12=st.sidebar.number_input('V12')
    V14=st.sidebar.number_input('V14')
    V16=st.sidebar.number_input('V16')
    #input_dict={'V4':V4,'V10':V10,'V12':V12,'V14':V14,'V16':V16}
    input_dict={'V4':V4,'V10':V10,'V12':V12,'V14':V14,'V16':V16}
    #input_dict=[V4,V10,V12,V14,V16]
    input_df=pd.DataFrame([input_dict])
   
    
    
    st.title("Credit Fraud Analysis")
    if st.button('Predict'):
        survived=classifier.predict(input_df)
        st.subheader("Predict Class is")
        st.success(survived)
        
    
if __name__=='__main__':
    main()
