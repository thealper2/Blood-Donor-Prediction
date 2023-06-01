import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("rf.pkl", "rb"))

st.title("Blood Donor Prediction")

rec = st.number_input("Recency (months)")
freq = st.number_input("Frequency (times)")
mone = st.number_input("Monetary (c.c. blood")
time = st.number_input("Time (months)")

if st.button("Predict"):
	test = np.array([[rec, freq, mone, time]])
	res = model.predict(test)
	print(res)
	st.success("Predict: " + str(res[0]))
