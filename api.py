from flask import Flask, jsonify, request as req
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load
from flask_cors import CORS
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
	return '<p>hola</p>'

@app.route('/https://marginal-signals.herokuapp.com/historical')
def historical():
    start_date = input('Introduce date: ')
    end_date = input('Introduce date: ')
    start_date = str(start_date)
    end_date = str(end_date)
    Dataset = pd.read_csv('penalty_signals.csv', encoding="UTF-8")
    if start_date and end_date in Dataset.PBF_datetime_utc.values:
        print("The date is in list")
              
        mask = (Dataset.PBF_datetime_utc >= start_date) & (Dataset.PBF_datetime_utc <= end_date)
        filtered_df = Dataset.loc[mask]
        response = filtered_df
        response = filtered_df.to_json()
	
    else:
        response = ['Requested datetime series is not available.']
        print('olala')

    return response 
