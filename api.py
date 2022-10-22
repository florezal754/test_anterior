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
