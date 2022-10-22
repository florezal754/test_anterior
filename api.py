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


@app.route("/clasificar-noticia/", methods=["GET"])
def clasificar_noticia():
	noticia = req.args.get('text')
	modelo = load('./modelo.joblib')
	vectorizer = modelo[1]
	clasificador = modelo[0]
	texto_introducido = [noticia]
	noticia_counts = vectorizer.transform(texto_introducido)
	predictions = clasificador.predict(noticia_counts)
	confianza = clasificador.predict_proba(noticia_counts)
	respuesta = jsonify({"Resultado": predictions.tolist(),"Confianza": confianza.tolist()})
	return respuesta


@app.route("/anadir-noticia/", methods=["POST"])
def anadir_noticia():
    noticia = req.form['text']
    etiqueta = req.form['label']
    with open('./prueba.csv', 'a', newline='\n') as f:
        fila = '"' + noticia + '"' + ',' + str(etiqueta) + '\n'
        f.write(fila)
        
    Dataset = pd.read_csv('./prueba.csv', encoding="UTF-8")
    datosY = Dataset['clase']
    datos = np.array(datosY)
    count = len(datos)
    unique, counts = np.unique(datos, return_counts=True)
    dic = dict(zip(unique, counts))
    plt.bar(['0','1'], [dic[0],dic[1]], color=['g','r'],tick_label=['0 - Noticias Verdaderas','1 - Noticias Falsas'])
    plt.text(-0.05,dic[0]+10,dic[0])
    plt.text(0.95,dic[1]+25,dic[1])
    plt.savefig('./datos.png', format='png')
    plt.close()
    respuesta = jsonify({"Noticia": noticia, "Etiqueta": etiqueta, "Respuesta": "Noticia añadida al dataset correctamente."})
    return respuesta
    
@app.route("/reentrenar-modelo/", methods=["GET"])
def reentrenar_modelo():
    Dataset = pd.read_csv('./prueba.csv', encoding="UTF-8")
    datosX = Dataset['texto']
    datosY = Dataset['clase']
   
    vector = CountVectorizer()
    counts = vector.fit_transform(datosX.values)
    clasificador = MultinomialNB()
    targets = datosY.values
    clasificador.fit(counts, targets)
    dump((clasificador,vector), './modelo.joblib')
    
    respuesta = jsonify({"Respuesta": "Modelo reentrenado correctamente."})
    return respuesta
