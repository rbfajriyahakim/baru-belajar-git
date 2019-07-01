
# coba nambah komentar disini 
from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
import matplotlib.pyplot as plt
import io
from flask import make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import send_file
from sklearn.externals import joblib

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '12345678'
app.config['MYSQL_DB'] = 'toko'
mysql = MySQL(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        details = request.form
        Pendapatan = details['dapat']
        Biayaiklan = details['iklan']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO keuangan (pendapatan, biayaiklan) VALUES (%s, %s)", (Pendapatan, Biayaiklan))
        mysql.connection.commit()
        cur.close()
        return 'success'
    return render_template('index.html')

@app.route('/untung')
def untung():
 cur = mysql.connection.cursor()
 cur.execute('''SELECT pendapatan, biayaiklan FROM keuangan''')
 rv = cur.fetchall()
 return render_template("tabelpendapatan.html",value=rv)

@app.route('/buatfile')
def buatfile():
 cur = mysql.connection.cursor()
 cur.execute('''SELECT biayaiklan, pendapatan FROM keuangan''')
 rv = cur.fetchall()
 c = csv.writer(open('datatoko2.csv','w'))
 for x in rv:
  c.writerow(x)
 return 'sukses'

@app.route('/model')
def modelml():
 dataset=pd.read_csv('datatoko2.csv')
 X=dataset.iloc[:,:-1].values
 y=dataset.iloc[:,1].values
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)
 regressor = LinearRegression()
 regressor.fit(X_train, y_train)
 y_pred = regressor.predict(X_test)
 pickle.dump(regressor, open('model2.pkl','wb'))
 model = pickle.load(open('model2.pkl','rb'))
 print(X)
 result = float(model.predict([[75]]))
 return render_template("hasil.html", result=result)

dataset=pd.read_csv('datatoko2.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
modelgabung = LinearRegression().fit(X,y)
filename="model.sav"
joblib.dump(modelgabung, filename)
loaded_model=joblib.load(filename)

@app.route('/api')
def student():
	return render_template("home.html")

def ValuePredictor(to_predict_list):
		to_predict = np.array(to_predict_list).reshape(-1,1)
		loaded_model = joblib.load('model.sav')
		result = loaded_model.predict(to_predict)
		return result[0]

@app.route('/api',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
    to_predict_list = request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    result = float(ValuePredictor(to_predict_list))
    return render_template("home.html",result = result)


if __name__ == '__main__':
    app.run()