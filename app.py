# IMPORTS
import os, psycopg2, json, io, base64
import numpy as np
import math
import pandas as pd
# LIB
from scipy import spatial
from scipy.sparse import data
from sklearn import preprocessing
# FLASK 
from flask import Flask, request, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
# maching learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# POSTGRES = {
#     'user': 'modulo4',
#     'pw': 'modulo4',
#     'db': 'delati',
#     'host': '128.199.1.222',
#     'port': '5432',
# }
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:\
# %(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES



def load_data():
    con = psycopg2.connect(database="delati", user="modulo4", password="modulo4", host="128.199.1.222", port="5432")
    cursor = con.cursor()
    cursor.execute("select distinct o.htitulo_cat, o.htitulo from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null order by 1,2 limit 500;")
    result = cursor.fetchall()
    return result

@app.route("/kmeans", methods = ['GET', 'POST', 'DELETE'])
def kmeans():
    con = psycopg2.connect(database="delati", user="modulo4", password="modulo4", host="128.199.1.222", port="5432")
    cursor = con.cursor()
    if request.method == 'GET':
        return jsonify(load_data())
    if request.method == 'POST':
        body = request.get_json()
        query = cursor.execute(body["query"])
        total_data = cursor.fetchall()
        # total_data = load_data()
        # CATCH DATA FROM BODY
        columns_name=body["columns"]
        n_clusters=body["n_clusters"]
        init= body['init']
        max_iter= body['max_iter']
        # end requests
        dataframe = pd.DataFrame(total_data, columns=columns_name)#.values #.tolist()
        # print(dataframe)
        label_encoder = preprocessing.LabelEncoder()
        transformed_data = dataframe.apply(label_encoder.fit_transform)
        # print(transformed_data)
        kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=1, random_state=0)
        pred_y = kmeans.fit_predict(transformed_data)
        print(pred_y)
        elements = kmeans.labels_  # values from kmeans.fit_predict(transformed_data)
        centroids = kmeans.cluster_centers_
        print(centroids)
        centroids_values = []
        for cd in centroids:
            # airports = [(10,10),(20,20),(30,30),(40,40)]
            airports = transformed_data
            tree = spatial.KDTree(airports)
            found = tree.query(cd)
            print(found)
            centroids_values.append(found[1])
        print(centroids_values)
        # plt.scatter(transformed_data[columns_name[0]].values,transformed_data[columns_name[1]].values, s=10, c='green')
        colors = "bgcmykw"
        for cluster in range(n_clusters):
            plt.scatter(transformed_data.iloc[pred_y==cluster, 0], transformed_data.iloc[pred_y==cluster, 1], s=10, c=colors[cluster], label =f'Cluster {cluster}')
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red')
        print("==========")
        dataframe["cluster"] = elements
        dataframe.sort_values(['cluster'], ascending=False)
        # print(centroids_values)
        # print(dataframe.sort_values(['cluster'], ascending=True))
        result = {}
        # result[""]
        columns_name.append("clusters")
        result["columns"] = columns_name
        result["data"] = json.loads(dataframe.sort_values(['cluster'], ascending=True).to_json(orient='table')) #orient='table'
        for item in range(n_clusters):
            temporal_cluster = 'Cluster {}'.format(item)
            length_actual_cluster = int(dataframe["cluster"].value_counts()[item])
            decimal_frequency_actual_cluster = float(dataframe["cluster"].value_counts(normalize=True)[item])
            result[temporal_cluster] = {
                "length": length_actual_cluster,
                "percentage":'{} %'.format(round(decimal_frequency_actual_cluster*100, 2)),
                "title_cluster": json.loads(pd.Series(dataframe.iloc[centroids_values[item]]).to_json(orient='values'))
                # "title_cluster": json.loads(pd.Series(dataframe.iloc[centroids_values[item]]).to_json())
            }

        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        # plt.savefig("graphic2.jpg")
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        result["centroids"] = centroids.tolist()
        result["graphic"] = my_base64_jpgData.decode()
        response = jsonify(result)
        return response

if __name__ == '__main__':
    app.run()







