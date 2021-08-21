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
    # cursor.execute("select distinct o.htitulo_cat, o.htitulo from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null order by 1,2 limit 500;")
    # cursor.execute("select o.htitulo_cat as categoria,o.htitulo as perfil, w.pagina_web,o.empresa,o.lugar,o.salario,date_part('year',o.fecha_publicacion) as periodo, f_dimPerfilOferta(o.id_oferta,7) as funciones, f_dimPerfilOferta(o.id_oferta,1) as conocimiento, f_dimPerfilOferta(o.id_oferta,3) as habilidades, f_dimPerfilOferta(o.id_oferta,2) as competencias, f_dimPerfilOferta(o.id_oferta,17) as certificaciones, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio, f_dimPerfilOferta(o.id_oferta,11) as formacion from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null;")
    cursor.execute("select o.htitulo_cat as categoria,o.htitulo as perfil, w.pagina_web,o.empresa,o.lugar,o.salario,date_part('year',o.fecha_publicacion) as periodo, f_dimPerfilOferta(o.id_oferta,7) as funciones, f_dimPerfilOferta(o.id_oferta,1) as conocimiento, f_dimPerfilOferta(o.id_oferta,3) as habilidades, f_dimPerfilOferta(o.id_oferta,2) as competencias, f_dimPerfilOferta(o.id_oferta,17) as certificaciones, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio, f_dimPerfilOferta(o.id_oferta,11) as formacion from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null limit 500;")
    result = cursor.fetchall()
    return result

@app.route("/algorithms", methods = ['GET', 'POST'])
def algorithms():
    name_algorithms = ['kmeans', 'spectral']
    return jsonify({
        'algorithms':name_algorithms
        })

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
        # columns_name=body["columns"]
        n_clusters=body["n_clusters"]
        init= body['init']
        n_init= body['n_init']
        random_state= body['random_state']
        max_iter= body['max_iter']        
        axis_x= int(body['axis_x'])
        axis_y= int(body['axis_y'])
        result = {}
        # end requests+
        field_names = [i[0] for i in cursor.description]
        dataframe = pd.DataFrame(total_data, columns=field_names)#.values #.tolist()
        # print(dataframe)
        label_encoder = preprocessing.LabelEncoder()
        transformed_data = dataframe.apply(label_encoder.fit_transform)
        # elbow method 
        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
            kmeanModel.fit(transformed_data)
            distortions.append(kmeanModel.inertia_)
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distorción')
        plt.title('El método del codo muestra el k clusters óptimo.')
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        # plt.savefig("graphic2.jpg")
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        result["elbow_method"] = my_base64_jpgData.decode()
        plt.clf() #clear current image plt
        
        kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
        # KMEANS
        pred_y = kmeans.fit_predict(transformed_data)
        print(pred_y)
        elements = kmeans.labels_  # values from kmeans.fit_predict(transformed_data)
        centroids = kmeans.cluster_centers_
        print(centroids)
        centroids_values = [] # for search in dataframe
        centroids_all_data = [] # centroids details
        for cd in centroids:
            # airports = [(10,10),(20,20),(30,30),(40,40)]
            airports = transformed_data
            tree = spatial.KDTree(airports)
            found = tree.query(cd)
            centroids_values.append(found[1])
            centroids_all_data.append(found)
        # print(centroids_values)
        # plt.scatter(transformed_data[columns_name[0]].values,transformed_data[columns_name[1]].values, s=10, c='green')
        colors = "gcmykwbr"
        for cluster in range(n_clusters):
            # print(cluster)
            plt.scatter(transformed_data.iloc[pred_y==cluster, axis_x], transformed_data.iloc[pred_y==cluster, axis_y], s=10, c=colors[cluster])
            scatter = plt.scatter(centroids[cluster, axis_x], centroids[cluster, axis_y], s=120, c=colors[cluster],alpha=0.3, label=f"Cluster {cluster}")
            plt.legend(title='Clusters', loc='upper left', fontsize='xx-small')
            # legend1 = plt.legend(*scatter.legend_elements(),
            #         loc="lower left", title="Clusters")
        plt.xlabel(field_names[axis_x])
        plt.ylabel(field_names[axis_y])
        print("==========")
        dataframe["cluster"] = elements
        dataframe.sort_values(['cluster'], ascending=False)
        # print(centroids_values)
        # print(dataframe.sort_values(['cluster'], ascending=True))
        # result[""]
        field_names.append("cluster")
        centroids_details = []
        x = 0
        for _centroid in centroids_all_data:
            obj = {}
            obj["point"] = (centroids.tolist())[x]
            obj["distance"] = float(_centroid[0])
            obj["position"] = int(_centroid[1])
            obj["title_cluster"]= json.loads((dataframe.iloc[centroids_values[x]]).to_json(orient='values'))

            centroids_details.append(obj)
            x+=1
        result["centroids"] = centroids_details
        result["inertia"] = kmeans.inertia_
        result["n_iter"] = kmeans.n_iter_
        result["total_instances"] = len(dataframe.index)
        result["columns"] = field_names
        result["data"] = json.loads(dataframe.sort_values(['cluster'], ascending=True).to_json(orient='table')) #orient='table'
        clusters = []
        for item in range(n_clusters):
            temporal_cluster = 'Cluster {}'.format(item)
            length_actual_cluster = int(dataframe["cluster"].value_counts()[item])
            decimal_frequency_actual_cluster = float(dataframe["cluster"].value_counts(normalize=True)[item])
            obj = {
                "cluster": temporal_cluster,
                "length": length_actual_cluster,
                "percentage": (round(decimal_frequency_actual_cluster*100, 2)),
                "title_cluster": json.loads((dataframe.iloc[centroids_values[item]]).to_json(orient='values'))
                # "title_cluster": json.loads(pd.Series(dataframe.iloc[centroids_values[item]]).to_json())
            }
            clusters.append(obj)
        result["clusters"] = clusters
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        # plt.savefig("graphic2.jpg")
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        result["graphic"] = my_base64_jpgData.decode()
        plt.clf() #clear current image plt
        response = jsonify(result)
        return response

if __name__ == '__main__':
    app.run()







