import os, psycopg2, json, io, base64
from flask import Flask, request, request, jsonify
from flask_sqlalchemy import SQLAlchemy
# maching learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
app = Flask(__name__)

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
con = psycopg2.connect(database="delati", user="modulo4", password="modulo4", host="128.199.1.222", port="5432")
cursor = con.cursor()


def load_data():
    cursor.execute("select distinct o.htitulo_cat, o.htitulo from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null order by 1,2 limit 500;")
    result = cursor.fetchall()
    return result

@app.route("/")
def init():
    data = load_data()
    json_result = json.dumps(data)
    return json_result

@app.route("/kmeans", methods = ['GET', 'POST', 'DELETE'])
def kmeans():
    if request.method == 'POST':
        total_data = load_data()
        data = []
        for selected_tuple in total_data:
            data.append(' '.join(selected_tuple))
        # print(data)
        # CATCH DATA FROM BODY
        body = request.get_json()
        n_clusters=body["n_clusters"]
        init= body['init']
        max_iter= body['max_iter']
        # end requests
        vectorizer = TfidfVectorizer(min_df = 0.01, ngram_range = (2,2))
        vec = vectorizer.fit(data)   # train vec using list1
        vectorized = vec.transform(data)   # transform list1 using vec
        # print(vectorized)
        kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=1, random_state=0)
        pred_y = kmeans.fit_predict(vectorized)

        clusters = {}
        n = 0
        for item in pred_y:
            if item in clusters:
                clusters[item].append(data[n])
            else:
                clusters[item] = [data[n]]
            n +=1
        result = {}
        for item in clusters:
            temporal_cluster = 'Cluster {}'.format(item)
            result[temporal_cluster] = len(clusters[item])
            # print ("Cluster {}, Length: {}".format(item, len(clusters[item])))
            # for i in clusters[item]:
            #     print(i)
        centroids = kmeans.cluster_centers_
        # print(centroids)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red')
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        # plt.savefig('/home/diego/Downloads/works.png')
        result["graphic"] = my_base64_jpgData.decode()
        return jsonify(result)


if __name__ == '__main__':
    app.run()