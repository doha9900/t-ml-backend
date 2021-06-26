# t-ml-backend
Link del backend en producción heroku: https://t-ml-backend.herokuapp.com/
Endpoints:
get Data :  METHOD: GET https://t-ml-backend.herokuapp.com/
kmeans : METHOD POST https://t-ml-backend.herokuapp.com/kmeans
          request: {
    "n_clusters": 5,
    "init": "random",
    "max_iter": 500
      }
