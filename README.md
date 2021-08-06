# t-ml-backend
#Link del backend en producción heroku: https://t-ml-backend.herokuapp.com/
#Endpoints:
> get ALgorithms :  METHOD: GET https://t-ml-backend.herokuapp.com/algorithms
> kmeans : METHOD POST https://t-ml-backend.herokuapp.com/kmeans
```
          {
    "query":"select distinct o.htitulo_cat, o.htitulo from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null order       by 1,2 limit 500;",
    "columns": ["titulo_cat", "full_descripcion"],
    "n_clusters": 5,
    "init": "random",
    "max_iter": 500
      }
```
