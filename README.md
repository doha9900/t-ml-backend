# t-ml-backend
#Link del backend en producción heroku: https://js17-ml-backend.herokuapp.com/
_______
#Link del repositorio frontend: https://github.com/Julisa2020/js17-ml-frontend
_______
#Endpoints:
_____________
> get ALgorithms :  METHOD: GET https://js17-ml-backend.herokuapp.com/algorithms
_____________
> kmeans : METHOD POST https://js17-ml-backend.herokuapp.com/kmeans
_____________
>QUERY: 
```
          {
    "query":"select distinct o.htitulo_cat, o.htitulo from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null order       by 1,2 limit 500;",
    "columns": ["titulo_cat", "full_descripcion"],
    "n_clusters": 5,
    "init": "random",
    "max_iter": 500
      }
```
