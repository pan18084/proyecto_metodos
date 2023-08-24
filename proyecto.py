#por si no se tiene el data set se utiliza "pip install scikit-learn" en el cmd y se instala

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

def g(x):
    return x - (x ** 3 - x ** 2 - 1) / (3 * x ** 2 - 2 * x)

def FixedPoint(x0, es, imax):
    xr = x0
    iter = 0
    ea = 0
    
    while True:
        xrold = xr
        xr = g(xrold)
        iter = iter + 1
        
        if xr != 0:
            ea = abs((xr - xrold) / xr) * 100
        
        if ea < es or iter >= imax:
            break
    
    return xr, iter, ea

# Cargar la base de datos Iris
iris = load_iris()
X = iris.data

# Parámetros para k-means
num_clusters = 3 #ya que el data set cuenta con 3 etiquetas se busca utlizar ese numero para hacerlo lo más acertado posible
es = 0.0001
imax = 100

# Usar el método FixedPoint para obtener el centroide inicial
x0 = 1.0
centroid_initialization, iterations, error = FixedPoint(x0, es, imax) #por medio del punto fijo se obtine un valor inicial para el centroide del clustering
print(f"Método de punto fijo: Solución aproximada: {centroid_initialization:.6f}")
print(f"Se realizaron {iterations} iteraciones con un error del {error:.6f}%")

# Inicializar todos los centroides con el valor fijo
initial_centroids = np.full((num_clusters, X.shape[1]), centroid_initialization)

# Utilizar KMeans de scikit-learn para realizar el clustering
kmeans = KMeans(n_clusters=num_clusters, init=initial_centroids, n_init=1, max_iter=100)
kmeans.fit(X)

final_cluster_labels = kmeans.labels_

print("Etiquetas de clúster finales:")
print(final_cluster_labels)


# Prueba con método de Newton Raphson
tol = 0.0001

def newton_raphson(x0, f, df, tol, max_iter):
    x1 = x0
    iter1 = 0
    ea1 = 0
    
    while True:
        xrold = x1
        x1 = x1 - f(x1) / df(x1)
        iter1 = iter1 + 1
        
        if x1 != 0:
            ea1 = abs((x1 - xrold) / x1) * 100
        
        if ea1 < tol or iter1 >= max_iter:
            break
    
    return x1, iter1, ea1

# Definir la función f(x) y su derivada df(x) para el método de Newton-Raphson
def f_newton(x):
    return x ** 3 - x ** 2 - 1

def df_newton(x):
    return 3 * x ** 2 - 2 * x

# Usar el método Newton-Raphson para obtener el centroide inicial
centroid_initialization1, iterations1, error1 = newton_raphson(x0, f_newton, df_newton, tol, imax)
print(f"Método de Newton-Raphson: Solución aproximada: {centroid_initialization1:.6f}")
print(f"Se realizaron {iterations1} iteraciones con un error del {error1:.6f}%")

# Inicializar todos los centroides con el valor obtenido de Newton-Raphson
initial_centroids1 = np.full((num_clusters, X.shape[1]), centroid_initialization1)

# Utilizar KMeans para realizar el clustering
kmeans_1 = KMeans(n_clusters=num_clusters, init=initial_centroids1, n_init=1, max_iter=100)
kmeans_1.fit(X)

final_cluster_labels1 = kmeans_1.labels_

print("Etiquetas de clúster finales:")
print(final_cluster_labels1)
