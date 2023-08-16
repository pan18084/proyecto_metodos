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
num_clusters = 3
es = 0.0001
imax = 100

# Usar el método FixedPoint para obtener el centroide inicial
x0 = 1.0
centroid_initialization, iterations, error = FixedPoint(x0, es, imax)
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
