from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering, OPTICS, KMeans


class Clustering2D:
    def __init__(self, clustering_type="AgglomerativeClustering",
                 n_clusters=5):
        self.clustering_type = clustering_type
        self.n_clusters = n_clusters
        if clustering_type == "AgglomerativeClustering":
            self.clustering = AgglomerativeClustering(n_clusters=n_clusters)
        elif clustering_type == "KMeans":
            self.clustering = KMeans(n_clusters=n_clusters)
#         elif clustering_type == "DBSCAN":
#             self.clustering = DBSCAN()
#         elif clustering_type == "SpectralClustering":
#             self.clustering = SpectralClustering(n_clusters=n_clusters)
#         elif clustering_type == "OPTICS":
#             self.clustering = OPTICS(n_clusters=n_clusters)
    
    def fit_predict(self, X):
        """
            X - train_coefs with small_polyos and big_polyos columns
        """
        
        clusters = self.clustering.fit_predict(X[['small_polyos', 'big_polyos']].values)
        clusters_map = {sat_id : clusters[i] for i, sat_id in enumerate(X.sat_id.unique())}
        return clusters_map