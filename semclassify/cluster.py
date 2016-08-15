# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from semclassify import helpers as h
from semclassify import dbtools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_best_clustering(df, n_clusters_range=range(2,10)):
  """ Clusters all rows in the DataFrame using all columns as features.

  Uses MiniBatchKMeans to establish number of clusters which maximize the
  average silhouette coefficient. Then, computes a final KMeans clustering
  with the optimized number of clusters.
  :param X: numpy 2-d array-like
  :param n_clusters_range: iterable specifying integer-valued numbers of
  clusters
  :return nd-array of cluster labels
  """

  # Compute clustering with MiniBatchKMeans
  scores = []
  batch_size = 100

  X = StandardScaler().fit_transform(df.fillna(df.mean()).as_matrix())
  for n_clusters in n_clusters_range:
    mbk = MiniBatchKMeans(init='random', n_clusters=n_clusters,
                          batch_size=batch_size,
                          n_init=3 * batch_size,
                          max_no_improvement=10,
                          verbose=False)
    with h.stopwatch('fitting minibatch kmeans'):
      mbk.fit(X)

    with h.stopwatch('computing silhouette average'):
      score = silhouette_score(X,
                               mbk.labels_,
                               sample_size=int(len(mbk.labels_) * 0.001))

      scores.append((score, n_clusters, mbk.cluster_centers_))

  for s, n, _ in scores:
    print('%d clusters => %f' % (n, s))

  _, best_n_clusters, best_centroids = max(scores)
  print('-- %d clusters maximize the mean silhouette coefficient --' %
        best_n_clusters)

  # Cluster all the FAF particles again using the centroids above
  km = KMeans(n_clusters=best_n_clusters,
              init=best_centroids,
              n_init=1,
              )

  with h.stopwatch('fitting entire dataset with kmeans'):
    km.fit(X)

  return km.labels_


from scipy.stats import f, chi2


def hotel2(X1, X2):
  """ Computes Hotelling t-squared statistic under two assumptions or variance.

  :param X1 pandas DataFrame with samples from first group
  :param X2 pandas DataFrame with samples from second group
  :return None
  """
  # TODO: Verify Hotelling results
  n1, k = X1.shape
  n2, k2 = X2.shape
  assert(k == k2)

  ybar1 = X1.mean().as_matrix()
  s1 = np.cov(X1, rowvar=False)
  ybar2 = X2.mean(axis=0).as_matrix()
  s2 = np.cov(X2, rowvar=False)

  alpha = 0.05
  diffs = (ybar1 - ybar2).reshape(1, k)

  # TODO: Incorporate a test for equal variances

  # If variances assumed equal, then pool
  if True:
    spool = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    t2 = diffs\
      .dot(np.linalg.inv(spool * (1.0 / n1 + 1.0 / n2)))\
      .dot(ybar1 - ybar2)\
      .item(0)
    eff = (n1 + n2 - k - 1) * t2 / (k * (n1 + n2 - 2))
    df1 = k
    df2 = n1 + n2 - k - 1
    p_value = f.sf(eff, df1, df2)
    print('If variances are assumed equal between classes')
    if p_value < alpha:
      print("\t=> Reject the null hypothesis that mean(X1) == mean(X2)")
    else:
      print("\t=> Accept null hypothesis that mean(X1) == mean(X2)")
    print(t2, p_value)

  # If variances not assumed equal, then use modified Hotelling
  if True:
    t2 = diffs\
      .dot(np.linalg.inv(s1 / n1 + s2 / n2))\
      .dot(ybar1 - ybar2)\
      .item(0)
    p_value = chi2.sf(t2, k)
    print('If variances are not assumed equal between classes')
    if p_value < alpha:
      print("\t=> Reject the null hypothesis that mean(X1) == mean(X2)")
    else:
      print("\t=> Accept null hypothesis that mean(X1) == mean(X2)")
    print(t2, p_value)


def get_cluster_labels():
  df = pd.read_hdf("../output/store.h5",
                   key="FAF",
                   where=["FAF==True"],
                   columns=["Al","Ca","Fe","K","Mg","Na","S","Si"])

  df.loc[:,'kmeans'] = get_best_clustering(df, n_clusters_range=range(2,10))
  df_full = pd.read_hdf("../output/store.h5",
                        key="FAF",
                        where=None).join(df.kmeans)
  print(df.index[-1], df_full.index[-1])
  print(df_full.head(10))


if __name__ == "__main__":
  get_cluster_labels()
  # hotel2(df.loc[df.kmeans == 0, columns], df.loc[df.kmeans == 1, columns])
