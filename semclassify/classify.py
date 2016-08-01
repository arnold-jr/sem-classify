import pandas as pd
import numpy as np
from semclassify.helpers import stopwatch
from semclassify.transformers import ColumnSelectorTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import dill as pickle



class SuperModel():
  def __init__(self, store_path, feat_cols, label_col):
    self.store_path = store_path
    self.feat_cols = feat_cols
    try:
      self.label_col = label_col
      if len(label_col) != 1:
        raise ValueError("Incorrect parameter for SuperModel.label_col")
    except:
      raise

    self.modelChoices = dict(
      logit=dict(model=LogisticRegression(
        class_weight='balanced',
        penalty='l1',
      ),
        param_grid={'C': np.logspace(-2, 2, 3)}
      ),
      lda=dict(model=LinearDiscriminantAnalysis(
        n_components=None,
        priors=None,
        shrinkage=None,
        solver='svd',
        store_covariance=False,
        tol=1e-4,
      ),
        param_grid={'tol': np.logspace(-5, -3, 3)}
      ),
      knn=dict(model=KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
      ),
        param_grid={'n_neighbors': list(range(1,8,2))}
      ),
      dtc=dict(model=DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        class_weight=None,
        presort=False,
      ),
        param_grid={'min_samples_split': list(range(2, 5))}
      ),
      rfc=dict(model=RandomForestClassifier(
        n_estimators=10,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
      ),
        param_grid={"min_samples_split": list(range(2, 5))}
      ),
      etc=dict(model=ExtraTreesClassifier(
        n_estimators=10,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        bootstrap=False,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
      ),
        param_grid={"class_weight": ['balanced',None,'balanced_subsample']}
      ),
      gnb=dict(model=GaussianNB(),
               param_grid=dict()
               ),
    )

  def train_model(self, chunk, model_name, model_choice, all_classes=None):

    model_pipe = Pipeline([
      ('features', FeatureUnion(
        [('numeric', ColumnSelectorTransformer(self.feat_cols)),]
      )),
      ('scaler', StandardScaler()),
      ('model', model_choice['model']),
    ])

    y = chunk.loc[:, self.label_col].as_matrix().ravel()

    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.3)
    clf_CV = GridSearchCV(model_pipe,
                          param_grid={'model__'+k:v
                                      for k,v in
                                      model_choice['param_grid'].items()},
                          cv=cv,
                          n_jobs=1)
    if True:
      clf_CV.fit(chunk, y)
    else:
      clf_CV.partial_fit(chunk, y, all_classes)

    print(model_choice['model'])
    print('Model: %s\t Score: %f\t' % (model_name, clf_CV.best_score_))
    print('Parameters: ', clf_CV.best_params_)

    '''
    with open(self.get_pickled_name(model_name),
              'wb') as pckl_output:
      pickle.dump(clf_CV.best_estimator_, pckl_output)
    '''


  def train_all_models(self, model_names=('dtc')):
    """ Trains all the classifiers specified by their model name.

    :param model_names: list of model code names as strings
    :return None
    """
    where = 'site="soi_001"'
    df_labels = pd.read_hdf(self.store_path,
                            'df',
                            where=where,
                            columns=self.label_col
                            )
    all_classes = np.unique(df_labels)
    n_rows = len(df_labels.index)
    chunksize = n_rows

    for chunk_iter in pd.read_hdf(self.store_path,
                             'df',
                             chunksize=chunksize,
                             where=where,
                             columns=self.feat_cols + self.label_col):
      for modelName, modelValues in self.modelChoices.items():
        if modelName not in model_names:
          continue
        with stopwatch('training model %s' % modelName):
          chunk = chunk_iter
          self.train_model(chunk, modelName, modelValues, all_classes)
          print(chunk)


  def get_pickled_name(self, model_name):
    return self.fpath.replace(".csv","_" + model_name + ".dpkl")


  def get_trained_model(self, model_name):
    with open(self.get_pickled_name(model_name), 'rb') \
        as pckl_input:
      return pickle.load(pckl_input)


  def test_models(self, model_names, test_fpath):
    df_out = pd.DataFrame()
    for model_name in model_names:
      df = self.create_data_frame(test_fpath, train=False)
      clf = self.get_trained_model(model_name)
      df_out[model_name] = clf.predict(df)

    print(df_out)
    df_out.to_csv(test_fpath.replace(".csv","_results.csv"))

if __name__ == "__main__":
  sm = SuperModel(
    "/Users/joshuaarnold/Documents/MyApps/sem-classify/output/store.h5",
    ['BSE'],
    ['BFS'])
  sm.train_all_models()
