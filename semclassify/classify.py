import pandas as pd
import numpy as np
from semclassify.helpers import stopwatch
from semclassify.transformers import ColumnSelectorTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (SGDClassifier,
                                  Perceptron,
                                  LogisticRegression,
                                  PassiveAggressiveClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import (StratifiedShuffleSplit, cross_val_score,
                                      train_test_split)
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
      gnb=dict(model=GaussianNB()),
      sgd=dict(model=SGDClassifier()),
      asgd=dict(model=SGDClassifier(average=True)),
      perceptron=dict(model=Perceptron()),
      pag1=dict(model=PassiveAggressiveClassifier(loss='hinge',
                                                  C=1.0)
                ),
      pag2=dict(model=PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0)
                ),
      logit=dict(model=SGDClassifier(loss='log'))
    )

  def train_model(self, chunk, model_name, model_values, all_classes=None):

    feature_steps = [
      ('features', FeatureUnion([
        ('numeric', ColumnSelectorTransformer(self.feat_cols)),
      ])
      ),
      ('scaler', StandardScaler()),
    ]

    feature_pipe = Pipeline(feature_steps)

    model_pipe = Pipeline(feature_steps +
                               [('model', model_values['model'])]
                          )

    y = chunk.loc[:, self.label_col].as_matrix().ravel()

    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.3)

    if False:
      param_grid = model_values.get('param_grid',None)
      if param_grid is not None:
        clf_CV = GridSearchCV(model_pipe,
                              param_grid={'model__'+k:v
                                          for k,v in
                                          param_grid.items()},
                              cv=cv,
                              n_jobs=1)
        clf_CV.fit(chunk, y)
        print(model_values['model'])
        print('Model: %s\t Score: %f\t' % (model_name, clf_CV.best_score_))
        print('Parameters: ', clf_CV.best_params_)
      else:
        scores = cross_val_score(model_pipe, chunk, y, cv=cv, n_jobs=-1)
        print('Model: %s\t Score: %f\t' % (model_name, scores.mean()))
    else:
      X_tform = feature_pipe.fit_transform(chunk)
      X_train, X_test, y_train, y_test = train_test_split(X_tform, y,
                                                          test_size=0.3,
                                                          random_state=42)
      clf = model_values['model']
      clf.partial_fit(X_train, y_train, classes=all_classes)
      scores = clf.score(X_test, y_test)
      print('Model: %s\t Score: %f\t' % (model_name, scores.mean()))
      return clf


  def train_all_models(self, model_names=('gnb')):
    """ Trains all the classifiers specified by their model name.

    :param model_names: list of model code names as strings
    :return None
    """
    # where = 'site="soi_001"'
    where = None
    df_labels = pd.read_hdf(self.store_path,
                            'df',
                            where=where,
                            columns=self.label_col
                            )
    all_classes = np.unique(df_labels)
    n_rows = len(df_labels.index)
    chunksize = n_rows // 10

    for chunk in pd.read_hdf(self.store_path,
                             'df',
                             chunksize=chunksize,
                             where=where,
                             columns=self.feat_cols + self.label_col):
      for modelName, modelValues in self.modelChoices.items():
        if modelName not in model_names:
          continue
        with stopwatch('training model %s' % modelName):
          modelValues.update(
            dict(model=self.train_model(chunk,
                                        modelName,
                                        modelValues,
                                        all_classes)
                 )
          )


  def get_pickled_name(self, model_name):
    """ Returns a transformation of a name for a pickled model.

    :param model_name: unique id of a trained model
    :return path to the model
    """
    return self.store_path.replace(".h5","_" + model_name + ".dpkl")


  def pickle_models(self, model_names):
    """ Pickles selected models using the dill package.

    :return None
    """
    for model_name in model_names:
      with open(self.get_pickled_name(model_name),
                'wb') as pckl_output:
        pickle.dump(self.modelChoices[model_name]['model'], pckl_output)



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
    ['Al','Si','Ca','S','BSE'],
    ['BFS'])
  sm.train_all_models()
