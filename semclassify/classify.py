import pandas as pd
import numpy as np
from scipy import stats
from semclassify.helpers import stopwatch
from semclassify.transformers import ColumnSelectorTransformer
from semclassify.plots import plot_confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import (SGDClassifier,
                                  Perceptron,
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
  def __init__(self, store_path, tables=None, feat_cols=None, label_cols=None):
    """ Constructs an instance of SuperModel which associates a data store,
    data columns of interest, and a set of models to describe the data.

    :param store_path: string specifying path to HDF5 store
    :param tables: list of strings specifying table names in the HDF store
    :param feat_cols: list of feature columns as strings
    :param label_cols: list of label columns as strings
    """
    self.store_path = store_path
    self.tables = tables
    self.feat_cols = feat_cols
    try:
      self.label_cols = label_cols
      if len(label_cols) < 1:
        raise ValueError("Incorrect parameter for SuperModel.label_cols")
    except:
      raise

    self.label_encoder = LabelEncoder()
    self.label_encoder.fit(self.label_cols)
    print(self.label_encoder.classes_)
    print(self.label_encoder.transform(self.label_cols))

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
      rfc=dict(model=RandomForestClassifier(),
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





  def train_model(self, chunk, model_name, model_instance):
    """ Trains a single model with specified data, name, and model instance.

    :param chunk: pandas DataFrame containing data
    :param model_name: identifier for the model type
    :param model_instance: specification of the model instance and its
    parameters.
    :return trained model
    """
    feature_steps = [
        ('numeric', ColumnSelectorTransformer(self.feat_cols)),
        ('scaler', StandardScaler()),
    ]

    feature_pipe = Pipeline(feature_steps)

    model_pipe = Pipeline(feature_steps +
                          [('model', model_instance['model'])]
                          )


    # Here, combine labels into a single series
    def one_hot_decode(row):
      try:
        return row.index[list(row).index(True)]
      except:
        return "None"

    chunk.loc[:, 'y'] = chunk.loc[:, self.label_cols] \
      .apply(one_hot_decode, axis=1)

    chunk = chunk.loc[chunk.y != "None", :]
    y = self.label_encoder.transform(chunk.y.as_matrix().ravel())

    if True:
      cv = StratifiedShuffleSplit(y, n_iter=10, train_size=0.7, random_state=42)
      param_grid = model_instance.get('param_grid',None)
      if False and param_grid is not None:
        clf_CV = GridSearchCV(model_pipe,
                              param_grid={'model__'+k:v
                                          for k,v in
                                          param_grid.items()},
                              cv=cv,
                              n_jobs=1)
        clf_CV.fit(chunk, y)
        print(model_instance['model'])
        print('Model: %s\t Score: %f\t' % (model_name, clf_CV.best_score_))
        print('Parameters: ', clf_CV.best_params_)
      else:
        if True:
          scores = cross_val_score(model_pipe, chunk, y, cv=cv, n_jobs=-1)
          print('Model: %s\t Score: %f\t' % (model_name, scores.mean()))

        model_pipe.fit(chunk, y)
        freq = stats.itemfreq(model_pipe.predict(chunk))
        for f in freq:
          print(f[0], f[1])

        plot_confusion_matrix(y,
                              model_pipe.predict(chunk),
                              label_encoder=self.label_encoder)
        return model_pipe
    else:
      clf = model_instance['model']

      if len(y) < 1:
        return clf

      X_tform = feature_pipe.fit_transform(chunk)

      X_train, X_test, y_train, y_test = train_test_split(X_tform, y,
                                                          train_size=0.9,
                                                          random_state=42)
      clf.partial_fit(X_train, y_train,
                      classes=list(range(len(self.label_cols))))
      scores = clf.score(X_test, y_test)
      print('Model: %s\t Score: %f\t' % (model_name, scores.mean()))
      return clf


  def train_all_models(self, model_names=('gnb'), where=None):
    """ Trains all the classifiers specified by their model name.

    :param model_names: list of model code names as strings
    :return None
    """
    with pd.HDFStore(self.store_path, mode='r') as store:
      n_rows = len(store.select_as_coordinates(self.tables[0],
                                               where=where)
                   )

      chunksize = n_rows // 1

      for chunk in store.select(self.tables[0],
                                chunksize=chunksize,
                                where=where,
                                columns=self.feat_cols + self.label_cols):
        for modelName, modelValues in self.modelChoices.items():
          if modelName not in model_names:
            continue
          with stopwatch('training model %s' % modelName):
            modelValues.update(
              dict(model=self.train_model(chunk,
                                          modelName,
                                          modelValues,
                                          )
                   )
            )

  def get_trained_model(self, model_name):
    """ Returns pickled trained model

    :param model_name: string specifiying unique model name
    :return sklearn classifier object"""
    with open(self.get_pickled_name(model_name), 'rb') \
        as pckl_input:
      return pickle.load(pckl_input)


  def get_model_prediction(self, clf, where=None):
    with stopwatch("getting model predictions"):
      with pd.HDFStore(self.store_path, mode='a') as store:
        n_rows = len(store.select_as_coordinates(self.tables[0],
                                                 where=where)
                     )

        chunksize = n_rows // 1

        new_table_name = self.tables[0] + "_pred"
        if new_table_name in store.keys():
          store.remove(new_table_name)

        for chunk in store.select(self.tables[0],
                                  chunksize=chunksize,
                                  ):
          indexer = chunk.ANH
          y_pred = clf.predict(chunk.loc[indexer,self.feat_cols])
          chunk.loc[indexer, "predicted_label"] = \
            self.label_encoder.inverse_transform(y_pred)
          freq = stats.itemfreq(self.label_encoder.inverse_transform(y_pred))
          for f in freq:
            print(f[0], f[1])

          store.append(new_table_name,
                       chunk,
                       format='table',
                       append=False)
          return y_pred



if __name__ == "__main__":
  pass
