import pandas as pd
pd.set_option('expand_frame_repr', False)
import numpy as np
from semclassify.plots import (plot_labeled_image,
                               PaletteController)
from semclassify.classify import *

def calc_phase_fraction():

  df = pd.read_hdf("../output/store.h5",
                   "BG2",
                   where=[
                     "site='soi_011'",
                   ])

  print(df.info())
  print(df.head())

  ymax, xmax = (df.imgCol.max() + 1, df.imgRow.max() + 1)

  p = PaletteController()
  labeled_image = np.array(list(map(lambda x: p.num_rgb[3 * int(round(x))],
                                    df.ANH)))\
    .reshape(xmax, ymax, 3)
  print(labeled_image)
  print(labeled_image.shape)
  bse_image = df.Bse.reshape(xmax, ymax)
  plot_labeled_image(bse_image, labeled_image)

def train_models(supermodel):
  # TODO: pickle the supermodel
  supermodel.train_all_models(
    model_names=['rfc'],
    where=['site=="soi_001" | site=="soi_002" | site=="soi_011"'])
  supermodel.pickle_models(['rfc'])

def classify_all_anhydrous(supermodel):
  clf = supermodel.get_trained_model('rfc')
  return supermodel.get_model_prediction(clf,
                                         where=["ANH==True", "site=='soi_001'"])


if __name__ == "__main__":
  # calc_phase_fraction()
  sm = SuperModel(
    "/Users/joshuaarnold/Documents/MyApps/sem-classify/output/store.h5",
    tables=['BG2'],
    feat_cols=['Al', 'Ca', 'Fe', 'K', 'Mg', 'Na', 'S', 'Si'],
    label_cols=['BFS', 'FAF', 'HYD', 'ILL', 'POR', 'QS'],
  )
  train_models(sm)
  print(sm.label_encoder.inverse_transform(classify_all_anhydrous(sm)))

