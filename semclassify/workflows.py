import pandas as pd
pd.set_option('expand_frame_repr', False)
import numpy as np
from semclassify.plots import plot_labeled_image, PaletteController


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


if __name__ == "__main__":
  calc_phase_fraction()
