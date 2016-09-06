import matplotlib
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']
matplotlib.rc('text', usetex=True)
from matplotlib import cm
import matplotlib.pyplot as plt
from pylab import imshow, show
import pandas as pd
pd.set_option('expand_frame_repr', False)
import numpy as np
from collections import OrderedDict
from skimage import io
from sklearn.metrics import confusion_matrix
from semclassify.helpers import stopwatch


class PaletteController():
  labels = ['None','POR','HYD','QS','ILL','BFS','OPC','FAF','FAF_0','FAF_1','FAF_2']
  numbers = range(0,len(labels)+1)
  colors = ["#ffffff","#000000","#ddccff", "#0000ff", "#ff00ff", "#00ff00", "#00ffff", "#ff0000",
           "#ff3333","#ff6666","#ff9999"]
  rgb = [(int(s[1:3],16),int(s[3:5],16),int(s[5:7],16)) for s in colors]

  label_num = OrderedDict(zip(labels,numbers))
  num_label = OrderedDict(zip(numbers,labels))
  label_color = OrderedDict(zip(labels,colors))
  num_color = OrderedDict(zip(numbers,colors))
  label_rgb = OrderedDict(zip(labels,rgb))
  num_rgb = OrderedDict(zip(numbers,rgb))

  cmap = matplotlib.colors.ListedColormap(colors,"mymap",len(colors))
  cmap.set_bad('w',0)
  cpal = sns.color_palette(colors)

  def show_palette(self):
    sns.palplot(self.cpal,)

    if False:
      ax = plt.gca()
      ax.annotate('FAF', xy=(2,0), xytext=(2,0),
                  horizontalalignment="center",
                  verticalalignment="middle",
                  color="white",
                  fontsize=16,
                  )
    plt.show()

  def print_palette_info(self):
    for d in (self.label_num, self.num_label, self.label_color):
      for t in d.items():
        print("%s -> %s" % t)


def plot_confusion_matrix(y_true,
                          y_pred,
                          title="Normalized confusion matrix",
                          cmap=plt.cm.Blues,
                          label_encoder=None):
  """ Plots the confusion matrix for a set of labels and predictions

  :param y_true: the matrix of integer class labels
  :param y_pred: the array of predicted classes
  :param title: plot title
  :param cmap: matplotlib colormap
  :return None
  """
  cm = confusion_matrix(y_true, y_pred)
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  labels = np.unique(y_true)
  print(labels)

  if label_encoder is not None:
    labels = label_encoder.inverse_transform(labels)
  print(labels)

  df_cm = pd.DataFrame(cm_normalized, columns=labels, index=labels)

  print('Normalized confusion matrix')
  print(df_cm)

  plt.figure()

  plt.imshow(df_cm.as_matrix(),
             interpolation='nearest',
             cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(df_cm.index))
  plt.xticks(tick_marks, df_cm.columns, rotation=0)
  plt.yticks(tick_marks, df_cm.index)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  plt.show()


def plot_labeled_image(bse_image, label_image, title="./output/classified.png"):
  """ Plots a BSE image above a label image
  :param bse_image: np array of BSE image values
  :param label_image: np array of label values"""
  with sns.axes_style(style='dark'):
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 5.5))
    axes[0].imshow(bse_image, cmap="gray")
    plt.setp(axes[0].get_yticklabels(), visible=False)
    plt.setp(axes[0].get_xticklabels(), visible=False)
    axes[1].imshow(label_image)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    fig.tight_layout()


     # fig.savefig(title, format='png', pad_inches=0.0,)
    plt.show()


if __name__ == "__main__":
  # p = PaletteController()
  # p.show_palette()
  # p.print_palette_info()
  # plt.show()

  plot_confusion_matrix([0, 0, 1, 1, 2], [0, 0, 0, 1, 1])

  # im = io.imread("/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/"
  #                "SEM-EDX-DATA/BFS/soi_001/TSV-TIFF/BSE.tif", flatten=True)

  # plot_labeled_image(im, im)
