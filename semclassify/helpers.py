# -*- coding: utf-8 -*-
import contextlib
import time
import json
import os
import pandas as pd
from collections import OrderedDict


@contextlib.contextmanager
def stopwatch(message):
  """Context manager that prints how long a block takes to execute."""
  t0 = time.time()
  try:
    yield
  finally:
    t1 = time.time()
    print('Total elapsed time for %s: %f s' % (message, t1 - t0))

def chunks(l, n):
  """ Yield successive n-sized chunks from l.

  :param l: list
  :param n: number of elements in chunk
  :return iterator of size n
  """
  for i in range(0, len(l), n):
    yield l[i:i + n]

def full_head(df):
  """ Prints entire DataFrame head, avoiding ellision in the normal
  DataFrame.head() function.

  :param df: pandas DataFrame instance
  :return None
  """
  for l in list(chunks(df.columns.values,8)):
    print(df[l].head())


def make_input_json():
  """ Creates a JSON-like dict specifying the construction of the database
  building job.

  This is a helper function for specifying a DB construction control
  file, but eventually, this should be generated with a GUI. An actual JSON
  file is generated as record of the job.

  :return control-file JSON-like dict
  """
  parent = "/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/SEM-EDX-DATA"
  matDirs = os.listdir(parent)

  mask_names = ["BFS", "FAF", "BG2", "ANH", "HYD", "POR", "ILL", "QS"]

  ctrlDict = {'materials':
                [{'name': m, 'path': os.path.join(parent, m),
                  'sites': [
                    {'name': s, 'path': os.path.join(parent, m, s),
                     'images': [
                       {'name': i.split(".")[0],
                        'path': os.path.join(parent, m, s, 'TSV-TIFF', i),
                        'type': 'EDX' if i.split(".")[-1] == 'tsv' else 'BSE',
                        'maskName': i.split(".")[0]
                        if i.split(".")[0] in mask_names
                        else None
                        } for i in
                       os.listdir(os.path.join(parent, m, s, 'TSV-TIFF'))
                       if i.split(".")[1] in ["tsv", "tif", "tiff"]
                       ]
                     } for s in os.listdir(os.path.join(parent, m)) if
                    'soi' in s
                    ]
                  } for m in matDirs if m in ["BG2", "BFS", "FAF"]
                 ]
              }

  with open('../input_data/ctrl_00.json','w') as f:
    json.dump(ctrlDict, f, indent=2)

  return ctrlDict

def get_input_json(fileName):
  """ Creates a nested dict from the input JSON file.

  :param fileName: absolute path of the JSON input file
  :return control dict structure
  """
  with open(fileName, "r") as f:
    try:
      ctrlDict = json.load(f)
      return ctrlDict
    except:
      print("Error: Input JSON file could not be read.")
      raise


def specify_material_constants():

  BINDERS = ("BFS", "FAF")

  df_mix = pd.DataFrame(OrderedDict([
    ('Specific Gravity / -',
     {'BFS': 2.89, 'FAF': 2.33, 'OPC': 3.01, 'QS': 2.54, 'ILL': 2.75,
      'Water': 1.00}),
    ('Mean particle diameter / micron',
     {'BFS': 3.87, 'FAF': 6.02, 'OPC': 4.16,
      'QS': 590.6}),
    ('Mass / %',
     {'BFS': 8.0, 'FAF': 16.9, 'OPC': 5.40, 'QS': 48.4, 'ILL': 6.6,
      'Water': 14.6})
  ])
  )

  df_mix['Volume / %'] = df_mix['Mass / %'] / df_mix['Specific Gravity / -']
  df_mix['Volume / %'] = df_mix['Volume / %'] / df_mix['Volume / %'].sum() * 100
  df_mix.fillna('-').to_latex('../tables/mix_design.tex',
                              float_format=(lambda x: '%2.2f' % x))
  print(df_mix)


  # Read available contents as mg/kg and convert to g/kg
  df_available = pd.DataFrame(
    {'Al': {0: 6014.0699999999997, 1: 6161.5299999999997},
     'Ca': {0: 36082.199999999997, 1: 37470.300000000003},
     'Fe': {0: 1434.98, 1: 1467.0899999999999},
     'K': {0: 1442.1100, 1: 1486.1000},
     'Mg': {0: 6028.3599999999997, 1: 6204.4899999999998},
     'Na': {0: 166.87, 1: 169.69999999999999},
     'S': {0: 2490.9299999999998, 1: 2179.0300000000002},
     'Element': {0: 'Rep. 1', 1: 'Rep. 2'},
     'Si': {0: 16162.4, 1: 16207.1}})
  df_available = df_available.set_index('Element')
  df_available = df_available.transpose()
  df_available['Mean'] = df_available.mean(axis=1)
  df_available = df_available / 1000
  df_ac_sum = df_available.transpose().drop('Mean').describe().loc[
              ['count', 'mean', 'std'], :]
  df_ac_sum.loc['cov', :] = df_ac_sum.loc['std', :] / df_ac_sum.loc['mean', :]
  df_ac_sum.to_latex('../tables/available_content.tex',
                     float_format=(lambda x: "%2.2f" % x))
  print(df_available)


  # Totals of binders in g/kg
  df_total = pd.DataFrame({'BFS': {'Al': 34.600000000000001,
                                   'Ca': 245.5,
                                   'C': 0.0,
                                   'Cl': 0.25340000000000001,
                                   'Fe': 1.794,
                                   'K': 3.9060000000000001,
                                   'Mg': 73.200000000000003,
                                   'Na': 1.4359999999999999,
                                   'S': 17.100000000000001,
                                   'Si': 190.59999999999999},
                           'FAF': {'Al': 148.80000000000001,
                                   'Ca': 8.3160000000000007,
                                   'C': 6.0800000000000001,
                                   'Cl': 0.1764,
                                   'Fe': 51.150000000000006,
                                   'K': 30.780000000000001,
                                   'Mg': 6.2050000000000001,
                                   'Na': 2.8780000000000001,
                                   'S': 0.20019999999999999,
                                   'Si': 248.70000000000002},
                           'OPC': {'Al': 25.939999999999998,
                                   'Ca': 439.39999999999998,
                                   'C': 3.9199999999999999,
                                   'Cl': 0.42910000000000004,
                                   'Fe': 44.939999999999998,
                                   'K': 5.3249999999999993,
                                   'Mg': 6.875,
                                   'Na': 2.1320000000000001,
                                   'S': 12.0,
                                   'Si': 91.469999999999999}})

  df_total['Total Content'] = df_total.loc[:, BINDERS].as_matrix().dot(
    df_mix.loc[BINDERS, ['Mass / %']].as_matrix() / 100)
  df_total.to_latex('../tables/binder_totals.tex',
                    float_format=(lambda x: '%2.2f' % x))
  print(df_total)

  return df_mix, df_available, df_total




if __name__ == "__main__":
  make_input_json()
  print(json.dumps(get_input_json("../input_data/ctrl_00.json"),indent=2))



