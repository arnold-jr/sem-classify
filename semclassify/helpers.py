# -*- coding: utf-8 -*-
import contextlib
import time
import json
import os

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
  root = "/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/SEM-EDX-DATA"
  matDirs = os.listdir(root)

  ctrlDict = {'materials':
            [{'name':m, 'path':os.path.join(root, m),
            'sites':[
                    {'name':s, 'path': os.path.join(root,m,s),
                      'images':[
                        {'name':i.split(".")[0],
                         'path': os.path.join(root,m,s,'TSV-TIFF',i),
                         'type':'EDX' if i.split(".")[-1] == 'tsv' else 'BSE',
                         'maskName': i.split(".")[0]
                              if i.split(".")[-1].lower() == "tif"  and
                                i.split(".")[0] not in ["inca","aztec"]
                              else None
                        } for i in os.listdir(os.path.join(root,m,s,'TSV-TIFF'))
                        if i.split(".")[1] in ["tsv", "tif", "tiff"]
                      ]
                     } for s in os.listdir(os.path.join(root, m)) if 'soi' in s
                    ]
              } for m in matDirs if m in ['BFS']
            ] }

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


if __name__ == "__main__":
  # make_input_json()
  # print(json.dumps(get_input_json("../input_data/ctrl_00.json"),indent=2))
  with stopwatch('hello',n_iter=1):
    print("hello")
  with stopwatch('hello',n_iter=1000000):
    print("hello")



