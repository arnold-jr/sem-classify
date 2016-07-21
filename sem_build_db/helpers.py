import contextlib
import time
import json
import os
import pprint

@contextlib.contextmanager
def stopwatch(message):
  """Context manager that prints how long a block takes to execute."""
  t0 = time.time()
  try:
    yield
  finally:
    t1 = time.time()
    print('Total elapsed time for %s: %.3f s' % (message, t1 - t0))


def make_input_json():
  root = "/Users/joshuaarnold/Documents/Papers/VU_SEM/analysis/SEM-EDX-DATA"
  matDirs = os.listdir(root)

  matList = {'materials':
            [{'name':m, 'path':os.path.join(root, m),
            'sites':[
                    {'name':s, 'path': os.path.join(root,m,s),
                      'images':[
                        {'name':i.split(".")[0],
                         'path': os.path.join(root,m,s,'TSV-TIFF',i),
                         'type':'EDX' if i.split(".")[-1] == 'tsv' else 'BSE',
                         'maskName': i.split(".")[0]
                              if i.split(".")[-1].lower() == "tif"  and
                                i.split(".") not in ["inca","aztec"] else None
                        } for i in os.listdir(os.path.join(root,m,s,'TSV-TIFF'))
                        if i.split(".")[1] in ["tsv","tiff"]
                      ]
                     } for s in os.listdir(os.path.join(root, m)) if 'soi' in s
                    ]
              } for m in matDirs if m in ['BFS']
            ] }

  return matList

