import os
from pathlib import Path
import pkg_resources
import pickle
import sys

print(os.getcwd())
DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))
print(DATA_PATH)
name = 'FB15K'
root = DATA_PATH / name
f = 'train'