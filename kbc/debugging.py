from kbc.process_datasets import prepare_dataset


import pkg_resources
import os
import errno
from pathlib import Path
import pickle

import numpy as np

from collections import defaultdict

DATA_PATH = pkg_resources.resource_filename('kbc', 'data/')

name = d = 'WN18RR'
path = 'src_data/WN18RR'


files = ['train', 'valid', 'test']
entities, relations = set(), set()
for f in files:
    file_path = os.path.join(path, f)
    to_read = open(file_path, 'r')
    for line in to_read.readlines():
        lhs, rel, rhs = line.strip().split('\t')
        entities.add(lhs)
        entities.add(rhs)
        relations.add(rel)
    to_read.close()

entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
print("{} entities and {} relations".format(len(entities), len(relations)))
n_relations = len(relations)
n_entities = len(entities)




# map train/test/valid with the ids
for f in files:
    file_path = os.path.join(path, f)
    to_read = open(file_path, 'r')
    examples = []
    for line in to_read.readlines():
        lhs, rel, rhs = line.strip().split('\t')
        try:
            examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
        except ValueError:
            continue
    out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
    pickle.dump(np.array(examples).astype('uint64'), out)
    out.close()

print("creating filtering lists")

# create filtering files
to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
for f in files:
    examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
    for lhs, rel, rhs in examples:
        to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
        to_skip['rhs'][(lhs, rel)].add(rhs)

to_skip_final = {'lhs': {}, 'rhs': {}}
for kk, skip in to_skip.items():
    for k, v in skip.items():
        to_skip_final[kk][k] = sorted(list(v))

out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
pickle.dump(to_skip_final, out)
out.close()

examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
counters = {
    'lhs': np.zeros(n_entities),
    'rhs': np.zeros(n_entities),
    'both': np.zeros(n_entities)
}

for lhs, rel, rhs in examples:
    counters['lhs'][lhs] += 1
    counters['rhs'][rhs] += 1
    counters['both'][lhs] += 1
    counters['both'][rhs] += 1
for k, v in counters.items():
    counters[k] = v / np.sum(v)
out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
pickle.dump(counters, out)
out.close()

