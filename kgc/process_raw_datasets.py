
import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import shutil

import numpy as np

from collections import defaultdict

# DATA_PATH = pkg_resources.resource_filename('kgc', 'data/')  # this is where the processed data get stored.
#
# if os.path.exists(DATA_PATH + '/YAGO3-10'):
#     shutil.rmtree(DATA_PATH + '/YAGO3-10')
#
# if os.path.exists(DATA_PATH + '/WN18RR'):
#     shutil.rmtree(DATA_PATH + '/WN18RR')
#
# if os.path.exists(DATA_PATH + '/FB237'):
#     shutil.rmtree(DATA_PATH + '/FB237')

def prepare_dataset(path='../dataset/raw_data', name='YAGO3-10'):  # this path is where the raw data lies
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()

    for f in files:

        file_path = os.path.join(path, name, f)
        to_read = open(file_path + '.txt', 'r')
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
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(path, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, name, f)
        to_read = open(file_path + '.txt', 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        out = open(os.path.join(path, name, (f + '.pickle')), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(os.path.join(path, name, (f + '.pickle')), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(os.path.join(path, name, 'to_skip.pickle'), 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(os.path.join(path, name, 'train.pickle'), 'rb'))
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
    out = open(os.path.join(path, name, 'probas.pickle'), 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    for each_data in ['YAGO3-10', 'WN18RR', 'FB237']:
        print("Preparing dataset {}".format(each_data))
        try:
            prepare_dataset('../dataset/raw_data', each_data)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise

