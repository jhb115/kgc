#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging

'''
Without linear projection in the attention layer
'''
def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    command = f'PYTHONPATH=. python kgc/learn.py --dataset FB237 ' \
        f'--model ContExt ' \
        f'--regularizer N4 ' \
        f'--max_epoch 100 ' \
        f'--optimizer {c["optimizer"]} ' \
        f'--mkdir 1 --rank {c["rank"]} --load_pre_train {c["load_pre_train"]} --max_NB {c["max_NB"]} --valid 3 ' \
        f'--learning_rate {c["learning_rate"]} --reg {c["reg"]} --batch_size 500 --g_weight {c["g_weight"]} ' \
        f'--n_freeze {c["n_freeze"]} --evaluation_mode 1 --n_hop_nb {c["n_hop_nb"]} --dropout_g {c["dropout_g"]} ' \
        f'--rcp_bool {c["rcp_bool"]} --temperature {c["temperature"]}'
    return command


def main(argv):
    hyp_space = dict(
        rank=[500],
        max_NB=[50],
        g_weight=[0.],
        reg=[0.05],
        optimizer=['Adagrad'],
        n_freeze=[0],
        n_hop_nb=[1],
        load_pre_train=[0],
        dropout_g=[0.],
        learning_rate=[0.1, 0.05],
        rcp_bool=[0],
        temperature=["default", 20., 50.]
    )

    configurations = list(cartesian_product(hyp_space))

    # Check that we are on the UCLCS cluster first

    command_lines = set()
    for cfg in configurations:
        command_lines |= {to_cmd(cfg)}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """
#$ -S /bin/bash
#$ -o /home/jeunbyun/sgelogs
#$ -j y
#$ -N test2_fb237
#$ -l tmem=14G
#$ -l h_rt=92:00:00
#$ -l gpu=1

hostname
date

source /home/jeunbyun/conda.source
conda activate py36

date

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/jeunbyun/jeung_project/kgc

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('{}'.format(command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
