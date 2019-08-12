#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    command = f'python kbc/learn_grid.py --dataset WN18RR ' \
        f'--model Context_ComplEx ' \
        f'--regularizer N4 ' \
        f'--max_epoch 140 ' \
        f'--optimizer {c["optimizer"]} ' \
        f'--mkdir 1 --rank {c["rank"]} --load_pre_train 1 --max_NB {c["max_NB"]} --valid 3 ' \
        f'--learning_rate 0.01 --reg {c["reg"]} --batch_size 300 --g_weight {c["g_weight"]} ' \
        f'--ascending {c["ascending"]}'
    return command


def main(argv):
    hyp_space = dict(
        rank=[200, 400],
        max_NB=[50, 150],
        g_weight=[0.03, 0.06, 0.1],
        reg=[0.01, 0.08],
        ascending=[-1, 1],
        optimizer=['Adagrad', 'Adam'],
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
#$ -N fb237_ContExt_v1
#$ -l tmem=9G
#$ -l h_rt=92:00:00
#$ -l gpu=1

hostname
date

source /home/jeunbyun/conda.source
conda activate py36

date

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/jeunbyun/jeung_project/kbc

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('{}'.format(command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
