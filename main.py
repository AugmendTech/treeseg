import os, sys

import json

import numpy as np

from treeseg import TreeSeg
from bertseg import BertSeg
from hyperseg import HyperSeg
from configs import treeseg_configs,bertseg_configs

from baselines import RandomSeg, EquiSeg
import datasets

import matplotlib.pyplot as plt
from nltk.metrics import windowdiff, pk


import structlog

logger = structlog.get_logger("base")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["treeseg", "bertseg", "hyperseg", "random", "equi","view"],
)
parser.add_argument(
    "--dataset", type=str, required=True, choices=["augmend", "ami", "icsi"]
)
parser.add_argument("--mid", type=int, required=False, help="Meeting index (zero-indexed) to segment. If this is not set the entire fold will be processed.")
parser.add_argument('--eval', action='store_true', help='Evaluation mode')
parser.add_argument('--fold', required=True, choices=["dev", "test"], help='Dataset fold')


args = parser.parse_args()

MODEL_NAME = args.model
DATASET = args.dataset
MID = args.mid
EVAL = args.eval
FOLD = args.fold


if DATASET == "icsi":
    dataset = datasets.ICSIDataset(fold=FOLD)
elif DATASET == "ami":
    dataset = datasets.AMIDataset(fold=FOLD)
elif DATASET == "augmend":
    dataset = datasets.AugmendDataset(fold=FOLD, include_actions=False)
else:
    raise Exception("UnsupportedDataset", DATASET)

dataset.load_dataset()

if EVAL:
    targets = dataset.meetings
else:
    targets = [dataset.meetings[MID]]



metrics = {}
max_lvl = {}

for meeting in targets:

    max_lvl[meeting] = -float("inf")

    metrics[meeting] = {
        "pk": {},
        "wdiff": {},
    }


num_entries = 0
num_raw_tr = [0]*4
num_tr = [0]*4
num_lvl = [0]*4


for meeting in targets:

    print("\n\n\n")
    entries = dataset.notes[meeting]

    hier_raw_transitions = dataset.hier_raw_transitions[meeting]
    hier_transitions = dataset.hier_transitions[meeting]

    logger.info(f"Segmenting meeting {meeting}")



    if MODEL_NAME == "view":
        anno_root = dataset.anno_roots[meeting]
        leaves = dataset.discover_anno_leaves(anno_root)

        for leaf in leaves:
            print("\n++++++++++++++++++++++++++++++++++++")
            print(f"Segment leaf identifier: {leaf.path}")
            print("Segment transcript:")
            for entry in leaf.convo:
                print(entry["composite"])
            print("++++++++++++++++++++++++++++++++++++\n")
            input("Press any key + ENTER to continue...")
        exit(0)

    elif MODEL_NAME == "treeseg":
        model = TreeSeg(configs=treeseg_configs[DATASET], entries=entries)
    elif MODEL_NAME == "bertseg":
        model = BertSeg(configs=bertseg_configs[DATASET], entries=entries)
    elif MODEL_NAME == "hyperseg":
        model = HyperSeg(entries=entries)
    elif MODEL_NAME == "random":
        model = RandomSeg(entries=entries)
    elif MODEL_NAME == "equi":
        model = EquiSeg(entries=entries)
    else:
        raise Exception("UnsupportedModel", MODEL_NAME)



    repeats = 100 if MODEL_NAME=="random" else 1

    for lvl, labs in enumerate(zip(hier_raw_transitions,hier_transitions)):

        pk_loss = 0.0
        wdiff_loss = 0.0

        for repeat in range(repeats):
            raw_transitions, transitions = labs

            true_K = int(sum(transitions)) + 1
            transitions_hat = model.segment_meeting(true_K)

            if MODEL_NAME!="bertseg":
                assert sum(transitions_hat) == sum(transitions)

            assert len(transitions_hat) == len(transitions)
            assert len(transitions_hat) == len(entries)

            logger.info(f"Ground truth K: {sum(transitions)}")
            logger.info(f"K returned from model: {sum(transitions_hat)}")

            diff_k = int(round(len(transitions) / (true_K * 2.0)))

            tr_str = "".join([str(lab) for lab in transitions])
            tr_hat_str = "".join([str(lab) for lab in transitions_hat])

            wdiff_loss += windowdiff(tr_str, tr_hat_str, diff_k)/repeats
            pk_loss += pk(tr_str, tr_hat_str, diff_k)/repeats

        logger.info(f"windowdiff:{wdiff_loss}")
        logger.info(f"pk:{pk_loss}")


        metrics[meeting]["pk"][lvl] = pk_loss
        metrics[meeting]["wdiff"][lvl] = wdiff_loss

        if not EVAL:
            plt.subplot(3, 1, 1)
            plt.title("Original segment transitions")
            plt.plot(raw_transitions, color="r")
            plt.subplot(3, 1, 2)
            plt.title("Pruned segment transitions")
            plt.plot(transitions)
            plt.subplot(3, 1, 3)
            plt.title("Inferred segment transitions")
            plt.plot(transitions_hat, color="g")
            plt.show()
            
    if MODEL_NAME=="treeseg":
        for leaf in model.leaves:
            max_lvl[meeting] = max(max_lvl[meeting],len(leaf.identifier)-1)


M = len(targets)

sum_pk_loss = 0.0
sum_wdiff_loss = 0.0
total_count = 0


sum_lvl_pk = [0.0]*4
sum_lvl_wdiff = [0.0]*4
lvl_count = [0]*4

avg_max_level = 0


for meeting in metrics.keys():

    assert set(metrics[meeting]["pk"]) == set(metrics[meeting]["wdiff"])

    if MODEL_NAME=="treeseg":
        avg_max_level += max_lvl[meeting]/len(targets)

    for lvl in metrics[meeting]["pk"]:
        lvl_count[lvl] += 1
        total_count += 1

    for lvl in metrics[meeting]["pk"]:
        val = metrics[meeting]["pk"][lvl]
        sum_pk_loss += val
        sum_lvl_pk[lvl] += val

    for lvl in metrics[meeting]["wdiff"]:
        val = metrics[meeting]["wdiff"][lvl]
        sum_wdiff_loss += val
        sum_lvl_wdiff[lvl] += val


avg_pk_loss = sum_pk_loss/total_count
avg_wdiff_loss = sum_wdiff_loss/total_count

avg_lvl_pk_loss = [pk_sum/count if count>0 else None for pk_sum,count in zip(sum_lvl_pk,lvl_count)]
avg_lvl_wdiff_loss = [wdiff_sum/count if count>0 else None for wdiff_sum,count in zip(sum_lvl_wdiff,lvl_count)]


logger.info(f"Avg total pk loss: {avg_pk_loss}")
logger.info(f"Avg level pk loss: {avg_lvl_pk_loss}")

logger.info(f"Avg total wdiff loss: {avg_wdiff_loss}")
logger.info(f"Avg level wdiff loss: {avg_lvl_wdiff_loss}")


results = {
    "detailed": metrics,
    "total": {
        "pk": avg_pk_loss,
        "wdiff": avg_wdiff_loss,
    },
    "level": {
        "pk": avg_lvl_pk_loss,
        "wdiff": avg_lvl_wdiff_loss,
    },
    "total_count": total_count,
    "level_counts": lvl_count,
    "avg_max_level": avg_max_level,
    "max_level": max_lvl,
}

f = open(f"./results/{DATASET}_{FOLD}_{MODEL_NAME}.json","w")
f.write(json.dumps(results,indent=4))
f.close()

