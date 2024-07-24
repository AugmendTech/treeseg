import os, sys
here = os.path.dirname(os.path.realpath(__file__))

import json

import structlog
logger = structlog.get_logger()

from .data_utils import to_hhmmss, strip_key, AnnoTreeNode
from .dataset import SegmentationDataset



class AugmendDataset(SegmentationDataset):

    def __init__(self, fold, include_actions=False):

        self.include_actions = include_actions
        self.MIN_SEGMENT_SIZE = 5

        if fold=="dev":
            dataset_dir = f"{here}/augmend/dev"
        else:
            dataset_dir = f"{here}/augmend/test"

        fnms = os.listdir(dataset_dir)

        self.anno_fnms = [f"{dataset_dir}/{fnm}" for fnm in fnms if "anno" in fnm]
        self.data_fnms = [f"{dataset_dir}/{fnm}" for fnm in fnms if "anno" not in fnm]

        self.meetings = sorted([fnm.replace(".json","") for fnm in fnms if "anno" not in fnm])




    def load_assets(self):

        self.data = {}
        self.annos = {}

        for fnm in self.anno_fnms:
            logger.info(f"Loading annos from file: {fnm}")
            f = open(fnm,"r")
            self.annos.update(json.loads(f.read()))
            f.close()
            
        for fnm in self.data_fnms:
            logger.info(f"Loading data from file: {fnm}")
            f = open(fnm,"r")
            self.data.update(json.loads(f.read()))
            f.close()


    def load_anno_trees(self):

        self.anno_roots = {}

        for meeting in self.meetings:
            coarse = self.annos[meeting]["coarse"]
            fine = self.annos[meeting]["fine"]

            root = AnnoTreeNode()
            root.tag = "root"
            root.path = "*"
            root.is_leaf = False


            for higher_branch_id, coarse_seg in enumerate(coarse):

                coarse_start, coarse_end, _ = coarse_seg

                coarse_node = AnnoTreeNode()
                coarse_node.tag = "coarse_topic"
                coarse_node.path = root.path + f".{higher_branch_id}"
                coarse_node.is_leaf = False


                lower_branch_id = 0

                while fine:

                    fine_start, fine_end, _ = fine[0]

                    if fine_start>=coarse_start and fine_end<=coarse_end:
                        fine_node = AnnoTreeNode()
                        fine_node.tag = "fine_topic"
                        fine_node.path = coarse_node.path + f".{lower_branch_id}"
                        fine_node.is_leaf = True
                        lower_branch_id += 1


                        fine_node.convo = []
                        fine_node.keys = []

                        for idx in range(fine_start,fine_end+1):
                            entry = self.data[meeting]["entries"][idx]

                            if "doc" in entry or self.include_actions:
                                fine_node.convo.append(entry)
                                fine_node.keys.append(idx)
                        coarse_node.nn.append(fine_node)
                        fine.pop(0)
                    else:
                        break
                
                root.nn.append(coarse_node)
                
            self.anno_roots[meeting] = root


    def compose_utterance(self,utt):

        if "doc" in utt:
            return f"-{utt['doc']}"
        else:
            return utt["emb_doc"]

