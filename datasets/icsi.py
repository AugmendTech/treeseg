import os.path

import json
import xml.etree.ElementTree as ET

import structlog
logger = structlog.get_logger()

here = os.path.dirname(os.path.realpath(__file__))

from .data_utils import to_hhmmss, strip_key, AnnoTreeNode
from .dataset import SegmentationDataset

"""
Token types:
    *None: Non-text token (e.g. "breath-laugh")
    *HYPH: Hyphen "-" => Included without spacing
    *W: Word (actual text) => Included
    #LQUOTE: Opening quote "
    #RQUOTE: Closing quote "
    *TRUNCW: Truncated (partial) word => Discarded or included as a word
    *QUOTE: Quote:
        single => Treat as APOSS
        double => Either LQUOTE or RQUOTE depending on positioning
    *ABBR: Abbreviation or special name => treated as a word
    *LET: Letter spelled-out => Treated as a word. This do not cover all cases: e.g. I(LET)Ds(W) is rendered as I Ds
        That said if we always append then other cases fail: e.g. O(LET)O(LET)two(W) rendered as OOtwo instead of O O two
        Cant be perfect!
    *CM: Comma => Included, with space only after
    *SYM: Special symbol: Either @ or - => @ is discarded, - is treated as HYPH but with spacing.
    *.: Ending symbols: ".", "!" or "?" => Obviously included
    *APOSS: Apostrophe "'" accompanied by one or more characters (e.g. from "That's" the "'s" part) => Include, no spacing
    *CD: Digit => Included
"""


class ICSIDataset(SegmentationDataset):

    def __init__(self, fold, timed_utterances=False):

        # Minimum allowed segment. Segments below this size will be appended to the next
        # segment or if this does not exist, the previous one.
        self.MIN_SEGMENT_SIZE = 5

        self.meta_file = f"{here}/ICSI/ICSI-metadata.xml"
        self.anno_dir = f"{here}/ICSI/Contributions/TopicSegmentation"
        self.seg_dir = f"{here}/ICSI/Segments"
        self.word_dir = f"{here}/ICSI/Words"

        # If we want meeting entries to have a time range
        self.timed_utterances = timed_utterances

        metadata_xml_fnm = self.meta_file
        metadata_xml_tree = ET.parse(metadata_xml_fnm)
        metadata_root = metadata_xml_tree.getroot()

        topic_fnms = os.listdir(self.anno_dir)
        meetings = sorted(
            [fnm.split(".")[0] for fnm in topic_fnms if fnm.endswith(".xml")]
        )

        agents = metadata_root.find("agents")
        speakers = [child.attrib["name"] for child in agents]

        if fold=="dev":
            self.meetings = meetings[:5]
        else:
            self.meetings = meetings[5:]

        self.speakers = speakers


    def load_assets(self):
        # Loads all words
        self.load_all_words()

        # Loads seg files and uses words to build segments.
        self.load_all_utterances()


    def load_all_words(self):

        self.words = {}

        for meeting in self.meetings:
            self.words[meeting] = self.load_meeting_words(meeting)

    def load_meeting_words(self, meeting):

        # ICSI contain a word file for every meeting and speaker
        # This file is essentially an index of all spoken words
        # (+ non-verbal stuff)

        # Here, for each meeting and speaker we have a list of
        # words "in-order" and an index translating a word key
        # into its position in the list. This is needed for
        # parsing segments

        word_dir = self.word_dir
        speakers = self.speakers

        meeting_words = {}

        for speaker in speakers:

            quote_stack = []

            words_xml_fnm = f"{word_dir}/{meeting}.{speaker}.words.xml"

            if not os.path.isfile(words_xml_fnm):
                # Sometimes files won't exist, since not every speaker was present
                # in every meeting.
                logger.warning(f"File not found: {words_xml_fnm} => Skipping")
                continue

            root = ET.parse(words_xml_fnm).getroot()

            meeting_words[speaker] = {"words": [], "index": {}}

            for child in root:

                key = child.attrib["{http://nite.sourceforge.net/}id"]

                # Words that do not contain the "c" attribute correspond to
                # comments or other non-spoken entries. These will be included
                # as placeholder None entries.
                word_type = child.attrib.get("c", None)
                text = child.text

                if text:
                    text = text.lstrip().rstrip()

                # Contains the logic for treating each word entry according to its type
                word = self.make_word_entry(word_type, text, quote_stack)

                meeting_words[speaker]["words"].append(word)
                meeting_words[speaker]["index"][key] = (
                    len(meeting_words[speaker]["words"]) - 1
                )

            assert len(meeting_words[speaker]["words"]) == len(root)
            assert len(meeting_words[speaker]["index"]) == len(root)

        return meeting_words

    def load_all_utterances(self):

        self.index = {}

        for meeting in self.meetings:

            self.index[meeting] = {}

            meeting_utterances = self.load_meeting_utterances(meeting)

            entries = []

            for speaker in meeting_utterances:
                entries.extend(meeting_utterances[speaker])

                self.index[meeting][speaker] = {}

            sot = min(
                [entry["start"] for entry in entries if entry["start"] is not None]
            )

            for i, entry in enumerate(entries):
                entry["sot"] = sot
                key = entry["key"]
                speaker = entry["speaker"]

                self.index[meeting][speaker][key] = entry

    def load_meeting_utterances(self, meeting):

        seg_dir = self.seg_dir

        utterances = {}

        logger.info(f"Extracting utterances for meeting {meeting}")

        logger.info(f"Speakers: {self.words[meeting].keys()}")

        for speaker in self.words[meeting]:

            logger.info(
                f"Extracting utterances for meeting {meeting} and speaker {speaker}"
            )
            segs_xml_fnm = f"{seg_dir}/{meeting}.{speaker}.segs.xml"

            if not os.path.isfile(segs_xml_fnm):
                logger.error(
                    f"Act file {segs_xml_fnm} not found, while words file exists. Are you missing an act file?"
                )
                exit(-1)
                continue

            segs = []

            for child in ET.parse(segs_xml_fnm).getroot():
                if child.attrib.get("type", "") == "supersegment":
                    for grandchild in child:
                        segs.append(grandchild)
                else:
                    segs.append(child)

            utterances[speaker] = []

            for seg in segs:

                if not list(seg):
                    continue

                seg_id = seg.attrib["{http://nite.sourceforge.net/}id"]
                seg_start = seg.attrib["starttime"]
                seg_end = seg.attrib["endtime"]
                word_range = list(seg)[0].attrib["href"].split("#")[-1]

                if ".." in word_range:
                    start_key, end_key = word_range.split("..")
                else:
                    start_key = end_key = word_range

                start_key = strip_key(start_key)
                end_key = strip_key(end_key)

                start_idx = self.words[meeting][speaker]["index"][start_key]
                end_idx = self.words[meeting][speaker]["index"][end_key]

                utterance = None

                for t in range(start_idx, end_idx + 1):

                    word = self.words[meeting][speaker]["words"][t]

                    if word is None:
                        continue

                    utterance = self.absorb_token(utterance, word)

                if utterance:
                    utterance["speaker"] = speaker
                    utterance["key"] = seg_id
                    utterance["start"] = float(seg_start) if seg_start else None
                    utterance["end"] = float(seg_end) if seg_end else None

                    # This only happens for 3 utterances in the whole corpus
                    if utterance["end"] is None:
                        utterance["end"] = utterance["start"] + 1.0
                    utterances[speaker].append(utterance)

        return utterances

    def absorb_token(self, utterance, token):

        if utterance is None:
            return token

        spaced = utterance.get("rspace", True) and token["lspace"]

        suffix = ""

        if spaced:
            suffix += " "

        suffix += token["text"]

        utterance["text"] += suffix
        utterance["rspace"] = token["rspace"]

        return utterance

    def make_word_entry(self, word_type, text, quote_stack):

        # tag: is the type of word
        # lspace: does the netry warrant a left-side space
        # rspace: does the netry warrant a right-side space
        # text: text content of entry

        entry = {
            "tag": word_type,
            "lspace": True,
            "rspace": True,
            "text": text,
        }

        if word_type in ["W", "TRUNCW"]:

            # words that start with ' are treated as APOSS (apostrophe) entries
            if text[0] == "'":
                return self.make_word_entry("APOSS", text, quote_stack)
            else:
                return entry

        # Abbreviations, letters and digits are all treated as text
        if word_type == "ABBR":
            return entry

        if word_type == "LET":
            return entry

        if word_type == "CD":
            return entry

        # Symbols are either hyphens or @ (unknown purpose)
        # - : treated as HYPH
        # @ : discarded

        if word_type == "SYM":
            if text == "-":
                return self.make_word_entry("HYPH", text, quote_stack)
            else:
                return None

        if word_type == "HYPH":
            # No spacing for hyphens
            entry["lspace"] = False
            entry["rspace"] = False
            return entry

        if word_type == "CM":
            # No left space for commas
            entry["lspace"] = False
            return entry

        if word_type == ".":
            # No left space for periods
            entry["lspace"] = False
            return entry

        if word_type == "APOSS":
            # No left space for entries starting with '
            entry["lspace"] = False
            return entry

        # This is the juicy terittory
        # ICSI can have QUOTE, LQUOTE or RQUOTE
        # LQUOTE and RQUOTE denote left and right double quotes
        # QUOTE can be single in which case we treat is as apostrophe
        if word_type == "LQUOTE":
            if text == "'":
                entry["tag"] = "APOSS"
                entry["lspace"] = False
                entry["rspace"] = False
                return entry
            quote_stack.append("LEFT")
            entry["rspace"] = False
            return entry

        if word_type == "RQUOTE":
            if text == "'":
                entry["tag"] = "APOSS"
                entry["lspace"] = False
                entry["rspace"] = False
                return entry
            entry["lspace"] = False
            if not quote_stack:
                logger.error(
                    "Mismatched quotes. Found closing quote with no opening quote."
                )
            else:
                quote_stack.pop()
            return entry

        if word_type == "QUOTE":
            if text == "'":
                entry["tag"] = "APOSS"
                entry["lspace"] = False
                entry["rspace"] = False
                return entry
            if not quote_stack:
                return self.make_word_entry("LQUOTE", text, quote_stack)
            else:
                return self.make_word_entry("RQUOTE", text, quote_stack)

        if word_type is None:
            return None

        raise Exception("UnsupportedType", word_type)

    def segments_to_utterances(self, segments):

        utterances = []

        for segment in segments:
            meta, utt_content = segment.attrib["href"].split("#")
            meeting = meta.split(".")[0]
            speaker = meta.split(".")[1]

            utt_keys = []

            if ".." in utt_content:
                start_utt, end_utt = utt_content.split("..")

                start_utt = strip_key(start_utt)
                end_utt = strip_key(end_utt)

                start_tokens = start_utt.split(".")
                end_tokens = end_utt.split(".")

                start_idx = int(start_tokens[-1].replace(",", ""))
                end_idx = int(end_tokens[-1].replace(",", ""))
                stem = ".".join(start_tokens[:-1])

                for i in range(start_idx, end_idx + 1):

                    utt_id = str(i)

                    if i >= 1000:
                        utt_id = utt_id[:-3] + "," + utt_id[-3:]

                    utt_keys.append(f"{stem}.{utt_id}")

            else:
                utt_keys.append(strip_key(utt_content))

            for key in utt_keys:
                if key not in self.index[meeting][speaker]:
                    # logger.warning(f"Pruning key: {speaker}:{key}")
                    continue
                utterances.append(self.index[meeting][speaker][key])

        for utterance in utterances:
            utterance["composite"] = self.compose_utterance(utterance)

        keys = [utt["key"] for utt in utterances]

        return keys, utterances

    def build_anno(self, path, xml_node):

        anno_node = AnnoTreeNode()
        anno_node.path = path

        branch_id = 0

        children = [entry for entry in xml_node]
        anno_node.is_leaf = not any([child.tag == "topic" for child in children])

        if "root" in xml_node.tag:
            anno_node.tag = "root"
        elif anno_node.is_leaf:
            anno_node.tag = "leaf"
        else:
            anno_node.tag = "topic"

        while children:

            if children[0].tag == "topic":
                entry = children.pop(0)
                if entry:
                    nn_path = f"{anno_node.path}.{branch_id}"
                    anno_node.nn.append(self.build_anno(nn_path, entry))
                    branch_id += 1
                continue

            segments = []

            while children and children[0].tag != "topic":
                segments.append(children.pop(0))

            keys, convo = self.segments_to_utterances(segments)

            if not keys:
                continue

            if not anno_node.is_leaf:

                composed_topic = AnnoTreeNode()
                composed_topic.keys = keys
                composed_topic.convo = convo
                composed_topic.tag = "leaf"

                composed_topic.path = f"{anno_node.path}.{branch_id}"
                composed_topic.is_leaf = True

                anno_node.nn.append(composed_topic)
                branch_id += 1
                continue
            else:
                anno_node.keys = keys
                anno_node.convo = convo
                break

        if anno_node.is_leaf and not anno_node.keys:
            logger.warning(f"Node {anno_node.path} has no keys!")

        return anno_node

    def load_anno_trees(self):

        anno_dir = self.anno_dir

        self.anno_roots = {}
        self.leaves = {}

        for meeting in self.meetings:

            logger.info(f"Building anno tree for meeting: {meeting}")
            anno_xml_fnm = f"{anno_dir}/{meeting}.topic.xml"

            if not os.path.isfile(anno_xml_fnm):
                logger.error(f"Anno file missing: {anno_xml_fnm}")
                exit(-1)

            root = ET.parse(anno_xml_fnm).getroot()

            anno_root = self.build_anno("*", root)
            self.anno_roots[meeting] = anno_root

    def compose_utterance(self, utterance):
        speaker = utterance["speaker"]
        sot = utterance["sot"]
        start = (
            round(utterance["start"] - sot, 1)
            if utterance["start"] is not None
            else None
        )
        end = round(utterance["end"] - sot, 1) if utterance["end"] is not None else None
        text = utterance["text"]
        key = utterance["key"]

        if self.timed_utterances:
            return f"[{to_hhmmss(start,include_milli=True)}-{to_hhmmss(end,include_milli=True)}] Speaker {speaker}: {text}"
        return f"-{text}"
