
import structlog
logger = structlog.get_logger()

class SegmentationDataset:

    def __init__(self):
        pass

    def load_assets(self):

        raise Exception(
            "AbstractMethod",
            "Method load_assets() needs to be implemented by sub-class!",
        )

    def load_anno_tree(self):
        raise Exception(
            "AbstractMethod",
            "Method load_anno_tree() needs to be implemented by sub-class!",
        )

    def load_dataset(self):

        self.load_assets()

        # Loads annotation tree
        self.load_anno_trees()

        self.compose_meeting_notes()

        for meeting in self.meetings:
            root = self.anno_roots[meeting]
            self.propagate_anno_segments(root)

            #self.print_anno_tree(root,meeting)

        self.get_hierarchical_transitions()

    def discover_anno_leaves(self, anno_root):

        stack = [anno_root]
        curr = anno_root

        leaves = []

        while stack:

            node = stack.pop()

            if node.is_leaf:
                leaves.append(node)

            for child in node.nn[::-1]:
                stack.append(child)

        return leaves


    def print_anno_node(self,node):
        print(f"Path: {node.path}:{node.tag}")
        print(f"Is leaf: {node.is_leaf}")

        if node.keys:
            print(f"Keys: {node.keys[0]}=>{node.keys[-1]}")

    def print_anno_tree(self, root, meeting):

        print(f"=============== {meeting} ==========================")
        queue = [root]

        while queue:

            node = queue.pop(0)

            print("\n<--------------")
            self.print_anno_node(node)
            print("-------------->\n")

            for child in node.nn:
                queue.append(child)
        print("==============================================")


    def compose_meeting_notes(self):

        self.notes = {}
        self.labs = {}
        self.transitions = {}
        self.raw_transitions = {}

        for meeting in self.meetings:

            logger.info(f"Composing notes for meeting: {meeting}")
            self.notes[meeting] = []
            self.raw_transitions[meeting] = []
            anno_root = self.anno_roots[meeting]

            prev_leaf_id = 0

            anno_leaves = self.discover_anno_leaves(anno_root)

            for leaf in anno_leaves:
                for utt in leaf.convo:
                    utt["composite"] = self.compose_utterance(utt)
                    self.notes[meeting].append(utt)

            raw_transitions, transitions = self.transitions_from_boundary(anno_leaves)

            self.raw_transitions[meeting] = raw_transitions
            self.transitions[meeting] = transitions

    def propagate_anno_segments(self, node):

        if node.is_leaf:
            return node.convo, node.keys

        convo = []
        keys = []

        for child in node.nn:

            if child.is_leaf:
                convo.extend(child.convo)
                keys.extend(child.keys)
            else:
                child_convo, child_keys = self.propagate_anno_segments(child)
                convo.extend(child_convo)
                keys.extend(child_keys)

        node.convo = convo
        node.keys = keys

        return convo, keys

    def transitions_from_boundary(self, boundary):

        first_node = boundary.pop(0)
        raw_transitions = [0] * len(first_node.convo)

        while boundary:
            node = boundary.pop(0)
            if node.keys:
                segment_transitions = [0] * len(node.convo)
                segment_transitions[0] = 1
                raw_transitions.extend(segment_transitions)
            else:
                logger.warning(f"Node with path {node.path} has no keys!")

        transitions = raw_transitions.copy()

        for i, _ in enumerate(raw_transitions):
            if sum(transitions[i - self.MIN_SEGMENT_SIZE + 1 : i]) > 0:
                transitions[i] = 0

        return raw_transitions, transitions

    def get_boundary_tag(self, boundary):
        return "|".join([node.path for node in boundary])

    def get_hierarchical_transitions(self):

        self.hier_raw_transitions = {}
        self.hier_transitions = {}

        for meeting in self.meetings:
            logger.info(f"Extracting hierarhical segment transitions for meeting: {meeting}")
            anno_root = self.anno_roots[meeting]
            raw_tr, tr, tags = self.get_meeting_hierarchical_transitions(anno_root)
            self.hier_raw_transitions[meeting] = raw_tr
            self.hier_transitions[meeting] = tr

            assert len(tags) == len(set(tags))


    def get_meeting_hierarchical_transitions(self, root):

        raw_boundary_transitions = []
        boundary_transitions = []

        boundary = [root]
        boundary_tag = self.get_boundary_tag(boundary)
        boundary_tags = []
        while True:

            new_boundary = self.expand_boundary(boundary.copy())
            new_boundary_tag = self.get_boundary_tag(new_boundary)

            if new_boundary_tag == boundary_tag:
                break

            raw_tr, tr = self.transitions_from_boundary(new_boundary.copy())

            raw_boundary_transitions.append(raw_tr)
            boundary_transitions.append(tr)
            boundary_tags.append(new_boundary_tag)

            boundary_tag = new_boundary_tag
            boundary = new_boundary

        return raw_boundary_transitions, boundary_transitions, boundary_tags

    def expand_boundary(self, boundary):

        expanded = []

        while boundary:

            node = boundary.pop(0)

            if node.is_leaf:
                expanded.append(node)
            else:
                expanded.extend(node.nn)
        return expanded
