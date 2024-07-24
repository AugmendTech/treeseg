import numpy as np


class RandomSeg:
    def __init__(self, entries):
        self.entries = entries

    def segment_meeting(self, K):

        entries = self.entries
        N = len(entries)

        transitions = [0] * N
        indices = np.random.choice(N, K-1, replace=False)

        for idx in indices:
            transitions[idx] = 1

        return transitions


class EquiSeg:
    def __init__(self, entries):
        self.entries = entries

    def segment_meeting(self, K):

        entries = self.entries
        N = len(entries)

        period = N // (K-1)
        offset = (N%period + period) // 2

        transitions = [0]*N

        for k in range(K-1):
            transitions[offset + k*period] = 1

        return transitions
