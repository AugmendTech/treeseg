import torch
import torchhd


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords


def n_grams(string, n):
    return [string[i : i + n] for i in range(len(string) - n + 1)]


def get_local_minima(array, threshold=None):
    local_minima_indices = []
    local_minima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] > array[i] and array[i] < array[i + 1]:
            # fixed from original code to correctly pick minima below threshold
            if threshold is None or array[i] < threshold:
                local_minima_indices.append(i)
                local_minima_values.append(array[i])
    return local_minima_indices, local_minima_values


class HyperSeg:

    # ngrams, tokenizer will be None
    def __init__(self, entries, dimension=10000, ngrams=None, tokenizer=None):
        
        # object initialized with the meeting entries
        self.entries = entries
        self.dimension = dimension
        self.tokenizer = tokenizer
        self.ngrams = ngrams

    def segment_meeting(self, K):

        # extracting meeting notes from meeting entries
        utterances = [entry["composite"] for entry in self.entries]

        if not utterances:
            return []

        torch.manual_seed(0)

        word_list, word_to_index = [], {}
        tokenized_utterances = []

        original_index = {}  # We only want to collect vectors from valid utterances.
        for ui, utterance in enumerate(utterances):
            if not utterance.strip():
                print(f"Empty utterance: {utterance}")
                continue

            utterance = utterance.lower()
            words = (
                utterance.split()
                if self.ngrams is None
                else n_grams(utterance, self.ngrams)
            )
            tokenized_utterance = []
            for word in words:
                # Original threw an error since stopwords is of type LazyCorpusLoader
                #if word in stopwords.words("english") and self.ngrams is None:
                #    continue

                if word in word_to_index:
                    word_id = word_to_index[word]
                else:
                    word_id = len(word_list)
                    word_to_index[word] = word_id
                    word_list.append(word)
                tokenized_utterance.append(word_id)
            if not tokenized_utterance:
                print(f"No token utterance: {utterance}, ngram={self.ngrams}")
                continue
            original_index[len(tokenized_utterances)] = ui
            tokenized_utterances.append(tokenized_utterance)

        device = "cpu"
        token_embeddings = torchhd.circular(
            len(word_list), self.dimension, device=device
        )

        utterance_vectors = []
        for tokenized_utterance in tokenized_utterances:
            symbols = token_embeddings[tokenized_utterance]
            utterance_vector = torchhd.bundle_sequence(symbols)
            utterance_vectors.append(utterance_vector)

        sims = [
            torchhd.cosine_similarity(prev, curr)
            for prev, curr in zip(utterance_vectors[:-1], utterance_vectors[1:])
        ]
        sims = torch.tensor(sims, device=device)

       
        minima_indices, minima_values = get_local_minima(sims)
        
        # Altered code to keep the first K-1 minima which partition the transcript into K segments
        minima_argmin = torch.argsort(torch.tensor(minima_values))[:K-1]
        segment_indices = [minima_indices[i] for ii, i in enumerate(minima_argmin)]

        # Adding back a zero in the end to match the dataset format
        transitions_hat = [0] * len(utterances)
        for index in segment_indices:
            transitions_hat[original_index[index]] = 1

        return transitions_hat