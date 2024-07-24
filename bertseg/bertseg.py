#!/usr/bin/env python3
import numpy as np
import pandas as pd

import json
import requests

import asyncio
import aiohttp

import matplotlib.pyplot as plt

import structlog

logger = structlog.get_logger()


class BertSeg:

    def __init__(self, configs, entries):
        self.configs = configs
        self.entries = entries

        # This part submits the utterance entries to OpenAI to obtain
        # utterance embeddings.

        loop = asyncio.get_event_loop()

        N = len(self.entries)
        BATCH_SIZE = 500
        num_batches = np.int32(np.ceil(N / BATCH_SIZE))

        logger.info(f"Extracting {N} embeddings in {num_batches} batches.")

        batches = []

        for batch_i in range(num_batches):
            batch = self.entries[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
            batches.append(batch)

        embedded = False

        while not embedded:

            try:
                result = loop.run_until_complete(
                    asyncio.gather(
                        *[
                            self.get_embeddings([entry["composite"] for entry in batch])
                            for batch in batches
                        ]
                    )
                )
                embedded = True
            except Exception as e:
                logger.warning(f"Embedding failed ({str(e)}): Retrying")
                continue

        embs = []

        for batch_embs in result:
            embs.extend(batch_embs)
            
        logger.info(f"I have collected {len(embs)}/{N} embeddings.")

        for entry, emb in zip(self.entries, embs):
            entry["embedding"] = emb

        # Add an index field to the utterance entries
        for i, entry in enumerate(self.entries):
            entry["index"] = i

        self.depth_score_timeseries = None

    ############################################
    # The following functions have been copied #
    # verbatim from the original repo          #
    # (+ the "self" argument)                  #
    ############################################

    def depth_score(self, timeseries):
        """
        The depth score corresponds to how strongly the cues for a subtopic changed on both sides of a
        given token-sequence gap and is based on the distance from the peaks on both sides of the valleyto that valley.

        returns depth_scores
        """
        depth_scores = []
        for i in range(1, len(timeseries) - 1):
            left, right = i - 1, i + 1
            while left > 0 and timeseries[left - 1] > timeseries[left]:
                left -= 1
            while (
                right < (len(timeseries) - 1)
                and timeseries[right + 1] > timeseries[right]
            ):
                right += 1
            depth_scores.append(
                (timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i])
            )
        return depth_scores

    def smooth(self, timeseries, n, s):
        smoothed_timeseries = timeseries[:]
        for _ in range(n):
            for index in range(len(smoothed_timeseries)):
                neighbours = smoothed_timeseries[
                    max(0, index - s) : min(len(timeseries) - 1, index + s)
                ]
                smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
        return smoothed_timeseries

    def arsort2(self, array1, array2):
        x = np.array(array1)
        y = np.array(array2)

        sorted_idx = x.argsort()[::-1]
        return x[sorted_idx], y[sorted_idx]

    def get_local_maxima(self, array):
        local_maxima_indices = []
        local_maxima_values = []
        for i in range(1, len(array) - 1):
            if array[i - 1] < array[i] and array[i] > array[i + 1]:
                local_maxima_indices.append(i)
                local_maxima_values.append(array[i])
        return local_maxima_indices, local_maxima_values

    ###################################################
    # The following functions have been altered       #
    # to work with OpenAI embeddings and the AMI/ICSI #
    # dataset classes                                 #
    ###################################################

    def compute_window(self, timeseries, start_index, end_index):

        embs = np.array([emb for emb in timeseries[start_index:end_index]])
        # embs.shape = [K,emb_dim]
        pooled_embs = np.max(embs, axis=0)
        # pooled_embs.shape = [emb_dim,]
        return pooled_embs

    def cosine_similarity(self, v1, v2):

        # Standard cosine sim calculation
        # OpenAI embeddings will not have zero norm
        dot = np.sum(v1 * v2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        return dot / (norm1 * norm2)

    def block_comparison_score(self, timeseries, k):

        # Only thing changed is the call to numpy cosine_similarity
        # vs torch
        res = []
        for i in range(k, len(timeseries) - k):
            first_window_features = self.compute_window(timeseries, i - k, i + 1)
            second_window_features = self.compute_window(timeseries, i + 1, i + k + 2)

            sim = self.cosine_similarity(first_window_features, second_window_features)

            res.append(sim)

        return res

    async def get_embeddings(self, chunks, model="text-embedding-ada-002"):

        # Calls the OpenAI embeddings endpoint

        task_params = json.dumps({"model": model, "input": chunks})

        async with aiohttp.ClientSession(
            headers=self.configs["EMBEDDINGS_HEADERS"]
        ) as session:
            async with session.post(
                self.configs["EMBEDDINGS_ENDPOINT"],
                data=task_params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:

                if response.status != 200:
                    raise Exception(
                        "EmbeddingRequestFailed", f"status={response.status}"
                    )

                obj = await response.json()
                return [entry["embedding"] for entry in obj["data"]]

    def depth_score_to_topic_change_indexes(
        self,
        depth_score_timeseries,
        K,
    ):

        # Just return the top K-1 local maxima as the
        # top K-1 best splitting points into K segments

        if depth_score_timeseries == []:
            return []

        local_maxima_indices, local_maxima = self.get_local_maxima(
            depth_score_timeseries
        )
        local_maxima, local_maxima_indices = self.arsort2(
            local_maxima, local_maxima_indices
        )
        local_maxima = local_maxima_indices[: K - 1]
        local_maxima_indices = local_maxima_indices[: K - 1]

        local_maxima, local_maxima_indices = self.arsort2(
            local_maxima, local_maxima_indices
        )
        local_maxima_indices, _ = self.arsort2(local_maxima_indices, local_maxima)

        # sort again for chronological order
        # argsort2 sorts in descending order
        return local_maxima_indices[::-1]

    def segment_meeting(
        self,
        K,
    ):
        entries = self.entries

        if self.depth_score_timeseries is None:
            # timeseries is just "embeddings" now
            embeddings = [entry["embedding"] for entry in entries]

            block_comparison_score_timeseries = self.block_comparison_score(
                embeddings, k=self.configs["SENTENCE_COMPARISON_WINDOW"]
            )
            block_comparison_score_timeseries = self.smooth(
                block_comparison_score_timeseries,
                n=self.configs["SMOOTHING_PASSES"],
                s=self.configs["SMOOTHING_WINDOW"],
            )
            self.depth_score_timeseries = self.depth_score(
                block_comparison_score_timeseries
            )

        transition_points = self.depth_score_to_topic_change_indexes(
            self.depth_score_timeseries,
            K,
        )
        # Add back the window size, since the valid splitting points start "SENTENCE_COMPARISON_WINDOW"
        # into the time series.
        transition_points = [
            pt + self.configs["SENTENCE_COMPARISON_WINDOW"] for pt in transition_points
        ]
        transitions_hat = [0] * len(entries)

        for pt in transition_points:
            transitions_hat[pt] = 1

        return transitions_hat
