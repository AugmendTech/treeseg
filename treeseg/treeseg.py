import os

import structlog

logger = structlog.get_logger()

import json
import asyncio
import aiohttp
import heapq
import numpy as np

import matplotlib.pyplot as plt


class SegNode:
    def __init__(self, identifier, entries, configs):
        self.configs = configs
        self.MIN_SEGMENT_SIZE = configs["MIN_SEGMENT_SIZE"]
        self.LAMBDA_BALANCE = configs["LAMBDA_BALANCE"]
        self.entries = entries
        self.segment = [entry["index"] for entry in entries]
        self.embs = [entry["embedding"] for entry in entries]
        self.embs = np.array(self.embs)
        self.mu = np.mean(self.embs, axis=0)

        self.left = None
        self.right = None

        self.identifier = identifier

        self.is_leaf = len(entries) < 2 * self.MIN_SEGMENT_SIZE

        if self.is_leaf:
            return

        self.compute_likelihood()
        self.optimize_split()

    def squared_error(self, X, mu):
        return np.sum(np.sum(np.square(X - mu), axis=-1))

    def compute_likelihood(self):
        self.mu = np.mean(self.embs, axis=0, keepdims=True)
        self.negative_log_likelihood = self.squared_error(self.embs, self.mu)
        self.negative_log_likelihood += self.LAMBDA_BALANCE * np.square(
            len(self.entries)
        )

    def optimize_split(self):
        N = len(self.entries)
        index = list(np.arange(N))

        min_loss = float("inf")

        losses = []

        S0 = None
        S1 = None

        for n in range(self.MIN_SEGMENT_SIZE - 1, N - self.MIN_SEGMENT_SIZE):

            if S0 is None:

                idx0 = index[: n + 1]
                idx1 = index[n + 1 :]

                X0 = self.embs[idx0]
                X1 = self.embs[idx1]

                S0 = np.sum(X0, axis=0)
                SS0 = np.sum(np.square(X0), axis=0)

                S1 = np.sum(X1, axis=0)
                SS1 = np.sum(np.square(X1), axis=0)

                M0 = len(idx0)
                M1 = len(idx1)
            else:

                M0 += 1
                M1 -= 1

                v = self.embs[n]

                S0 += v
                S1 -= v

                SS0 += np.square(v)
                SS1 -= np.square(v)

            assert M0 + M1 == N

            mu0 = S0 / M0
            mu1 = S1 / M1

            loss = np.sum(SS0 - np.square(mu0) * M0)
            loss += np.sum(SS1 - np.square(mu1) * M1)

            balance_penalty = self.LAMBDA_BALANCE * np.square(M0)
            balance_penalty += self.LAMBDA_BALANCE * np.square(M1)
            loss += balance_penalty

            losses.append(loss)

            if loss < min_loss:
                min_loss = loss
                self.split_negative_log_likelihood = loss
                self.split_entries = [self.entries[: n + 1], self.entries[n + 1 :]]

        # if self.identifier=="*":
        #    plt.plot(losses)
        #    plt.show()

    def split_loss_delta(self):
        return self.split_negative_log_likelihood - self.negative_log_likelihood

    def split(self):
        left_entries, right_entries = self.split_entries

        self.left = SegNode(
            identifier=self.identifier + "L",
            entries=left_entries,
            configs=self.configs,
        )
        self.right = SegNode(
            identifier=self.identifier + "R",
            entries=right_entries,
            configs=self.configs,
        )
        return self.left, self.right


class TreeSeg:

    def __init__(self, configs, entries):
        self.configs = configs

        self.entries = entries

        self.extract_blocks()

        embedded = False

        while not embedded:

            try:
                self.embed_blocks()
                embedded = True
            except Exception as e:
                logger.warning(f"Embedding failed ({str(e)}): Retrying")
                continue

    def discover_leaves(self, cond_func=lambda x: True, K=float("inf")):

        leaves = []
        stack = [self]

        while stack:
            node = stack.pop(0)

            if node.left is None and node.right is None:

                if cond_func(node):
                    leaves.append(node)
                    print(node.identifier, node.segment)

                    if len(leaves) == K:
                        return leaves
                continue

            if node.right:
                stack = [node.right] + stack
            if node.left:
                stack = [node.left] + stack

        return leaves

    async def get_embeddings(self, chunks, model="text-embedding-ada-002"):

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
                    print(await response.json())
                    raise Exception(
                        "EmbeddingRequestFailed", f"status={response.status}"
                    )

                obj = await response.json()
                return [entry["embedding"] for entry in obj["data"]]

    def embed_blocks(self):

        logger.info(f"Submitting blocks to OpenAI")

        loop = asyncio.get_event_loop()

        N = len(self.blocks)
        BATCH_SIZE = 500
        num_batches = np.int32(np.ceil(N / BATCH_SIZE))

        logger.info(f"Extracting {N} embeddings in {num_batches} batches.")

        batches = []

        for batch_i in range(num_batches):
            batch = self.blocks[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
            batches.append(batch)

        embs = []
        result = loop.run_until_complete(
            asyncio.gather(
                *[
                    self.get_embeddings([block["convo"] for block in batch])
                    for batch in batches
                ]
            )
        )

        for batch_embs in result:
            embs.extend(batch_embs)
        logger.info(f"I have collected {len(embs)}/{N} embeddings.")

        for block, emb in zip(self.blocks, embs):
            block["embedding"] = emb

    # Can be done more efficiently
    def extract_blocks(self):

        entries = self.entries
        N = len(entries)
        blocks = []

        width = self.configs["UTTERANCE_EXPANSION_WIDTH"]

        for i, entry in enumerate(entries):

            convo = []

            for idx in range(max(0, i - width), i + 1):
                convo.append(entries[idx]["composite"])

            block = {"convo": "\n".join(convo), "index": i}
            block.update(entry)
            blocks.append(block)

        for i, block in enumerate(blocks):
            block["index"] = i

        self.blocks = blocks

    def segment_meeting(self, K):

        self.root = SegNode(
            identifier="*", entries=self.blocks, configs=self.configs
        )
        root = self.root

        if root.is_leaf:
            logger.info(f"Cannot split root further. Nothing to do here.")
            return

        leaves = []
        boundary = []
        heapq.heappush(boundary, (root.split_loss_delta(), root))

        loss = root.negative_log_likelihood

        while boundary:

            if len(boundary) + len(leaves) == K:
                logger.warning(f"Reached maximum of {K} segments.")
                while boundary:
                    loss_delta, node = heapq.heappop(boundary)
                    node.is_leaf = True
                    heapq.heappush(leaves, (node.segment[0], node))
                break

            loss_delta, node = heapq.heappop(boundary)
            loss += loss_delta

            perc = -loss_delta / loss
            logger.info(f"Loss reduction: {loss-loss_delta}=>{loss} | {100*perc}%")
            left, right = node.split()

            if left.is_leaf:
                heapq.heappush(leaves, (left.segment[0], left))
            else:
                heapq.heappush(boundary, (left.split_loss_delta(), left))

            if right.is_leaf:
                heapq.heappush(leaves, (right.segment[0], right))
            else:
                heapq.heappush(boundary, (right.split_loss_delta(), right))

        sorted_leaves = []
        while leaves:
            sorted_leaves.append(heapq.heappop(leaves)[1])

        self.leaves = sorted_leaves

        transitions_hat = [
            0,
        ] * len(self.leaves[0].segment)

        for leaf in self.leaves[1:]:

            segment_transition = [
                0,
            ] * len(leaf.segment)
            segment_transition[0] = 1
            transitions_hat.extend(segment_transition)

        self.transitions_hat = transitions_hat
        return transitions_hat
