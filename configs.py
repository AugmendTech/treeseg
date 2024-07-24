
import os

EMBEDDINGS_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
    }
EMBEDDINGS_ENDPOINT = "https://api.openai.com/v1/embeddings"


bertseg_configs = {

    "ami": {
        "SENTENCE_COMPARISON_WINDOW": 15,
        "SMOOTHING_PASSES": 2,
        "SMOOTHING_WINDOW": 5,
        "EMBEDDINGS_HEADERS": EMBEDDINGS_HEADERS,
        "EMBEDDINGS_ENDPOINT": EMBEDDINGS_ENDPOINT,
    },
    "icsi": {
        "SENTENCE_COMPARISON_WINDOW": 15,
        "SMOOTHING_PASSES": 2,
        "SMOOTHING_WINDOW": 5,
        "EMBEDDINGS_HEADERS": EMBEDDINGS_HEADERS,
        "EMBEDDINGS_ENDPOINT": EMBEDDINGS_ENDPOINT,
    },
    "augmend": {
        "SENTENCE_COMPARISON_WINDOW": 15,
        "SMOOTHING_PASSES": 2,
        "SMOOTHING_WINDOW": 5,
        "EMBEDDINGS_HEADERS": EMBEDDINGS_HEADERS,
        "EMBEDDINGS_ENDPOINT": EMBEDDINGS_ENDPOINT,
    },
}


treeseg_configs = {

    "ami": {
        "MIN_SEGMENT_SIZE": 5,
        "LAMBDA_BALANCE": 0,
        "UTTERANCE_EXPANSION_WIDTH": 4,
        "EMBEDDINGS_HEADERS": EMBEDDINGS_HEADERS,
        "EMBEDDINGS_ENDPOINT": EMBEDDINGS_ENDPOINT,
    },
    "icsi": {
        "MIN_SEGMENT_SIZE": 5,
        "LAMBDA_BALANCE": 0,
        "UTTERANCE_EXPANSION_WIDTH": 4,
        "EMBEDDINGS_HEADERS": EMBEDDINGS_HEADERS,
        "EMBEDDINGS_ENDPOINT": EMBEDDINGS_ENDPOINT,
    },
    "augmend": {
        "MIN_SEGMENT_SIZE": 5,
        "LAMBDA_BALANCE": 0,
        "UTTERANCE_EXPANSION_WIDTH": 2,
        "EMBEDDINGS_HEADERS": EMBEDDINGS_HEADERS,
        "EMBEDDINGS_ENDPOINT": EMBEDDINGS_ENDPOINT,
    },
}