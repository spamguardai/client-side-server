import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTORIZER_PATH = os.path.join(BASE_DIR, "params", "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "params", "model.pkl")
IMPORTANT_INDICES_PATH = os.path.join(BASE_DIR, "params", "important_indicies.pkl")
KEY_PATH = os.path.join(BASE_DIR, "Client", "keys")

MAX_FEATURES = 48
LOG_SLOTS = 15