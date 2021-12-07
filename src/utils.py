import torch
import numpy as np
import re
from typing import List, Dict
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()

def _featurize(toks: List[str], tok2id: Dict[str, int], device: str = "cpu") -> torch.tensor:
    tok_ids = []
    for tok in toks:
        try:
            tok_ids.append(tok2id[tok.lower()])
        except KeyError:
            tok_ids.append(0)
    return torch.Tensor(tok_ids).to(int).to(device)


def load_glove_embeddings(filepath) -> Dict[str, np.ndarray]:
    print("Loading Glove...")
    glove_model = {}
    with open(filepath, "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def get_pretrained_weights(glove_path: str, tok2id: Dict[str, int]) -> np.ndarray:
    glove_embeddings = load_glove_embeddings(glove_path)
    glove_as_arr = np.array(list(glove_embeddings.values()))
    # So that we can initiate OOV words (relative to glove) with same distribution as others
    glove_mean = glove_as_arr.mean()
    glove_std = glove_as_arr.std()
    glove_dim = glove_embeddings["the"].shape[0]

    weights = np.zeros((len(tok2id) + 1, glove_dim))
    found = 0
    for tok, ix in tok2id.items():
        tok = tok.lower()
        try:
            weights[ix, :] = glove_embeddings[tok]
            found += 1
        except KeyError:
            # Try to split by hyphen, and average vecs
            subword_vecs = np.zeros(glove_dim)
            for word_ix, subword in enumerate(re.split(r"-", tok)):
                try:
                    subword_vecs += glove_embeddings[subword]
                except KeyError:
                    pass
            if not np.all(subword_vecs == 0):
                found += 1
                subword_vecs = subword_vecs / word_ix + 1
                weights[ix, :] = subword_vecs
            else:
                print(f"Word not in glove: {tok}")
                weights[ix, :] = torch.normal(glove_mean, glove_std, size=(glove_dim,))
    print(f"{found} out of {len(tok2id)} words found.")
    return torch.from_numpy(weights).float()


def get_example_script(corpus: "TACLCorpus", script_type: str) -> str:
    """
    Returns an example script from the test set of InScript, with the next token being the mask.
    """
    genre_docs = [doc for doc in corpus.test if doc.name.startswith(script_type)]
    chosen_doc = genre_docs[random.randint(0, len(genre_docs)-1)]
    masked_word_ixs = [ix for ix, word in enumerate(chosen_doc) if word.masked]
    chosen_word_ix = masked_word_ixs[random.randint(0, len(masked_word_ixs)-1)]
    text = [chosen_doc[i].text for i in range(chosen_word_ix)]
    return detokenizer.detokenize(text)
