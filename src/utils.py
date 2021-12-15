import os
import string
import torch
import numpy as np
import re
from typing import List, Dict, Tuple
import random
from tqdm import tqdm

def _featurize(toks: List[str], tok2id: Dict[str, int], device: str = "cpu") -> torch.tensor:
    tok_ids = []
    for tok in toks:
        try:
            tok_ids.append(tok2id[tok.lower()])
        except KeyError:
            tok_ids.append(0)
    return torch.Tensor(tok_ids).to(int).to(device)


def load_glove_embeddings(filepath: str, embed_dim: int) -> Tuple[torch.tensor, str]:
    print("Loading Glove...")
    def get_num_lines(f):
      """take a peek through file handle `f` for the total number of lines"""
      num_lines = sum(1 for _ in f)
      f.seek(0)
      return num_lines
    itos=[]
    with open(filepath, "r") as f:
        num_lines = get_num_lines(f)
        vectors = torch.zeros(num_lines, embed_dim, dtype=torch.float32)
        for i, l in enumerate(tqdm(f, total=num_lines)):
            l = l.split(' ')  # using bytes here is tedious but avoids unicode error
            word, vector = l[0], l[1:]
            itos.append(word)
            vectors[i] = torch.tensor([float(x) for x in vector])
    print(f"{len(itos)} words loaded!")
    return (vectors, itos)


def get_pretrained_weights(embed_dim: int, tok2id: Dict[str, int]) -> torch.Tensor:
    glove_path = f"glove/glove.6B.{embed_dim}d.txt"
    if not os.path.exists(glove_path):
        raise ValueError(f"Glove file does not exist: {glove_path}")
    vectors, itos = load_glove_embeddings(filepath=glove_path, embed_dim=embed_dim)
    # So that we can initiate OOV words (relative to glove) with same distribution as others
    glove_mean = vectors.mean()
    glove_std = vectors.std()

    weights = torch.zeros((len(tok2id), embed_dim), dtype=torch.float32)
    found = 0
    for tok, ix in tok2id.items():
        tok = tok.lower()
        if tok in itos:
            weights[ix, :] = vectors[itos.index(tok)]
            found += 1
        else:
            print(f"Word not in glove: {tok}")
            weights[ix, :] = torch.normal(glove_mean, glove_std, size=(embed_dim,))
    print(f"{found} out of {len(tok2id)} words found.")
    return weights


def get_example_script(corpus: "TACLCorpus", script_type: str) -> str:
    """
    Returns an example script from the test set of InScript, with the next token being the mask.
    """
    genre_docs = [doc for doc in corpus.test if doc.name.startswith(script_type)]
    chosen_doc = genre_docs[random.randint(0, len(genre_docs)-1)]
    masked_word_ixs = [ix for ix, word in enumerate(chosen_doc) if word.masked]
    chosen_word_ix = masked_word_ixs[random.randint(0, len(masked_word_ixs)-1)]
    tokens = [chosen_doc[i].text for i in range(chosen_word_ix)]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
