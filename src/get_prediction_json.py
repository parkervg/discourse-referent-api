import os
import torch
from nltk import word_tokenize
from typing import Dict, Tuple
from utils import _featurize
import numpy as np

DEVICE = torch.device("cpu")

np.random.seed(42)
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

def get_prediction_json(
        ref_model, coref_model, text: str, tok2id: Dict[str, int], id2tok: Dict[int, str]
) -> Dict:
    tokenized_text = word_tokenize(text)
    X = _featurize(tokenized_text, tok2id)
    E, R = get_coreference_model_output(coref_model, X)
    assert len(R) == len(E) == len(tokenized_text)
    next_tok, next_E = get_ref_model_output(ref_model, X, E, R, id2tok)
    # Formatting json output
    json_out = {
        "R": R,
        "E": E,
        "tokenized_text": tokenized_text,
        "next_E": next_E,
        "next_tok": next_tok,
    }
    return json_out


def get_ref_model_output(model, X, E, R, id2tok) -> Tuple[str, int]:
    with torch.no_grad():
        model.eval()
        model.reset_state()
        e_current = model.get_new_entity()
        t = 0  # In case length of text is just one
        h_t, states = model.forward_rnn(X[0], states=None)
        h_t = h_t.squeeze(0)
        # Build up history before predicting next entity
        for t in range(X.size(0) - 1):
            # No need for entity offset, since coref model handled that
            next_E = E[t + 1]
            next_R = R[t + 1]
            if next_R == 1:
                E_dist = model.get_next_E(h_t, t)
                model.register_predicted_entity(next_E)
            h_t, states = model.forward_rnn(X[t + 1], states)
            h_t = h_t.squeeze(0)
            # Update entity state
            if next_R == 1:
                model.update_entity_embedding(next_E, h_t, t)
        # Advance rnn to final token
        h_t, states = model.forward_rnn(X[-1], states=states)
        h_t = h_t.squeeze(0)
        # Now, predict next X and E
        E_dist = model.get_next_E(h_t, t + 1)
        next_E = E_dist.argmax()
        model.register_predicted_entity(next_E)
        e_current = model.get_entity_embedding(next_E)
        X_dist = model.get_next_X(h_t, e_current)
        next_tok = id2tok[X_dist.argmax().item()]
    return (next_tok, next_E.item())


def get_coreference_model_output(model, X):
    E = []
    R = []
    L = []
    #entity_offset = 1
    with torch.no_grad():
        model.eval()
        model.reset_state()
        e_current = model.get_new_entity()
        t = 0  # In case length of text is just one
        # Build up context before predicting on next entity
        states = None
        for t in range(X.size(0)):
            h_t, states = model.forward_rnn(X[t], states)
            h_t = h_t.squeeze(0)
            R_dist = model.get_next_R(h_t)
            curr_R = R_dist.argmax()
            R.append(curr_R.item())
            if curr_R == 1:  # If we predict word is an entity
                E_dist = model.get_next_E(h_t, t)
                curr_E = E_dist.argmax()
                E.append(curr_E.item())
                # register entity
                model.register_predicted_entity(curr_E)
            else:  # Not an entity, append -1
                E.append(-1)
            # 5. Update entity state
            if curr_R == 1:
                model.update_entity_embedding(curr_E, h_t, t)
    return E, R
