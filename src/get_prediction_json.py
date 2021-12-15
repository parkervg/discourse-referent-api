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
        ref_model: 'EntityNLM', coref_model: 'EntityNLM', text: str, tok2id: Dict[str, int], id2tok: Dict[int, str]
) -> Dict:
    tokenized_text = word_tokenize(text)
    X = _featurize(tokenized_text, tok2id)
    E, R, E_softmax_ents, E_softmax_scores, R_softmaxes = get_coreference_model_output(coref_model, X, id2tok)
    assert len(R) == len(E) == len(tokenized_text) == len(E_softmax_ents) == len(E_softmax_scores) == len(R_softmaxes)
    next_tok, next_E = get_ref_model_output(ref_model, X, E, R, id2tok)
    # Formatting json output
    json_out = {
        "R": R,
        "E": E,
        "E_softmax_ents": E_softmax_ents,
        "E_softmax_scores": E_softmax_scores,
        "R_softmaxes": R_softmaxes,
        "tokenized_text": tokenized_text,
        "next_E": next_E,
        "next_tok": next_tok,
    }
    return json_out


def get_ref_model_output(model: 'EntityNLM', X: torch.Tensor, E, R, id2tok) -> Tuple[str, int]:
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


def get_coreference_model_output(model: 'EntityNLM', X: torch.Tensor, id2tok: Dict[int, str]):
    E: List[int] = []
    R: List[int] = []
    R_softmaxes: List[float] = []
    E_softmax_ents: List[str] = []
    E_softmax_scores: List[float] = []
    ent_to_original_mention: Dict[int, str] = {} # Mapping so we can display intuitive output for the demo
    with torch.no_grad():
        model.eval()
        model.reset_state()
        # e_current = model.get_new_entity()
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
                R_softmaxes.append(round(torch.nn.functional.softmax(R_dist, dim=1)[0][1].item(), 2)) # Grab softmax score for that word being an entity
                if model.entities.shape[0] == 0: # First entity, force to 0
                    e_current = model.get_new_entity()
                    curr_E = torch.tensor(0)
                    E.append(0)
                    E_softmax_ents.append(["New Entity"])
                    E_softmax_scores.append([1])
                else:
                    E_dist = model.get_next_E(h_t, t)
                    curr_E = E_dist.argmax()
                    E.append(curr_E.item())
                    # Get softmaxes for each entity
                    E_softmax_dist = torch.nn.functional.softmax(E_dist, dim=1)[0]
                    E_softmaxes_as_dict = {i: round(E_softmax_dist[i].item(), 2) for i in range(E_softmax_dist.shape[0])}
                    sorted_softmaxes = sorted(E_softmaxes_as_dict.items(), key = lambda x: x[1], reverse=True)[:3]
                    E_softmax_ents.append([ent_to_original_mention[k] + f" ({k})" if k in ent_to_original_mention else "New Entity" for k, v in sorted_softmaxes])
                    E_softmax_scores.append([v for k, v in sorted_softmaxes])
                # register entity
                model.register_predicted_entity(curr_E)
                if curr_E.item() not in ent_to_original_mention:
                    ent_to_original_mention[curr_E.item()] = id2tok[X[t].item()]
            else:  # Not an entity, append -1
                E.append(-1)
                R_softmaxes.append(-1)
                E_softmax_ents.append('')
                E_softmax_scores.append(-1)
            # 5. Update entity state
            if curr_R == 1:
                model.update_entity_embedding(curr_E, h_t, t)
    return E, R, E_softmax_ents, E_softmax_scores, R_softmaxes
