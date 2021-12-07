import torch
from typing import List, Dict, Iterable
from pathlib import Path
import time
import json
import re
import argparse

from dataset import load_tacl_corpus, get_masked_refs
from model import EntityNLM
from helpers import save_model, load_model, load_state
import utils as utils
from train import train_discriminative_nlm, train_nlm # Used to evaluate here


def evaluate(
    model,
    corpus: List["TACLDocument"], # test corpus
    device: str,
    id2tok: Dict[int, str],
    verbose: bool = False,
):
    """
    Iterate through test corpus, evaluate on masked words.
    See if E prediction is correct when it is given that R_t = 1.
    Map all entities not previously mentioned to "new entity" (E is new)
    """
    with torch.no_grad():
        model.eval()
        entity_offset = 1
        num_correct = 0
        total_possible = 0
        total_new_entities = 0
        new_ent_true_positive = 0
        new_ent_false_positive = 0
        for i_doc, doc in enumerate(corpus):
            model.reset_state()
            # initialize e_current
            e_current = model.get_new_entity()
            # forward first token through Embedding and RNN
            # initialize states
            # lstm initializes states with zeros when given None
            h_t, states = model.forward_rnn(doc.X[0], states=None)
            h_t = h_t.squeeze(0)

            for t in range(doc.X.size(0) - 1):
                # define targets
                next_X = doc.X[t + 1]
                next_E = doc[t + 1].E - entity_offset
                next_R = doc[t + 1].R
                next_L = doc[t + 1].L

                # ***START PAPER ALGORITHM***
                current_L = doc[t].L
                if current_L == 1:  # Not a continuing entity
                    if next_R == 1:  # next token is within an entity mention
                        E_dist = model.get_next_E(h_t, t)
                        if doc[t + 1].masked:
                            total_possible += 1
                            # Select the entity (Equation 4)
                            pred = E_dist.argmax().item()
                            actual = next_E.item()
                            tok = id2tok[next_X.item()] if next_X.item() != 0 else 0
                            if pred == actual:
                                num_correct += 1
                                if verbose:
                                    print("Correct!")
                            # Check to see if actual/pred created new entities
                            new_entity_pred = pred == model.entities.size(0) - 1
                            new_entity_actual = actual == model.entities.size(0) - 1
                            if new_entity_actual:  # The actual entity is new
                                total_new_entities += 1
                                new_ent_true_positive += int(
                                    new_entity_actual == new_entity_pred
                                )
                            else:  # The actual entity is not new
                                new_ent_false_positive += int(new_entity_pred)
                            if verbose:
                                print(tok)
                                print(actual)
                                print(f"Predicted new: {new_entity_pred}")
                                print(f"Actual new: {new_entity_actual}")
                                try:
                                    print(
                                        f"Got {doc.all_coref_labels[pred]}, expected {doc.all_coref_labels[actual]}"
                                    )
                                except IndexError:
                                    print("error")
                                print()
                                assert (
                                    doc.all_coref_labels[actual]
                                    == doc[t + 1].coref_participant_label
                                )
                        # register entity
                        model.register_predicted_entity(next_E)
                        # Set e_current to entity embedding e_t-1
                        e_current = model.get_entity_embedding(next_E)
                    else:
                        pass
                else:  # A continuing entity mention
                    pass
                h_t, states = model.forward_rnn(doc.X[t + 1], states)
                h_t = h_t.squeeze(0)
                # Update entity state
                if next_R == 1:
                    model.update_entity_embedding(next_E, h_t, t)
                    # Set e_current to embedding e_t
                    e_current = model.get_entity_embedding(next_E)
        total_accuracy = num_correct / total_possible
        new_ent_precision = new_ent_true_positive / max(
            (new_ent_true_positive + new_ent_false_positive), 1
        )
        new_ent_recall = new_ent_true_positive / max(total_new_entities, 1)
        print("Total accuracy:")
        print(total_accuracy)
        print("New entity precision:")
        print(new_ent_precision)
        print("New entity recall:")
        print(new_ent_recall)
        return total_accuracy, new_ent_precision, new_ent_recall

def get_result(corpus, discriminative: bool, device, model_dir: str, min_epoch: int = 0):
    """
    Gets best result from a single model_dir, containing epoch pkl files.

    Saves `evaluation.json` file in model_dir
    """
    model_dir = Path(model_dir)
    base_model = load_model(corpus, device, model_dir)
    best_acc, best_precision, best_recall, best_epoch = 0, 0, 0, 0
    for model_load_path in model_dir.iterdir():
        if model_load_path.name.endswith(".json"):
            continue
        epoch = int(re.search(r"\d+$", model_load_path.stem).group())
        if min_epoch:  # Anything not 0
            if epoch < min_epoch:
                continue
        model = load_state(model_load_path, base_model)
        if discriminative:
            accuracy = train_discriminative_nlm(model=model,
                                     corpus=corpus.test,
                                     device=device,
                                     num_epochs=1,
                                     status_interval = None,
                                     )
            precision, recall = 0, 0
        else:
            accuracy, precision, recall = evaluate(
                model=model,
                corpus=corpus.test,
                device=device,
                id2tok=corpus.id2tok,
                verbose=False,
            )
        if accuracy > best_acc:
            print("***************************************************")
            print("Updating best accuracy!")
            print(accuracy)
            print("***************************************************")
            best_acc = accuracy
            best_precision = precision
            best_recall = recall
            best_epoch = epoch
            print(f"Best epoch: {best_epoch}")
            print(f"Best accuracy: {best_acc}")
            print(f"Best precision: {best_precision}")
            print(f"Best recall: {best_recall}")
    with open(model_dir / "evaluation.json", "w") as f:
        json.dump(
            {
                "accuracy": best_acc,
                "precision": best_precision,
                "recall": best_recall,
                "epoch": best_epoch,
            },
            f,
        )


def get_all_results(
    corpus, discriminative: bool, device, models_dir: str, output_json_path: str = None, min_epoch: int = 0
):
    """

    Gets all results for models in a specified directory.
    """
    results = {}
    for model_subd in Path(models_dir).iterdir():
        if not model_subd.is_dir():
            continue
        base_model = load_model(corpus, device, model_subd)
        if not base_model:  # Couldn't load from params.json
            continue
        if (model_subd / "evaluation.json").is_file():
            print(f"evaluation.json file already exists for {model_subd.name}")
            continue
        best_acc, best_precision, best_recall, best_epoch = 0, 0, 0, 0
        for model_load_path in model_subd.iterdir():
            if model_load_path.name.endswith(".json"):
                continue
            epoch = int(re.search(r"\d+$", model_load_path.stem).group())
            if min_epoch:  # Anything not 0
                if epoch < min_epoch:
                    continue
            model = load_state(model_load_path, base_model)
            with open(Path(model_subd) / "params.json", "rb") as f:
                params = json.load(f)
            if discriminative:
                accuracy = train_discriminative_nlm(model=model,
                                         corpus=corpus.test,
                                         device=device,
                                         num_epochs=1,
                                         status_interval=None,
                                         )
                precision, recall = 0, 0
            else:
                accuracy, precision, recall = evaluate(
                    model=model,
                    corpus=corpus.test,
                    device=device,
                    id2tok=corpus.id2tok,
                    verbose=False,
                )
            if accuracy > best_acc:
                print("***************************************************")
                print("Updating best accuracy!")
                print(accuracy)
                print("***************************************************")
                best_acc = accuracy
                best_precision = precision
                best_recall = recall
                best_epoch = epoch
                results[model_subd.name] = {
                    "params": params,
                    "accuracy": best_acc,
                    "precision": best_precision,
                    "recall": best_recall,
                    "best_epoch": epoch,
                }
        with open(model_subd / "evaluation.json", "w") as f:
            json.dump(
                {
                    "accuracy": best_acc,
                    "precision": best_precision,
                    "recall": best_recall,
                    "epoch": best_epoch,
                },
                f,
            )
    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(results, f)
    return results


if __name__ == "__main__":
    tacl_dir = "data/taclData"
    device = torch.device("cpu")
    masked_refs = get_masked_refs(tacl_dir)
    corpus = load_tacl_corpus(tacl_dir, masked_refs, device=device)
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", help="Directory containing model pkl files")
    parser.add_argument("output_json_path", nargs='?', default=None, help="Where to save the output results.json")
    parser.add_argument('-d', action='store_true')
    args = parser.parse_args()
    if (Path(args.models_dir) / "params.json").is_file():
        get_result(corpus=corpus,
                   discriminative=args.d,
                   model_dir=args.models_dir,
                   device=device)
    else:
        get_all_results(
            corpus=corpus,
            discriminative=args.d,
            models_dir=args.models_dir,
            output_json_path=args.output_json_path,
            device=device,
            min_epoch=15,
        )
