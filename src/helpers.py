import torch
import json
from pathlib import Path
from typing import Union

from model import EntityNLM
import utils as utils


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to: '{path}'")


def load_model(corpus, device, model_load_dir: Union[Path, str]):
    """
    Loads specific model from epoch.pkl file.
    """
    device = torch.device(device)
    model_load_dir = Path(model_load_dir)
    params_path = model_load_dir / "params.json"
    if params_path.is_file():
        try:
            with open(params_path, "rb") as f:
                params = json.load(f)
        except json.decoder.JSONDecodeError:
            print("Error reading json")
            return False
    else:
        print(f"Can't find params.json for {model_load_dir.name}")
        return False
    if params["use_pretrained"]:
        glove_path = f"glove/glove.6B.{params['embedding_size']}d.txt"
        pretrained_weights = utils.get_pretrained_weights(glove_path, corpus.tok2id)
    else:
        pretrained_weights = None
    model = EntityNLM(
        max(corpus.id2tok) + 1,
        device=device,
        dropout=params["dropout"],
        embedding_size=params["embedding_size"],
        hidden_size=params["hidden_size"],
        pretrained_weights=pretrained_weights,
    ).to(device)
    return model


def load_state(path: Union[Path, str], model=None):
    new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(
        new_state_dict, strict=False
    )  # strict = False ignores "missing keys" error
    print(f"Model loaded from: '{path}'")
    return model


def load_best_state(model_dir: Union[Path, str], model):
    """
    Given a model_dir, reads from evaluation.json file and loads best performing model.
    """
    model_dir = Path(model_dir)
    evaluation_path = model_dir / "evaluation.json"
    if evaluation_path.is_file():
        with open(evaluation_path, "r") as f:
            evaluation_data = json.load(f)
        best_epoch = evaluation_data["epoch"]
        print(f"Loading epoch file with {evaluation_data['accuracy']} accuracy...")
    else:
        raise ValueError("evaluation.json file not found.")
    for epoch_file in model_dir.iterdir():
        if epoch_file.name.endswith(f"{best_epoch}.pkl"):
            return load_state(epoch_file, model)
    raise ValueError(f"epoch{best_epoch} file not found....")
