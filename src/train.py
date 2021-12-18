import os
import torch
from typing import List, Dict
import time
import sys
import json
import random
from pathlib import Path

from helpers import save_model, load_model
from dataset import load_tacl_corpus
from model import EntityNLM
import utils as utils


def display_exec_time(begin: float, msg_prefix: str = ""):
    """Displays the script's execution time

    Args:
      begin (float): time stamp for beginning of execution
      msg_prefix (str): display message prefix
    """
    exec_time = time.time() - begin

    msg_header = "Execution Time:"
    if msg_prefix:
        msg_header = msg_prefix.rstrip() + " " + msg_header

    if exec_time > 60:
        et_m, et_s = int(exec_time / 60), int(exec_time % 60)
        print("%s %dm %ds" % (msg_header, et_m, et_s))
    else:
        print("%s %.2fs" % (msg_header, exec_time))


def train_nlm(
    model,
    corpus: List["TACLDocument"],
    device,
    num_epochs,
    rz_amplifier=5,
    status_interval=100,
    eval_corpus=None,
    optimizer=None,
    str_pattern="{}_{}_epoch_{}.pkl",
    save_dir="models",
):
    save_dir_path = Path(save_dir)
    entity_offset = 1
    for epoch in range(1, num_epochs + 1):
        X_epoch_loss, E_epoch_loss, R_epoch_loss, L_epoch_loss = 0, 0, 0, 0
        epoch_tokens, epoch_r_div, epoch_l_div, epoch_e_div = 0, 0, 0, 0
        epoch_start = time.time()
        count_E = 0
        count_E_correct = 0
        count_R = 0
        r_true_positive = 0
        r_false_positive = 0
        random.shuffle(corpus)  # Shuffle in place
        for i_doc, doc in enumerate(corpus):
            model.reset_state()
            # initialize e_current
            e_current = model.get_new_entity()

            # forward first token through Embedding and RNN
            # initialize states
            # lstm initializes states with zeros when given None
            h_t, states = model.forward_rnn(doc.X[0], states=None)
            h_t = h_t.squeeze(0)

            # initialize loss tensors
            X_loss = torch.tensor(0, dtype=torch.float, device=device)
            E_loss = torch.tensor(0, dtype=torch.float, device=device)
            R_loss = torch.tensor(0, dtype=torch.float, device=device)
            L_loss = torch.tensor(0, dtype=torch.float, device=device)

            # counters to properly divide losses
            r_div = 0
            l_div = 0
            e_div = 0

            # counter for stats
            doc_r_true_positive = 0
            doc_r_false_positive = 0
            doc_count_R = 0

            # iterate over document
            for t in range(doc.X.size(0) - 1):
                # define target values
                next_X = doc.X[t + 1]  # next Token
                next_E = (
                    doc[t + 1].E - entity_offset
                )  # next Entity, offset to match indices with self.entities
                next_R = doc[t + 1].R  # next R type
                next_L = doc[t + 1].L  # next Length

                # ***START PAPER ALGORITHM***

                # Define current value for L
                current_L = doc[t].L
                if current_L == 1:
                    # 1.
                    # last L equals 1: not continuing entity mention

                    # predict next R
                    R_dist = model.get_next_R(h_t)
                    # create loss for R
                    r_current_loss = (
                        torch.nn.functional.cross_entropy(R_dist, next_R.view(-1))
                        * rz_amplifier
                    )
                    # r_current_loss is used to make amplification of loss possible
                    R_loss += r_current_loss

                    # add division counter for R loss
                    r_div += 1

                    if next_R == 1:
                        # next token is within an entity mention
                        doc_count_R += 1
                        if R_dist.argmax():
                            # both True - correct pred
                            doc_r_true_positive += 1
                            # R_loss += r_current_loss
                        else:
                            # false negative prediction
                            # extra loss to increase recall
                            R_loss += r_current_loss * rz_amplifier
                            pass

                        # select the entity
                        E_dist = model.get_next_E(h_t, t)
                        # count for stats
                        count_E += 1
                        count_E_correct += int(E_dist.argmax() == next_E)
                        # calculate entity loss
                        E_loss += torch.nn.functional.cross_entropy(
                            E_dist, next_E.view(-1)
                        )
                        e_div += 1

                        # register entity
                        model.register_predicted_entity(next_E)

                        # set e_current to entity embedding e_t-1
                        e_current = model.get_entity_embedding(next_E)

                        # predict length of entity and calculate loss
                        L_dist = model.get_next_L(h_t, e_current)
                        L_loss += torch.nn.functional.cross_entropy(
                            L_dist, next_L.view(-1)
                        )
                        l_div += 1

                    else:
                        # only for stats and possibility to amplify loss
                        if R_dist.argmax():
                            # wrong True pred
                            doc_r_false_positive += 1
                            # extra loss
                            # R_loss += r_current_loss * rz_amplifier
                        else:
                            # correct False pred
                            # R_loss += r_current_loss
                            pass
                else:
                    # 2. Otherwise
                    # last L unequal 1, continuing entity mention
                    # set last new_L = last_L - 1
                    # new_R = last_R
                    # new_E = last_E

                    # additional prediction for E to get more training cases
                    # (it also makes stats more comparable to deep-mind paper)
                    E_dist = model.get_next_E(h_t, t)
                    count_E += 1
                    count_E_correct += int(E_dist.argmax() == next_E)
                    E_loss += torch.nn.functional.cross_entropy(E_dist, next_E.view(-1))
                    e_div += 1
                    pass

                # 3. Sample X, get distribution for next Token
                X_dist = model.get_next_X(h_t, e_current)
                X_loss += torch.nn.functional.cross_entropy(X_dist, next_X.view(-1))

                # 4. Advance the RNN on predicted token, here in training next token
                h_t, states = model.forward_rnn(doc.X[t + 1], states)
                h_t = h_t.squeeze(0)
                # new hidden state of next token from here (h_t, previous was h_t-1)

                # 5. Update entity state
                if next_R == 1:
                    model.update_entity_embedding(next_E, h_t, t)
                    # set e_current to embedding e_t
                    e_current = model.get_entity_embedding(next_E)

                # 6. Nothing toDo?

            # ***END PAPER ALGORITHM***

            # calculate stats and divide loss values
            r_true_positive += doc_r_true_positive
            r_false_positive += doc_r_false_positive
            count_R += doc_count_R
            doc_r_prec = doc_r_true_positive / max(
                (doc_r_true_positive + doc_r_false_positive), 1
            )
            doc_r_recall = doc_r_true_positive / max(doc_count_R, 1)
            doc_rf_score = 2 * (
                (doc_r_prec * doc_r_recall) / max(doc_r_prec + doc_r_recall, 1)
            )

            R_loss = R_loss / max(doc_rf_score, 0.35)

            X_epoch_loss += X_loss.item()
            R_epoch_loss += R_loss.item()
            E_epoch_loss += E_loss.item()
            L_epoch_loss += L_loss.item()
            X_loss /= len(doc)
            R_loss /= max(r_div, 1)
            E_loss /= max(e_div, 1)
            L_loss /= max(l_div, 1)

            epoch_tokens += len(doc)
            epoch_r_div += r_div
            epoch_l_div += l_div
            epoch_e_div += e_div

            if optimizer:
                # optimization step
                optimizer.zero_grad()
                loss = X_loss + R_loss + E_loss + L_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            if status_interval and i_doc % status_interval == 0:
                # status output
                r_prec = r_true_positive / max((r_true_positive + r_false_positive), 1)
                r_recall = r_true_positive / max(count_R, 1)
                rf_score = 2 * ((r_prec * r_recall) / max(r_prec + r_recall, 1))
                print(
                    f"Doc {i_doc}/{len(corpus) - 1}: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct / count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}"
                )
                sys.stdout.flush()

        # calulate readable time format
        seconds = round(time.time() - epoch_start)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        x_hour_and_ = f"{h} hours and " * bool(h)
        if optimizer:
            print(f"Epoch {epoch} finished after {x_hour_and_}{m} minutes.")
        else:
            print(f"Evaluation on dev_corpus finished after {x_hour_and_}{m} minutes.")

        # calculate epoch stats: precision, recall and F-Score
        r_prec = r_true_positive / max((r_true_positive + r_false_positive), 1)
        r_recall = r_true_positive / max(count_R, 1)
        rf_score = 2 * ((r_prec * r_recall) / max(r_prec + r_recall, 1))

        print(
            f"Loss: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct / count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}, R_Fscore {rf_score:0.3}"
        )
        print()

        # if in train mode
        if optimizer:
            file_name = save_dir_path / str_pattern.format(
                model.__class__.__name__, model.lstm.hidden_size, epoch
            )
            save_model(model, file_name)
            if eval_corpus:
                # evaluate on evaluation corpus
                with torch.no_grad():
                    print("Evaluating on eval_corpus...")
                    model.eval()
                    train_nlm(
                        model,
                        eval_corpus,
                        num_epochs=1,
                        device=device,
                        status_interval=None,
                        rz_amplifier=rz_amplifier,
                    )
                    model.train()


def train_discriminative_nlm(
    model,
    corpus: List["TACLDocument"],
    device,
    num_epochs,
    rz_amplifier=5,
    status_interval=100,
    eval_corpus=None,
    optimizer=None,
    str_pattern="{}_{}_epoch_{}.pkl",
    save_dir="models",
):
    """
    At each timestep, model condition probability `Q(R_t, E_t, L_t | X_t)`
    """
    save_dir_path = Path(save_dir)
    entity_offset = 1
    for epoch in range(1, num_epochs + 1):
        X_epoch_loss, E_epoch_loss, R_epoch_loss, L_epoch_loss = 0, 0, 0, 0
        epoch_tokens, epoch_r_div, epoch_l_div, epoch_e_div = 0, 0, 0, 0
        epoch_start = time.time()
        count_E = 0
        count_E_correct = 0
        count_R = 0
        r_true_positive = 0
        r_false_positive = 0
        random.shuffle(corpus)  # Shuffle in place
        for (
            i_doc,
            doc,
        ) in enumerate(corpus):
            model.reset_state()
            states = None

            e_current = model.get_new_entity()

            # initialize loss tensors
            E_loss = torch.tensor(0, dtype=torch.float, device=device)
            R_loss = torch.tensor(0, dtype=torch.float, device=device)
            L_loss = torch.tensor(0, dtype=torch.float, device=device)

            # counters to properly devide losses
            r_div = 0
            l_div = 0
            e_div = 0

            # counter for stats
            doc_r_true_positive = 0
            doc_r_false_positive = 0
            doc_count_R = 0

            # iterate over document
            for t in range(doc.X.size(0)):
                # 4. (now 1, in discriminative model) Advance the RNN on predicted token, here in training next token
                h_t, states = model.forward_rnn(doc.X[t], states)
                h_t = h_t.squeeze(0)
                # define targets
                curr_E = doc[t].E - entity_offset
                curr_R = doc[t].R
                curr_L = doc[t].L
                if t == 0 or doc[t - 1].L == 1:  # Not a continuing entity mention
                    R_dist = model.get_next_R(h_t)
                    r_current_loss = (
                        torch.nn.functional.cross_entropy(R_dist, curr_R.view(-1))
                        * rz_amplifier
                    )

                    R_loss += r_current_loss
                    r_div += 1

                    if curr_R == 1:
                        doc_count_R += 1
                        if R_dist.argmax():
                            doc_r_true_positive += 1
                        else:
                            R_loss += r_current_loss * rz_amplifier
                            pass

                        E_dist = model.get_next_E(h_t, t)
                        count_E += 1
                        count_E_correct += int(E_dist.argmax() == curr_E)
                        E_loss += torch.nn.functional.cross_entropy(
                            E_dist, curr_E.view(-1)
                        )
                        e_div += 1

                        # register entity
                        model.register_predicted_entity(curr_E)

                        e_current = model.get_entity_embedding(curr_E)

                        L_dist = model.get_next_L(h_t, e_current)
                        L_loss += torch.nn.functional.cross_entropy(
                            L_dist, curr_L.view(-1)
                        )
                        l_div += 1
                    else:
                        if R_dist.argmax():
                            doc_r_false_positive += 1
                else:  # A continuing entity mention
                    E_dist = model.get_next_E(h_t, t)
                    count_E += 1
                    count_E_correct += int(E_dist.argmax() == curr_E)
                    E_loss += torch.nn.functional.cross_entropy(E_dist, curr_E.view(-1))
                    e_div += 1
                    pass
                if curr_R == 1:
                    model.update_entity_embedding(curr_E, h_t, t)
                    e_current = model.get_entity_embedding(curr_E)
            # calculate stats and divide loss values
            r_true_positive += doc_r_true_positive
            r_false_positive += doc_r_false_positive
            count_R += doc_count_R
            doc_r_prec = doc_r_true_positive / max(
                (doc_r_true_positive + doc_r_false_positive), 1
            )
            doc_r_recall = doc_r_true_positive / max(doc_count_R, 1)
            doc_rf_score = 2 * (
                (doc_r_prec * doc_r_recall) / max(doc_r_prec + doc_r_recall, 1)
            )

            R_loss = R_loss / max(doc_rf_score, 0.35)

            R_epoch_loss += R_loss.item()
            E_epoch_loss += E_loss.item()
            L_epoch_loss += L_loss.item()

            R_loss /= max(r_div, 1)
            E_loss /= max(e_div, 1)
            L_loss /= max(l_div, 1)

            epoch_tokens += len(doc)
            epoch_r_div += r_div
            epoch_l_div += l_div
            epoch_e_div += e_div

            if optimizer:
                # optimization step
                optimizer.zero_grad()
                loss = R_loss + E_loss + L_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            if status_interval and i_doc % status_interval == 0:
                # status output
                r_prec = r_true_positive / max((r_true_positive + r_false_positive), 1)
                r_recall = r_true_positive / max(count_R, 1)
                rf_score = 2 * ((r_prec * r_recall) / max(r_prec + r_recall, 1))
                print(
                    f"Doc {i_doc}/{len(corpus) - 1}: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct / count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}"
                )
                sys.stdout.flush()

        # calulate readable time format
        seconds = round(time.time() - epoch_start)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        x_hour_and_ = f"{h} hours and " * bool(h)
        if optimizer:
            print(f"Epoch {epoch} finished after {x_hour_and_}{m} minutes.")
        else:
            print(f"Evaluation on dev_corpus finished after {x_hour_and_}{m} minutes.")

        # calculate epoch stats: precision, recall and F-Score
        r_prec = r_true_positive / max((r_true_positive + r_false_positive), 1)
        r_recall = r_true_positive / max(count_R, 1)
        rf_score = 2 * ((r_prec * r_recall) / max(r_prec + r_recall, 1))

        print(
            f"Loss: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct / count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}, R_Fscore {rf_score:0.3}"
        )
        print()

        # if in train mode
        if optimizer:
            file_name = save_dir_path / str_pattern.format(
                model.__class__.__name__, model.lstm.hidden_size, epoch
            )
            save_model(model, file_name)
            if eval_corpus:
                # evaluate on evaluation corpus
                with torch.no_grad():
                    print("Evaluating on eval_corpus...")
                    model.eval()
                    train_nlm(
                        model,
                        eval_corpus,
                        num_epochs=1,
                        device=device,
                        status_interval=None,
                        rz_amplifier=rz_amplifier,
                    )
                    model.train()
        else:  # In eval mode
            return count_E_correct / count_E


def train(discriminative: bool = False, **kwargs):
    corpus = load_tacl_corpus(
        kwargs.get("tacl_dir"),
        {},
        device=kwargs.get("device"),
        freq_cutoff=kwargs.get("freq_cutoff"),
    )
    if kwargs.get("use_pretrained", False):
        kwargs["pretrained_weights"] = utils.get_pretrained_weights(
            kwargs.get("embedding_size"), corpus.tok2id
        )
    model = EntityNLM(max(corpus.id2tok) + 1, **kwargs).to(kwargs.get("device"))
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr"))
    # create directory, and save params
    save_dir_path = Path(kwargs.get("save_dir"))
    save_dir_path.mkdir(parents=True, exist_ok=True)
    with open(save_dir_path / "params.json", "w") as f:
        json.dump(
            {
                k: v
                for k, v in kwargs.items()
                if k not in ["pretrained_weights", "device"]
            },
            f,
        )
    if discriminative:
        print("Training discriminative model...")
        train_discriminative_nlm(
            model=model,
            corpus=corpus.train,
            num_epochs=kwargs.get("num_epochs"),
            device=kwargs.get("device"),
            optimizer=optimizer,
            eval_corpus=corpus.dev,
            save_dir=kwargs.get("save_dir"),
        )
    else:
        print("Training original model...")
        train_nlm(
            model=model,
            corpus=corpus.train,
            num_epochs=kwargs.get("num_epochs"),
            device=kwargs.get("device"),
            optimizer=optimizer,
            eval_corpus=corpus.dev,
            save_dir=kwargs.get("save_dir"),
        )


if __name__ == "__main__":
    """
    TODO:
        - With best hyperparameters, train 2 seperate EntityNLM models:
            - One with the generative story we've been using, to predict next entity
            - One with the 'discriminative variant' settings described in the 'proposal distribution' for coreference resolution
    """
    kwargs = {
        "dropout": 0.4,
        "embedding_size": 100,
        "hidden_size": 128,
        "num_epochs": 50,
        "lr": 0.001,
        "num_layers": 1,
        "save_dir": f"models/discriminative/fixed_glove_embed_size100_hiddensize128_lr0.001",
        "device": torch.device("cpu"),
        "use_pretrained": True,
        "tacl_dir": "data/taclData",
        "freq_cutoff": 0,
    }
    train(discriminative=True, **kwargs)

    # # Baseline no glove
    # for hidden_size in [128, 64]:
    #     for embedding_size in [50, 100]:
    #         for lr in [0.01, 0.001]:
    #             kwargs = {"dropout": 0.3,
    #                       "embedding_size": embedding_size,
    #                       "hidden_size": hidden_size,
    #                       "num_epochs": 50,
    #                       "lr": lr,
    #                       "save_dir": f'models/embed_size{embedding_size}_hiddensize{hidden_size}_lr{lr}',
    #                       "device": torch.device("cpu"),
    #                       "use_pretrained": False,
    #                       "tacl_dir": "data/taclData",
    #                       "freq_cutoff": 0}
    #             train(**kwargs)
    # kwargs = {"dropout": 0.4,
    #           "embedding_size": 100,
    #           "hidden_size": 128,
    #           "num_epochs": 50,
    #           "lr": 0.001,
    #           "save_dir": 'models/test',
    #           "device": torch.device("cpu"),
    #           "use_pretrained": False,
    #           "tacl_dir": "data/taclData"}
    # train(**kwargs)

    """
    Notes:
        - Predicting next_R == 1 when preceded by 'with'
        - 
    """
