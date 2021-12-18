"""
Both directories in data/ were downloaded from https://ashutosh-modi.github.io/datasets/index.html
    - InScript/ was taken from the "InScript (Narrative Texts annotated with Script Information)" link
        - This is the original InScript corpus of 1000 narrative texts covering 10 different scenarios
    - taclData/ was taken from the "Modeling Semantic Expectations" link
        - It includes the discourse referent predictions data

TODO:
    - Because of small corpus size: potentially look into instance based learning
    - Error in taclData?
        - e.g. tree_044: 'tubes that' both marked as entity, but in InScript, it's "plastic tubes" (which seems correct.)
"""
import csv
import re
import xml.etree.ElementTree as ET
import string
from collections import defaultdict, Counter
from itertools import islice

import torch
from attr import attrs, attrib
from typing import List, Dict
from pathlib import Path
import copy
import json
from attr.validators import instance_of
from torch import nn
from utils import _featurize


def vocab_filter_func(word: "TACLWord") -> bool:
    """
    Filter function to pre-process unwanted vocab.
    :param word: TACLWord to filter
    Returns:
        :bool: True or False
    """
    if word.text in string.punctuation:
        return False
    return True


@attrs()
class TACLCorpus:
    train: List["TACLDocument"] = attrib()
    dev: List["TACLDocument"] = attrib()
    test: List["TACLDocument"] = attrib()
    vocab: Counter = attrib(init=False)
    tok2id: Dict[str, int] = attrib(init=False)
    id2tok: Dict[int, str] = attrib(init=False)
    device: str = attrib(default="cpu")
    freq_cutoff: int = attrib(default=0)

    def __attrs_post_init__(self):
        self.create_vocab()
        self.featurize_docs()

    def create_vocab(self):
        self.vocab = Counter()
        for doc in self.train:
            self.vocab.update([word.text for word in doc if vocab_filter_func(word)])
        if self.freq_cutoff:
            self.vocab = Counter(
                {tok: count for tok, count in self.vocab.items() if count > freq_cutoff}
            )
        self.tok2id = {tok: ix for ix, tok in enumerate(self.vocab)}
        self.id2tok = {ix: tok for tok, ix in self.tok2id.items()}
        print(f"Loaded vocab of size {len(self.tok2id)}")

    def featurize_docs(self):
        all_doc_types = [self.train, self.dev, self.test]
        for doc_type in all_doc_types:
            for doc in doc_type:
                doc.featurize(self.tok2id)


@attrs()
class TACLDocument:
    name: str = attrib()
    words: List["TACLWord"] = attrib()
    all_coref_labels: List[
        str
    ] = attrib()  # Used to index and transform E labels to string
    X: torch.Tensor = attrib(init=False)
    device: str = attrib(default="cpu")

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return self.name

    def __getitem__(self, index):
        return self.words[index]

    def featurize(self, tok2id: Dict[str, int]):
        self.X = _featurize([word.text for word in self.words], tok2id, self.device)


@attrs()
class TACLWord:
    id: str = attrib()
    text: str = attrib()
    headVerbID_dependencyRelation: str = attrib()
    verbLemma: str = attrib()
    POS: str = attrib()
    coref_participant_label: str = attrib()
    device: str = attrib(default="cpu")
    masked: bool = attrib(default=False)
    # Variables from entitynlm paper
    R: torch.Tensor = attrib(init=False)
    L: torch.Tensor = attrib(init=False)
    E: torch.Tensor = attrib(default=torch.tensor(0).to(int))

    def __attrs_post_init__(self):
        self.R = (
            torch.tensor(0).to(int).to(self.device)
            if self.coref_participant_label == "O"
            else torch.tensor(1).to(int).to(self.device)
        )
        self.L = torch.tensor(1).to(int).to(self.device)  # Will be modified later

    @id.validator
    def contains_ints(self, _, value):
        """
        Checks to make sure all characters are valid integers, since id follows format
        of integers split by "-"/
        :param value:
        :return:
        """
        if not all([x.isdigit() for s in re.split("-", value) for x in s]):
            raise ValueError(f"Invalid id passed: {value}")

    @classmethod
    def from_row(cls, row: List[str], device: str = "cpu"):
        id = row[0]
        word = row[1]
        headVerbID_dependencyRelation = row[2]
        verbLemma = row[3]
        POS = row[4]
        coref_participant_label = row[5]
        return cls(
            id,
            word,
            headVerbID_dependencyRelation,
            verbLemma,
            POS,
            coref_participant_label,
            device,
        )

    def __str__(self):
        return self.text


def load_tacl_corpus(
    tacl_dir: str,
    masked_refs: Dict[str, Dict[str, str]],
    device="cpu",
    freq_cutoff: int = 0,
) -> TACLCorpus:
    num_files = 0
    tacl_data_path = Path(tacl_dir) / "data"
    train_docs: List[TACLDocument] = []
    test_docs: List[TACLDocument] = []
    dev_docs: List[TACLDocument] = []
    for data_type_subd in tacl_data_path.iterdir():
        if data_type_subd.name.startswith("."):
            continue
        # Decide what type of data it is (train, test, dev)
        if data_type_subd.name == "devFiles":
            docs = dev_docs
        elif data_type_subd.name == "testFiles":
            docs = test_docs
        else:
            docs = train_docs
        for narrative_type_subd in data_type_subd.iterdir():
            if narrative_type_subd.name.startswith("."):
                continue
            for file_path in narrative_type_subd.iterdir():
                participant_labels = []  # Tracks introduction of participant labels
                doc_name = re.sub(r"\.txt$", "", file_path.stem)
                try:
                    masked_ref_ids = masked_refs[doc_name]
                except KeyError:
                    assert narrative_type_subd.name != "testFiles"
                    masked_ref_ids = {}
                num_files += 1
                with open(file_path, "r") as f:
                    prev_participant_label: str = None
                    doc_words: List[TACLWord] = []
                    for idx, row in enumerate(csv.reader(f, delimiter="\t")):
                        if len(row) == 0:  # Start of a new sentence.
                            continue
                        elif len(row) == 1 and row[0].startswith("#id="):
                            sent_id = int(re.sub("^#id=", "", row[0]))
                        elif len(row) == 1 and row[0].startswith("#text="):
                            sent_text = re.sub("^#text=", "", row[0])
                        elif len(row) == 6:
                            tacl_word = TACLWord.from_row(row, device=device)
                            if (
                                tacl_word.id in masked_ref_ids
                            ):  # Determine if this word is masked
                                assert (
                                    masked_ref_ids[tacl_word.id]
                                    == tacl_word.coref_participant_label
                                )
                                tacl_word.masked = True
                            if (
                                tacl_word.R == 1
                            ):  # Check from previous registered entities to see what l should be
                                # Assign l
                                if (
                                    tacl_word.coref_participant_label
                                    == prev_participant_label
                                ):
                                    doc_words[-1].L += 1
                                # Assign e
                                if (
                                    tacl_word.coref_participant_label
                                    not in participant_labels
                                ):
                                    participant_labels.append(
                                        tacl_word.coref_participant_label
                                    )
                                tacl_word.E = (
                                    torch.tensor(
                                        participant_labels.index(
                                            tacl_word.coref_participant_label
                                        )
                                        + 1
                                    )
                                    .to(int)
                                    .to(device)
                                )  # Since E_t starts at 1
                            prev_participant_label = tacl_word.coref_participant_label
                            doc_words.append(tacl_word)
                        else:
                            print(f"Malformed row: {row}")
                docs.append(
                    TACLDocument(
                        name=doc_name,
                        words=doc_words,
                        all_coref_labels=participant_labels,
                        device=device,
                    )
                )
    tacl_corpus = TACLCorpus(
        train=train_docs,
        test=test_docs,
        dev=dev_docs,
        device=device,
        freq_cutoff=freq_cutoff,
    )
    all_doc_count = (
        len(tacl_corpus.train) + len(tacl_corpus.test) + len(tacl_corpus.dev)
    )
    assert (
        all_doc_count == num_files
    ), f"Length mismatch \n expected {num_files}, got {all_doc_count}"
    verify_corpus(tacl_corpus)
    return tacl_corpus


def get_masked_refs(tacl_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Extracts those words in the taclData test set which were masked for Mechanical Turk evaluation.
    :param tacl_dir: The directory containing the "humanPredictions.json" file to read from.
    Returns:
        :masked_refs: Dict of dict, with file names as first key, and word_ids as 2nd keys.
    """
    masked_refs = {}
    tacl_predictions_path = Path(tacl_dir) / "humanPredictions.json"
    with open(tacl_predictions_path, "r") as f:
        predictions_json = json.load(f)
    for doc_id, word_ids in predictions_json.items():
        doc_refs = {}
        for word_id in word_ids:
            assert len(word_ids[word_id].keys()) == 1
            correct_ref = list(word_ids[word_id].keys())[0]
            doc_refs[word_id.replace("_", "-")] = correct_ref
        masked_refs[doc_id] = doc_refs
    return masked_refs


def iter_word_ids(start_id: str, end_id: str):
    start_digits = [int(x) for x in re.split("-", start_id)]
    end_digits = [int(x) for x in re.split("-", end_id)]
    assert start_digits[0] == end_digits[0], "Mismatch in first ids"
    for i in range(start_digits[1], end_digits[1] + 1):  # Inclusive of end_digits
        yield str(start_digits[0]) + "-" + str(i)


def parse_inscript(inscript_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Parses InScript corpus to get participant labels.
    This is neccessary, since there seems to be some mistakes in the taclData format.
    e.g. tree_044, taclData only lists 'tree' as participant from 'Colorado Blue Spruce evergreen tree'
        but, the entire phrase is linked to participant label in InScript.

    BUT, taclData predicts specific entity references, whereas InScript only has broad narrative participant labels.
    :param inscript_dir:
    :return:
    """
    inscript_corpus_path = Path(inscript_dir) / "corpus"
    doc_name_to_ids = {}
    for narrative_type_subd in inscript_corpus_path.iterdir():
        if not narrative_type_subd.is_dir():
            continue
        for file_path in narrative_type_subd.iterdir():
            doc_name = re.sub(r"\.xml$", "", file_path.stem)
            root = ET.parse(file_path).getroot()
            participants = root.find("annotations").find("participants")
            id_to_partipant = {}
            for label in participants:
                data = label.attrib
                if data.get("name") in [
                    "No_label",
                    "NPart",
                ]:  # Evaluation does not contain NParts, so we ignore
                    continue
                if data.get("to"):  # This is a span tag
                    for word_id in iter_word_ids(data.get("from"), data.get("to")):
                        id_to_partipant[word_id] = data.get("name")
                else:
                    id_to_partipant[data.get("from")] = data.get("name")
            doc_name_to_ids[doc_name] = id_to_partipant
    return doc_name_to_ids


def verify_corpus(corpus: TACLCorpus):
    """
    Verifies that when R != 0, E > 0
    """
    for subsection in [corpus.train, corpus.test, corpus.dev]:
        for doc in subsection:
            for word in doc:
                if word.R == 1:
                    assert word.E > 0, "Failure on verify_corpus()"


if __name__ == "__main__":
    tacl_dir = "data/taclData"
    masked_refs = get_masked_refs(tacl_dir)
    corpus = load_tacl_corpus(tacl_dir, masked_refs)

