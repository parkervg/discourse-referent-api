import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np
import time, sys

np.random.seed(42)
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

class EntityNLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        device,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.5,
        pretrained_weights=None,
        **kwargs,
    ):
        super(EntityNLM, self).__init__()
        print(f"Embedding size: {embedding_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Dropout: {dropout}")
        self.device = device
        # assert hidden_size == entity_size, "hidden_size should be equal to entity_size"
        # embedding matrix for input tokens
        if pretrained_weights != None:
            print("Using pretrained weights...")
            self.embedding_matrix = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=True, padding_idx=0
            )
        else:
            print("Not using pretrained weights")
            self.embedding_matrix = nn.Embedding(vocab_size, embedding_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers
        )

        # Final layer, outputs probability distribution over vocab
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # r is the parameterized embedding associated with r, which paves the way for exploring entity type representations in future work
        self.r_embeddings = torch.nn.Parameter(
            torch.FloatTensor(2, hidden_size), requires_grad=True
        ).to(self.device)

        # W_r is parameter matrix for the bilinear score for h_t−1 and r.
        self.W_r = nn.Bilinear(hidden_size, hidden_size, 1)

        # W_length is the weight matrix for length prediction
        self.W_length = nn.Linear(2 * hidden_size, 25)

        # W_entity is the weight matrix for predicting entities using their continuous representations
        self.W_entity = nn.Bilinear(hidden_size, hidden_size, 1)

        # For distance feature
        self.w_dist = nn.Linear(1, 1)

        # Used in equation 8 to create interpolation sigma_t
        # bilinear: nonlinear function between two matrices
        # equal to torch.mm(x1, torch.mm(A, x2)) + b
        self.W_delta = nn.Bilinear(hidden_size, hidden_size, 1)

        # W_e is a transformation matrix to adjust the dimensionality of e_current
        self.W_e = nn.Linear(hidden_size, hidden_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # Set of entities E_t
        self.entities = torch.tensor([], dtype=torch.float, device=self.device)
        # distance features for entities
        self.dist_features = torch.tensor([], dtype=torch.float, device=self.device)
        self.max_entity_index = 0

        self.init_weights()

    def init_weights(self, init_range=(-0.01, 0.01)):
        if not init_range:
            return
        for param in self.parameters():
            if param.dim() > 1:
                init.xavier_uniform(param, gain=np.sqrt(2))
        self.W_entity.weight.data.uniform_(*init_range)
        self.W_entity.bias.data.fill_(0)

        self.w_dist.weight.data.uniform_(*init_range)
        self.w_dist.bias.data.fill_(0)

        self.W_e.weight.data.uniform_(*init_range)
        self.W_e.bias.data.fill_(0)

    def forward_rnn(self, x, states):
        # Input: LongTensor with token indices
        # Creates embedding vectors for input and feeds trough lstm
        x = self.embedding_matrix(x.view(1, -1))
        return self.lstm(x, states)

    def get_new_entity(self):
        # Creates a new entity, returns reference
        self.add_new_entity()
        return self.get_entity_embedding(-1)  # The one we just added

    def add_new_entity(self, t=0.0):
        # Append new embedding u to entity matrix
        self.entities = torch.cat(
            (self.entities, self.initialize_entity_embedding()), dim=0
        )
        # Create distance features
        self.dist_features = torch.cat(
            (
                self.dist_features,
                torch.tensor([[t]], dtype=torch.float, device=self.device),
            ),
            dim=0,
        )

    def get_entity_embedding(self, entity_index: int):
        # returns the entity embedding to the respective index
        return self.entities[entity_index].unsqueeze(0)

    def get_dist_feat(self, t):
        # subtract current time step from dist feature vector
        return self.dist_features - t

    def initialize_entity_embedding(self, sigma=0.01):
        """
        Equation 7
        Initializes a new entity embedding
        :param sigma:
        :return:
        """
        # Get R_t = 1
        # Expected to encode some generic information about entities.
        r1 = self.r_embeddings[1]
        # Normal init based on r1 with sigma
        u = r1 + sigma * torch.normal(
            torch.zeros_like(r1, device=self.device),
            torch.ones_like(r1, device=self.device).view(1, -1),
        )
        # Normalize
        u = u / torch.norm(u, p=2)
        return u

    def update_entity_embedding(self, entity_index, h_t, t):
        """
        Equation 8

        New embedding is a combination of old embedding and current LSTM hidden state (h_t)
        :return:
        """
        # get entity
        entity_embedding = self.get_entity_embedding(entity_index)

        # Calculate interpolation
        delta = torch.sigmoid(self.W_delta(entity_embedding, h_t)).view(-1)

        # Update entity embedding based with h_t using sigma_t
        u = delta * entity_embedding + (1 - delta) * h_t

        # index_copy: takes dim, index, tensor args
        # Update entities in set E_t
        self.entities = self.entities.index_copy(
            0, torch.tensor(entity_index), (u / torch.norm(u))
        )

        # updating entities in dist_features
        self.dist_features = self.dist_features.index_copy(
            0,
            torch.tensor(entity_index),
            torch.tensor([[t]], dtype=torch.float, device=self.device),
        )

    def get_next_R(self, h_t):
        """
        Equation 3
        :param h_t:
        :return:
        """
        # Predict distribution for next R using bilinear layer
        pred_r = self.W_r(
            self.dropout(self.r_embeddings),
            self.dropout(h_t.expand_as(self.r_embeddings)),
        ).view(1, -1)
        return pred_r

    def get_next_E(self, h_t, t):
        """
        Equation 4
        :param h_t:
        :param t:
        :return:
        """
        # predict next entity
        if (
            self.max_entity_index == self.entities.size(0) - 1
        ):  # max_entity is the last element
            self.add_new_entity()
        dist_feat = self.get_dist_feat(t)
        # Apply bilinear layer
        pred_e = self.W_entity(
            self.dropout(self.entities),
            self.dropout(h_t.expand_as(self.entities))
            + self.w_dist(self.dropout(dist_feat)),
        )
        return pred_e.view(1, -1)

    def get_next_L(self, h_t, entity_embedding):
        """
        Equation 5
        :param h_t:
        :param entity_embedding:
        :return:
        """
        # predict length of next entity
        return self.W_length(self.dropout(torch.cat((h_t, entity_embedding), dim=1)))

    def get_next_X(self, h_t, e_current):
        # predict next token
        return self.output_layer(self.dropout(h_t + self.W_e(self.dropout(e_current))))

    def register_predicted_entity(self, e_index):
        # this function registers entities to determine
        # if there is a free slot in the entity set
        new_max = max(int(e_index), self.max_entity_index)
        self.max_entity_index = new_max

    def reset_state(self):
        # reset all entity states
        self.entities = torch.tensor([], dtype=torch.float, device=self.device)
        self.dist_features = torch.tensor([], dtype=torch.float, device=self.device)
        self.max_entity_index = 0
