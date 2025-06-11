import pandas as pd
import torch
import torch.nn as nn


df_click = pd.read_csv("placeholder.csv")

sessions_to_remove = df_click[df_click['new_session_hit_number']>45][['anonymous_id', 'session_number']]
# a filter for training dataset, sessions longer than 45 clicks will be removed. this roughly corresponds to removing the outliers, the long sequences that are not representative
#also, adjusting for them would require higher computational resources, since sequences get longer, and we need to pad every sequence shorter than the threshold to match the length, so lots of unnecessary 0s.

df_fixed_embed = pd.read_csv("product_embeddings.csv")

embedding_columns = df_click.columns[:-1]
product_ids = df_click['product_id'].tolist()

product_id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

pretrained_embeddings = torch.tensor(df_click[embedding_columns].values, dtype=torch.float32)


#An attention mechanism will be implemented to possibly increase the performance of the model
#this mechanism captures how every each input relates to every other input in the sequence



class LSTMModel(nn.Module):
    def __init__(self, pretrained_embeddings, alpha=1.0, hidden_dim=128, num_classes=1, num_layers=1, bidirectional = False, patience=3, threshold=0.7, other_feature_dim = 0):
        """
        Arguments
        :param pretrained_embeddings: fixed embeddings tensor of shape (vocab_size, embedding_dim)
        alpha: Float in [0,1] controlling the balance of fixed and dynamic embeddings:
            1.0 -> use completely fixed embeddings
            0.0 -> use completely dynamic embeddings
            intermediate values mix both based on the given value
        :param hidden_dim: hidden dimensionality for the LSTM model
        :param num_classes:output dimension, 1 for binary
        :param num_layers: number of LSTM layers
        :param bidirectional: whether LSTM is bidirectional
        :param patience: number of consecutive timesteps in predictions to trigger halting
        :param threshold: prediction probability threshold
        other_feature_dim: dimension of additional per-time-step features.
            These features should already be preprocessed (e.g. one-hot encoding)
                and provided as a tensor of shape (batch_size, seq_length, additional_feature_dim)
        """
        super(LSTMModel, self).__init__()
        self.alpha = alpha
        self.bidirectional = bidirectional
        self.patience = patience
        self.threshold = threshold

        vocab_size, fixed_dim = pretrained_embeddings.size()

        self.Dimension_fixed = int(round(alpha * fixed_dim))
        self.Dimension_dynamic = fixed_dim - self.Dimension_fixed

        if self.Dimension_fixed > 0:
            self.fixed_embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
            #if dimensions do not match, project it to a lower dimensional space, so we can fill in the remainder of the matrix with values trained during training
            if self.Dimension_fixed != fixed_dim:
                self.fixed_projection = nn.Linear(fixed_dim, self.Dimension_fixed)
            else:
                self.fixed_projection = nn.Identity()
        else:
            self.fixed_embedding = None

        if self.Dimension_dynamic > 0:
            self.dynamic_embedding = nn.Embedding(vocab_size, self.Dimension_dynamic)
        else:
            self.dynamic_embedding = None

        combined_product_embedding_dim = fixed_dim
        total_input_dim = combined_product_embedding_dim + other_feature_dim

        self.LSTM = nn.LSTM(
            input_size= total_input_dim,
            hidden_size = hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional = bidirectional
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim


        self.fc = nn.Linear(lstm_out_dim, num_classes)

    #forward pass in the network
    def forward(self, x, x_other = None, apply_patience = False):
        """

        :param x: input tensor of shape (batch_size, seq_length) containing data points
        apply_patience : when true, the forward pass will post-process each sequence and halt predictions after the patience criterion is met
        :x_other: other features analyzed as a tensor
        """
        if self.fixed_embedding is not None:
            fixed_embeds = self.fixed_projection(self.fixed_embedding(x))  # shape: (batch_size, seq_length, Dimension_fixed)
        else:
            fixed_embeds = None

        if self.dynamic_embedding is not None:
            dynamic_embeds = self.dynamic_embedding(x)   # shape: (batch_size, seq_length, D_fixed)
        else:
            dynamic_embeds = None

        if fixed_embeds is not None and dynamic_embeds is not None:
            product_embeds = torch.cat((fixed_embeds, dynamic_embeds), dim=-1)
        else:
            product_embeds = fixed_embeds if fixed_embeds is not None else dynamic_embeds

        if x_other is not None:
            #x_other should be pre-processed
            #we concatenate them along the last dimension
            lstm_input = torch.cat((product_embeds, x_other), dim=-1)
        else:
            lstm_input = product_embeds

        #forward pass
        lstm_out, _ = self.lstm(lstm_input)  # shape: (batch_size, seq_length, lstm_out_dim)
        logits = self.fc(lstm_out)           # shape: (batch_size, seq_length, num_classes)

        if apply_patience:
            #convert logit probabilities  for inference
            probs = torch.sigmoid(logits)
            batch_size, seq_length, _ = probs.shape
            #we need to put these probabilities on cpu
            probs_np = probs.detach().cpu().numpy()
            for i in range (batch_size):
                count = 0
                halt_index = None
                for t in range(seq_length):
                    if probs_np[i, t, 0] > self.threshold:
                        count += 1
                        if count >= self.patience:
                            halt_index = t
                            break
                    else:
                        count = 0
                    if halt_index is not None:
                        probs_np[i, halt_index + 1:, 0] = probs_np[i, halt_index, 0] #Train halted. We could assign 1 instead of the latest probability estimated since we are 'certain' the customer will buy

                probs = torch.tensor(probs_np, dtype=probs.dtype, device=probs.device)
                return probs
        else:
            #for training return raw logits, nn,BCEWithLogitsLoss will be used. Claimed to be more stable
            return logits











#criterion = nn.BCEWithLogitsLoss() ## no need to apply sigmoid on the last layer because it applies it itself

# 1- build a lookup tensor from the coming csv file
# define LOOKUP BEHAVIOR INSIDE THE LSTMfixed class.
