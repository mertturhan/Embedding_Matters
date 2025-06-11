import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import max_len_seq
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from fixed_embeddings import load_fixed_embeddings
from model import LSTMModel
from dataset import ClickstreamDataset




def custom_collate_fn(batch):
    """
    pads sequences in a batch. i.e., add zeros to make every session of equal length
    each item in a batch is a tuple of
    - prod_seq: tensor of shape (seq_length,)
    - additional_features: tensor of shape (seq_length, feature_dim)
    - target: Tensor of shape (seq_length, 1)
    :return:
    - prod_seqs: long tensor of shape (batch_size, padded_seq_length)
    - add_feats: float tensor of shape (batch_size, padded_seq_length, feature_dim)
    - targets: float tensor of shape (batch_size, padded_seq_length, 1)
    """
    prod_seqs, add_feats, targets = zip(*batch)
    prod_seqs_padded = pad_sequence(prod_seqs, batch_first=True, padding_value=0)
    # If additional features are provided, pad each separately.
    add_feats_padded = pad_sequence(add_feats, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return prod_seqs_padded, add_feats_padded, targets_padded

#load product embeddings and their mappings from csv
pretrained_embeddings_array, product_id_to_index = load_fixed_embeddings('fixed_embeddings.csv')
pretrained_embeddings = torch.tensor(pretrained_embeddings_array, dtype=torch.float32)


dframe = pd.read_csv('placeholder.csv')
#to perform a session-level random split
session_keys = dframe[['anonymous_id','session_number']].drop_duplicates()



shuffled_keys = session_keys.sample(frac=1, random_state=31)
num_sessions = len(shuffled_keys)
train_size = int(0.8 * num_sessions) #0.8 represents the relative size of the training set

train_keys = shuffled_keys.iloc[:train_size]
test_keys = shuffled_keys.iloc[train_size:]

train_set = pd.merge(dframe, train_keys, on=['anonymous_id','session_number'], how='inner')
test_set = pd.merge(dframe, test_keys, on=['anonymous_id', 'session_number'], how='inner')



additional_features =[
    'anonymous_id','timestamp', 'new_session_number', 'new_session_hit_number','carryover', 'event', 'page_type','country_code','device', 'browser',
    'operating_system', 'time_spent_on_product_pages','with_order','sin_day','cos_day','sin_minute','cos_minute'
]
#for 1-hot encoding
categorical_features = ['event','page_type','country_code','device','browser','operating_system']

#this is the function to set our target variable, the main and only assumption done in the whole model. This was a hard assumption to make, and it is probably not representative of all the sessions.
#it had to be done. this is the only way to prevent data leakage. because i am predicting a behavior i have to set a threshold to tell the model 'Look, this is when users signal buying something from the website"
#the model performance will heavily rely on this assumption, as well as hyperparameters, even more because hyperparameter space could be explored to find the best option.

def lookback_function(degree=4):
    anchor_x = np.array([1, 2, 5.5, 9, 13.0, 18, 28, 50])
    anchor_y = np.array([1, 0.95, 0.7, 0.45, 0.35, 0.42, 0.47, 0.5])

    coeffs = np.polyfit(anchor_x, anchor_y, deg=degree)
    poly_func = np.poly1d(coeffs)
    return poly_func

poly = lookback_function(degree=4)



train_dataset = ClickstreamDataset(
    dataframe=train_set,
    product_id_to_index=product_id_to_index,
    additional_columns=additional_features,
    categorical_columns=categorical_features,
    lookback_function = poly,
    max_seq_length=45)

test_dataset = ClickstreamDataset(
    dataframe=test_set,
    product_id_to_index=product_id_to_index,
    additional_columns=additional_features,
    categorical_columns=categorical_features,
    lookback_function=poly,
    max_seq_length=45)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

sample = train_dataset[0]
_, additional_features_sample, _ = sample
additional_feature_dim =additional_features_sample.shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(
    pretrained_embeddings=pretrained_embeddings,
    alpha=1.0,
    hidden_dim=128,
    num_layers=1,
    num_classes=1,
    bidirectional=False,
    threshold=0.7,
    other_feature_dim=additional_feature_dim
)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for prod_seqs, additional_features, targets in loader:
        #move all tensors to GPU (if available)
        prod_seqs = prod_seqs.to(device)
        additional_features = additional_features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(prod_seqs, x_other = additional_features, apply_patience=False)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for prod_seqs, additional_features, targets in loader:
            #to GPU
            prod_seqs = prod_seqs.to(device)
            additional_features= additional_features.to(device)
            targets = targets.to(device)
            logits = model(prod_seqs, x_other = additional_features, apply_patience=True)
            loss = criterion(logits, targets)
            epoch_loss += loss.item()
        return epoch_loss / len(loader)



##Main training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model ,train_loader, optimizer, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)




