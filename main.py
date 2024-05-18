from dataset import LocalDataset, NERDataset
from logger import Logger
from baseline_lstm import LSTM_Model
from bidirectional_lstm import BiLSTM_Model
from bidirectional_lstm_crf import BiLSTM_CRF_Model

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import wandb

############################################################################################
###                           Create a checkpoints folder                                ###
############################################################################################

checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

############################################################################################
###                            Setup the Dataset                                         ###
############################################################################################

local_raw_dataset = LocalDataset('dataset')
local_raw_dataset.loadDataset()
local_data_dict = local_raw_dataset.prepareDataset()

label_ids = local_data_dict["label_ids"]

train_dataset = NERDataset((local_data_dict["train_sentences"], local_data_dict["train_labels"]))
val_dataset = NERDataset((local_data_dict["val_sentences"], local_data_dict["val_labels"]))
test_dataset = NERDataset((local_data_dict["test_sentences"], local_data_dict["test_labels"]))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

############################################################################################
###                            Training Variables                                        ###
############################################################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Device: {device}')


############################################################################################
###                    Training annd Testing Functions                                   ###
############################################################################################


def flatten_y(pred_sentences, true_sentences):
    flattened_preds = []
    flattened_trues = []
    for preds, trues in zip(pred_sentences, true_sentences):
        for pred, true in zip(preds, trues):
            flattened_preds.append(int(pred.item()))
            flattened_trues.append(int(true.item()))

    return flattened_trues, flattened_preds


def train_model(model, loss_function, optimizer, num_of_epochs, training_dataloader, validation_dataloader, checkpoint_name, logger):
    epoch_training_losses = []
    for epoch in range(num_of_epochs):
        print(f'Running Epoch {epoch + 1}')
        model.train()
        for sentences, labels in train_dataloader:
            x = sentences.to(device)
            y = labels.to(device)

            model.zero_grad()
            tag_scores = model(x)

            # Flatten the sequences and labels
            tag_scores = tag_scores.view(-1, local_data_dict["tagset_size"])
            y = y.view(-1).type(torch.LongTensor).to(device)
            # print(tag_scores.shape)
            # print(labels.shape)
            
            # print(f'tag_scores device: {tag_scores.get_device()}\ny device: {y.get_device()}')
            loss = loss_function(tag_scores, y)
            loss.backward()
            optimizer.step()

        epoch_training_losses.append(loss.item())
        print(f"Epoch {epoch + 1} Complete, Loss: {loss.item()}")
        print(f"Average Training Loss: {np.average(epoch_training_losses)}")

        model.eval()
        preds = []
        true_labels = []
        print(f'Validating Epoch {epoch + 1}')
        for sentences, labels in val_dataloader:

            x = sentences.to(device)
            y = labels.to(device)
            
            y_pred_scores = model(x)
            y_pred = torch.argmax(y_pred_scores, dim=2)
            preds.extend(y_pred.cpu().numpy())
            true_labels.extend(labels)

        flattened_trues, flattened_preds = flatten_y(preds, true_labels)
        print(f"Validation for Epoch {epoch + 1} Complete, Accuracy: {accuracy_score(flattened_preds, flattened_trues)}")
        logger.log({
            'training_loss': loss.item(),
            'validation_accuracy': accuracy_score(flattened_preds, flattened_trues)
            })
        
        print("Saving model")
        with open(f'{checkpoint_dir}\\{checkpoint_name}.pkl', 'wb') as file:
            pickle.dump(model, file)


def test_crf_model(model, optimizer, num_of_epochs, training_dataloader, validation_dataloader, checkpoint_name, logger):
    epoch_training_losses = []
    for epoch in range(num_of_epochs):
        print(f'Running Epoch {epoch + 1}')
        model.train()
        for sentences, labels in train_dataloader:
            x = sentences.to(device)
            y = labels.to(device)

            model.zero_grad()
            loss = model(x, y)

            loss.backward()
            optimizer.step()

        epoch_training_losses.append(loss.item())
        print(f"Epoch {epoch + 1} Complete, Loss: {loss.item()}")
        print(f"Average Training Loss: {np.average(epoch_training_losses)}")

        model.eval()
        preds = []
        true_labels = []
        print(f'Validating Epoch {epoch + 1}')
        for sentences, labels in val_dataloader:

            x = sentences.to(device)
            y = labels.to(device)
            
            with torch.no_grad():
                y_pred_scores, tag_seq = model.predict(sentences)
                print(tag_seq)

            y_pred = torch.argmax(y_pred_scores, dim=2)
            preds.extend(y_pred.cpu().numpy())
            true_labels.extend(labels)

        flattened_trues, flattened_preds = flatten_y(preds, true_labels)
        print(f"Validation for Epoch {epoch + 1} Complete, Accuracy: {accuracy_score(flattened_preds, flattened_trues)}")
        logger.log({
            'training_loss': loss.item(),
            'validation_accuracy': accuracy_score(flattened_preds, flattened_trues)
            })
        
        print("Saving model")
        with open(f'{checkpoint_dir}\\{checkpoint_name}.pkl', 'wb') as file:
            pickle.dump(model, file)



def test_model(model, test_dataloader, label_ids, logger):
    model.eval()
    preds = []
    true_labels = []
    tags = [key for key in label_ids.keys()]
    print(f'Testing Model')
    for sentences, labels in val_dataloader:

        x = sentences.to(device)
        y = labels.to(device)
        
        y_pred_scores = model(x)
        y_pred = torch.argmax(y_pred_scores, dim=2)
        preds.extend(y_pred.cpu().numpy())
        true_labels.extend(labels)

    flattened_trues, flattened_preds = flatten_y(preds, true_labels)

    true_tags = [tags[label] for label in flattened_trues]
    pred_tags = [tags[label] for label in flattened_preds]
    # print(true_tags)
    # print(pred_tags)


    test_report = classification_report(true_tags, pred_tags, labels=tags, output_dict=True, zero_division=0)
    print(f'classification_report:\n{classification_report(true_tags, pred_tags, labels=tags, output_dict = False, zero_division = 0)}')

    matrix = confusion_matrix(true_tags, pred_tags, labels=tags)

    ax = sns.heatmap(pd.DataFrame(test_report).iloc[:-1, :].T, cmap = 'coolwarm', annot=True)
    plt.tight_layout()
    logger.log({'classification_report': wandb.Image(ax.figure)})
    plt.close()

    ax1 = sns.heatmap(matrix/np.sum(matrix), cmap = 'Blues', annot=True, xticklabels=tags, yticklabels=tags, fmt='.2f')
    logger.log({'confusion_matrix': wandb.Image(ax1.figure)})
    plt.close()

    logger.log({
        'test_acc': accuracy_score(flattened_preds, flattened_trues)
        })



############################################################################################
###                            Training calls                                            ###
############################################################################################

wandb_logger = Logger(f"inm706_cw_bidirectional_LSTM_CRF_training", project='INM706_CW')
logger = wandb_logger.get_logger()

# bi_LSTM_model = BiLSTM_Model(local_data_dict["vocab_size"], device)
bi_LSTM_crf_model = BiLSTM_CRF_Model(local_data_dict["vocab_size"], local_data_dict["tagset_size"], device)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(bi_LSTM_crf_model.parameters(), lr=0.01)

# train_model(bi_LSTM_model, loss_function, optimizer, 500, train_dataloader, val_dataloader, "biLSTM_500ep", logger)


test_crf_model(bi_LSTM_crf_model, optimizer, 5, train_dataloader, val_dataloader, "biLSTM_crf_5ep", logger)

############################################################################################
###                            Testing calls                                             ###
############################################################################################

# checkpoint_name = 'baseline_LSTM'

# wandb_logger = Logger(f"inm706_cw_base_LSTM_first_test", project='INM706_CW')
# test_logger = wandb_logger.get_logger()

# print(f'loading model')
# with open(f'{checkpoint_dir}\\{checkpoint_name}.pkl', 'rb') as file:
#     testing_model = pickle.load(file)

# test_model(testing_model, test_dataloader, label_ids, test_logger)