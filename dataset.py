import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
from sklearn.utils import shuffle


class LocalDataset():
    def __init__(self, dir):
        # Directory of dataset csv file
        self.dir = dir
        self.unpadded_sentences = []
        self.unpadded_tokens = []
        self.padded_tokens = []
        self.vocab_size = 0
        self.tagset_size = 0
        self.vocab_ids = {}
        self.label_ids = {}
        self.tags = []
        self.unpadded_labels = []
        self.padded_labels = []
        self.max_length = 0


    # Retrieve the dataset from the CSV file
    def loadDataset(self):
        print(f'Loading local csv from: {os.path.join(os.getcwd(), self.dir)}')
        self.raw_data = pd.read_csv(os.path.join(os.getcwd(), self.dir+'.csv'), encoding='unicode_escape')


    # Pad sentences to match the longest sentece in the dataset, padding value of 0
    def pad_sentence(self, sentence):
        if len(sentence) < self.max_length:
            return np.append(sentence, np.zeros(self.max_length - len(sentence), dtype=int))
        else:
            return sentence

    # convert the string tags to integer values
    def convert_tags_to_labels(self, tags):
        return [self.label_ids[tag] for tag in tags]

    # convert the string words to integer values
    def convert_words_to_tokens(self, sentence):
        return [self.vocab_ids[word] for word in sentence]

    # main function used to convert the data into a 
    def prepareDataset(self, train_split=0.6, validation_split=0.2, testing_split=0.2):

        if not (train_split + validation_split + testing_split == 1):
            raise ValueError("Expected a sum of the splits equal 1")

        for ind, tag in enumerate(self.raw_data['Tag'].unique()):
            self.label_ids[tag] = ind

        for ind, word in enumerate(self.raw_data['Word'].unique()):
            self.vocab_ids[word] = ind + 1

        self.tagset_size = len(self.raw_data['Tag'].unique())
        self.vocab = self.raw_data['Word'].unique()
        self.vocab_size = len(self.raw_data['Word'].unique())

        start_of_setence_indices = self.raw_data[self.raw_data['Sentence #'].notnull()].index.to_numpy()
        for ind, val in enumerate(start_of_setence_indices):
            if ind < len(start_of_setence_indices) - 1:
                sentence_arr = self.raw_data[val:start_of_setence_indices[ind + 1]]['Word'].to_numpy()
                tags_arr = self.raw_data[val:start_of_setence_indices[ind + 1]]['Tag'].to_numpy()
                self.unpadded_sentences.append(sentence_arr)
                self.tags.append(tags_arr)
            else:
                sentence_arr = self.raw_data[val:]['Word'].to_numpy()
                tags_arr = self.raw_data[val:]['Tag'].to_numpy()
                self.unpadded_sentences.append(sentence_arr)
                self.tags.append(tags_arr)
            if len(sentence_arr) > self.max_length:
                self.max_length = len(sentence_arr)

        # print(self.unpadded_sentences)
        # print(f'Max Length is: {self.max_length}')

        self.unpadded_labels = [self.convert_tags_to_labels(tags) for tags in self.tags]
        self.unpadded_tokens = [self.convert_words_to_tokens(sentence) for sentence in self.unpadded_sentences]

        self.padded_tokens = [self.pad_sentence(sentence) for sentence in self.unpadded_tokens]
        self.padded_labels = [self.pad_sentence(sentence_labels) for sentence_labels in self.unpadded_labels]

        # print(self.padded_sentences[2])
        # print(self.padded_labels[2])
        print(self.vocab_size)

        shuffles_sentences, shuffled_labels = shuffle(self.padded_tokens, self.padded_labels)

        train_sentences = self.padded_tokens[:int(len(self.padded_tokens)*train_split)]
        train_labels = self.padded_labels[:int(len(self.padded_labels)*train_split)]
        val_sentences = self.padded_tokens[int(len(self.padded_tokens)*train_split):int(len(self.padded_tokens)*(train_split+validation_split))]
        val_labels = self.padded_labels[int(len(self.padded_labels)*train_split):int(len(self.padded_labels)*(train_split+validation_split))]
        test_sentences = self.padded_tokens[int(len(self.padded_tokens)*(train_split+validation_split)):]
        test_labels = self.padded_labels[int(len(self.padded_labels)*(train_split+validation_split)):]


        return {
            "train_sentences": train_sentences,
            "train_labels": train_labels,
            "val_sentences": val_sentences,
            "val_labels": val_labels,
            "test_sentences": test_sentences,
            "test_labels": test_labels,
            "vocab_size":  self.vocab_size,
            "tagset_size": self.tagset_size,
            "label_ids": self.label_ids,
            "vocab": self.vocab
        }

class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sentences = data[0]
        self.labels = data[1]

    def __getitem__(self, index):
        return torch.tensor(self.sentences[index]), torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.sentences)
