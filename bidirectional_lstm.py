import torch.nn as nn
import torch

class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, device):
        super(BiLSTM_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64, device=device)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True, device=device)
        self.hidden2tag = nn.Linear(256, 17, device=device)

    def forward(self, sentences):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores