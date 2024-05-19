import torch
import torch.nn as nn
import torch.optim as optim

""" 
All the code of this model is a mixture of the following:
 - Pytorch documentation: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
 - ChatGpt
 - Further modified to fit my use/the format of my data
"""

class BiLSTM_CRF_Model(nn.Module):
    def __init__(self, vocab_size, tagset_size, device):
        super(BiLSTM_CRF_Model, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, 64, device=self.device)
        self.lstm = nn.LSTM(64, 128, num_layers=1, bidirectional=True, batch_first=True, device=self.device)
        self.hidden2tag = nn.Linear(256, tagset_size, device=self.device)

        # CRF layer parameters
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.transitions.data[:, 0] = -10000  # No transition to start
        self.transitions.data[0, :] = -10000  # No transition from end, except to start

        self.tagset_size = tagset_size

    def forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_alphas[0][0] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1).to(self.device)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[0].to(self.device)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def get_lstm_features(self, sentences):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        # lstm_out = lstm_out.view(len(sentences), -1, self.hidden2tag.out_features)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def score_sentence(self, feats, tags):
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([0], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[0, tags[-1]]
        return score

    def viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][0] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[0]
        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        best_path.reverse()

        return path_score, best_path

    def forward(self, sentences, tags):
        print('forward call')
        feats = self.get_lstm_features(sentences)
        batch_size = sentences.size(0)
        forward_scores = torch.zeros(batch_size, device=self.device)
        gold_scores = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            print(f'loop: {i}')
            forward_scores[i] = self.forward_alg(feats[i])
            gold_scores[i] = self.score_sentence(feats[i], tags[i])
        return forward_scores.sum() - gold_scores.sum()

    def predict(self, sentences):
        lstm_feats = self.get_lstm_features(sentences)
        score, tag_seq = self.viterbi_decode(lstm_feats)
        return score, tag_seq