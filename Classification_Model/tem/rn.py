import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


class AttributeNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_rn(an, rn, train_loader, val_loader, word_embeddings, val_label_range, device=torch.device('cpu:0'), lr=1e-4, wd=1e-9, n_ep=100):
    an = an.to(device)
    rn = rn.to(device)

    opt_an = torch.optim.Adam(an.parameters(), lr=lr, weight_decay=wd)
    sch_an = torch.optim.lr_scheduler.StepLR(opt_an, step_size=1, gamma=0.9)

    opt_rn = torch.optim.Adam(rn.parameters(), lr=lr)
    sch_rn = torch.optim.lr_scheduler.StepLR(opt_rn, step_size=1, gamma=0.9)

    for i_ep in range(n_ep):
        an.train()
        rn.train()
        sch_an.step(i_ep)
        sch_rn.step(i_ep)
        for _, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
            feature = sample['feature'].to(device)
            batch_size = feature.size()[0]

            idx = sample['idx'].numpy()
            idx_u = np.unique(idx)
            word_emb = torch.from_numpy(word_embeddings[idx_u].astype('float32')).to(device)
            n_classes = word_emb.size()[0]
            mapping = {idx: i for i, idx in enumerate(idx_u)}
            label = [mapping[i] for i in idx]

            one_hot = np.zeros([batch_size, n_classes], dtype=np.float32)
            for i in range(len(one_hot)):
                one_hot[i, label[i]] = 1
            one_hot = torch.from_numpy(one_hot).to(device)

            word_emb = word_emb.repeat(batch_size, 1)
            feature = feature.repeat(1, n_classes).view(batch_size * n_classes, -1)

            logits = rn(torch.cat([an(word_emb), feature], 1)).view(batch_size, n_classes)
            loss = F.mse_loss(logits, one_hot)

            opt_an.zero_grad()
            opt_rn.zero_grad()
            loss.backward()
            opt_an.step()
            opt_rn.step()


        an.eval()
        rn.eval()
        with torch.no_grad():
            val_word_emb = torch.from_numpy(word_embeddings[val_label_range].astype('float32'))
            n_classes = len(val_label_range)

            preds = []
            ys = []
            for _, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
                feature = sample['feature'].to(device)
                idx = sample['idx'].numpy()
                batch_size = feature.size()[0]

                val_we = val_word_emb.repeat(batch_size, 1).to(device)
                feature = feature.repeat(1, n_classes).view(batch_size * n_classes, -1)

                logits = rn(torch.cat([an(val_we), feature], 1)).view(batch_size, n_classes)
                pred = logits.max(1)[1].detach().cpu().numpy()
                pred = np.asarray([val_label_range[p] for p in pred])
                preds += [pred]
                ys += [idx]

            preds = np.concatenate(preds, 0)
            ys = np.concatenate(ys, 0)
            acc = accuracy_score(ys, preds)
            print(f'EP_{i_ep}: {acc:.5%}')
