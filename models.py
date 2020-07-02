# from apex import amp
import torch


class ResNeXtModel(torch.nn.Module):
    def __init__(self):
        super(ResNeXtModel, self).__init__()
        resnext = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        self.base = torch.nn.Sequential(*list(resnext.children())[:-1])
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 6)
        )

    def forward(self, input):
        features = self.base(input).reshape(-1, 2048)
        out = self.fc(features)
        return out, features


class EmbeddingSmootherModel(torch.nn.Module):

    def __init__(self, features=120, hidden_size=256):
        super(EmbeddingSmootherModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(features + 6, self.hidden_size, num_layers=3, dropout=0.3, batch_first=True,
                                  bidirectional=True)
        self.scan_rnn = torch.nn.GRU(6, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size * 2 + 6, 6)
        )
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, seq, preds):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        hidden = (
            torch.zeros(6, 1, self.hidden_size).to(device),
            torch.zeros(6, 1, self.hidden_size).to(device)
        )

        out, hidden = self.lstm(seq, hidden)
        combined_out = torch.cat((out, preds), 2)
        out = self.classifier(self.dropout(combined_out))

        return out
