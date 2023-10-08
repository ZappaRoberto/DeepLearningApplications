from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class distilroberta_base(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model = AutoModel.from_pretrained('distilroberta-base', return_dict=True)
        self.hidden = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_class)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = torch.mean(output.last_hidden_state, 1)
        output = F.relu(self.hidden(output))
        return self.classifier(output)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = distilroberta_base(n_class=10).to(device)
    input_ids = torch.randint(0, 1000, (32, 128)).to(device)
    attention_mask = torch.randint(0, 2, (32, 128)).to(device)
    print(model(input_ids, attention_mask).shape)