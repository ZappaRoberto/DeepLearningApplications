from torch.utils.data import Dataset
from tokenizer import Tokenization
import torch
from textwrap import wrap
from datasets import load_dataset


class Dante(Dataset):
    def __init__(self, text_path, datatype, max_token_len):
        self.tokenizer = Tokenization()
        self.max_token_len = max_token_len
        with open(text_path, "r", encoding='utf-8') as file:
            self.text = file.read()
            self.text = self.tokenizer.dataset_preparation(self.text)

        self.train = self.text[:int(len(self.text)*0.8)]
        self.test = self.text[int(len(self.text)*0.8):]
        self.datatype = datatype

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.datatype == 'train':
            batch_data = self.train
        else:
            batch_data = self.test

        # pick random starting points
        ix = torch.randint(0, batch_data.size(0) - self.max_token_len - 1, (1,))
        data = (batch_data[ix: ix + self.max_token_len]).long()
        target = (batch_data[ix + self.max_token_len: ix + self.max_token_len + 1]).long()
        return data, target

"""
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test

    # pick random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    MASTER_CONFIG.update({
        'batch_size': 8,
        'context_window': 16
    })

    xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

    [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]
    return x, y
"""

if __name__ == '__main__':
    d = Dante('Dataset/dataset.txt', 'train')
    print(d[0])
    tokenizer = Tokenization(128)
    print(tokenizer.decode(d[0][0]))
    print('-------------------')
    print(tokenizer.decode(d[0][1]))
