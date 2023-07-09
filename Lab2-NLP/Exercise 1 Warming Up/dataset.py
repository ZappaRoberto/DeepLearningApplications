from torch.utils.data import Dataset
from tokenizer import Tokenization
from textwrap import wrap


class Dante(Dataset):
    def __init__(self, text_path, datatype):
        self.text = [Tokenization().encode(t) for t in wrap(open(text_path, 'r').read(), 1024)]
        self.train = self.text[:int(len(self.text)*0.8)]
        self.test = self.text[int(len(self.text)*0.8):]
        self.datatype = datatype

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.text[idx]


if __name__ == '__main__':
    d = Dante('Dataset/Divina Commedia.txt', 'train')
    print(d[1])
