from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class YahooDataset(Dataset):
    def __init__(self, path, model_name, max_token_len=128):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_token_len = max_token_len
        self.text, self.label = [], []
        self.read_csv(path)

    def read_csv(self, path):
        df = pd.read_csv(path,
                         delimiter=',',
                         names=['label', 'question_title', 'question_content', 'best_answer']).iloc[:, :-1]

        df = df.fillna('')
        df = df.astype(str)
        df['label'] = df['label'].astype(int)
        df['question_title'] = df['question_title'].str.lower()
        df['question_content'] = df['question_content'].str.lower()
        for row in df.itertuples():
            self.label.append(row.label - 1)  # from 0 to 4
            string = " ".join([row.question_title, row.question_content])
            string = string.replace(r'\n', ' ')
            string = ' '.join(string.split())
            self.text.append(string)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        comment = self.text[index]
        label = self.label[index]
        tokens = self.tokenizer.encode_plus(comment,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.max_token_len,
                                            return_attention_mask=True)
        return tokens.input_ids.flatten(), tokens.attention_mask.flatten(), label
