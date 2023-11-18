import torch
from torch.utils.data import Dataset
import sentencepiece as spm
import os
import json
from tqdm import tqdm

class TinyStoriesDataset(Dataset):
    def __init__(self, data_path, tokenizer_model_path, processed_data_path, vocab_size=10000, model_type="bpe", num_files=50):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.data_path = data_path
        self.processed_data_path = os.path.join(processed_data_path, f'data_{self.model_type}_{self.vocab_size}_{num_files}')
        if not os.path.exists(self.processed_data_path):
            self.tokenizer = self.load_tokenizer(tokenizer_model_path, vocab_size, model_type, num_files)
        self.data = self.load_data()

    def load_tokenizer(self, model_path, vocab_size, model_type, num_files):
        model_path = model_path + f"{model_type}{vocab_size}{num_files}"
        model_file = model_path + ".model"
        if not os.path.exists(model_file):
            tokenizer = self.train_tokenizer(model_path, vocab_size, model_type, num_files)
        else:
            tokenizer = spm.SentencePieceProcessor(model_file=model_path)
        return tokenizer

    def train_tokenizer(self, model_path, vocab_size, model_type, num_files=50):
        files = [f"{self.data_path}/data{str(i).zfill(2)}.json" for i in range(num_files)]
        with open("tmp.txt", "w") as out:
            for file in tqdm(files, desc="Building tokenizer input"):
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    out.write(" ".join([data_point["story"] for data_point in data]))
                    out.write("\n")
        
        spm.SentencePieceTrainer.train(
            input="tmp.txt",
            model_prefix=model_path,
            vocab_size=vocab_size,
            model_type=model_type,
            user_defined_symbols=['[PAD]', '[UNK]']
        )

        tokenizer = spm.SentencePieceProcessor(model_file=model_path)
        return tokenizer

    def load_data(self):
        if os.path.exists(self.processed_data_path):
            return torch.load(self.processed_data_path)
        files = [f"data/data{str(i).zfill(2)}.json" for i in range(50)]
        data = []
        for file in tqdm(files, desc="Loading data"):
            with open(file, "r", encoding="utf-8") as f:
                data_point = json.load(f)
                story = self.tokenizer.encode_as_ids(data_point["story"])
                data.append(torch.tensor(story, dtype=torch.long))

        data = torch.cat(data)
        torch.save(data, self.processed_data_path)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens
