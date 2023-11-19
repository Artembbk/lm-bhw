import torch
from torch.utils.data import Dataset
import sentencepiece as spm
import os
import json
import numpy as np
from tqdm import tqdm

class TinyStoriesDataset(Dataset):
    def __init__(self, data_path, tokenizer_model_path, processed_data_path, vocab_size=10000, model_type="bpe", num_files_for_tokenizer=50, num_files_for_data=50):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.data_path = data_path
        self.processed_data_path = os.path.join(processed_data_path, f'data_{self.model_type}_{self.vocab_size}_{num_files_for_tokenizer}.npy')
        self.tokenizer = self.load_tokenizer(tokenizer_model_path, vocab_size, model_type, num_files_for_tokenizer)
        self.data = self.load_data(num_files_for_data)

    def load_tokenizer(self, model_path, vocab_size, model_type, num_files_for_tokenizer):
        model_path = model_path + f"{model_type}{vocab_size}{num_files_for_tokenizer}"
        model_file = model_path + ".model"
        if not os.path.exists(model_file):
            tokenizer = self.train_tokenizer(model_path, vocab_size, model_type, num_files_for_tokenizer)
        else:
            tokenizer = spm.SentencePieceProcessor(model_file=model_path+".model")
        return tokenizer

    def train_tokenizer(self, model_path, vocab_size, model_type, num_files_for_tokenizer=50):
        files = [f"{self.data_path}/data{str(i).zfill(2)}.json" for i in range(num_files_for_tokenizer)]
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
            user_defined_symbols=['[PAD]', '[BOS]', '[EOS]', '[UNK]']
        )

        tokenizer = spm.SentencePieceProcessor(model_file=model_path+".model")
        return tokenizer

    def load_data(self, num_files_for_data=50):
        if os.path.exists(self.processed_data_path):
            return np.load(self.processed_data_path)
        files = [f"{self.data_path}/data{str(i).zfill(2)}.json" for i in range(num_files_for_data)]
        res = []
        for file in tqdm(files, desc="Loading data"):
            with open(file, "r", encoding="utf-8") as f:
                all_data = json.load(f)
                data = np.zeros((len(all_data), 256), dtype=np.int16)
                for i, data_point in enumerate(all_data):
                    story = self.tokenizer.encode(data_point["story"])
                    story = story[:min(254, len(story))]
                    data[i, 0] = self.tokenizer['[BOS]']
                    data[i, 1:len(story)+1] = story
                    data[i, len(story)+1] = self.tokenizer['[EOS]']
            res.append(data)

        res = np.concatenate(res)
        np.save(self.processed_data_path, res)

        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens
