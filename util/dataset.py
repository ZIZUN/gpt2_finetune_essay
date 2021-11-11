from torch.utils.data import Dataset

class essay_dataset(Dataset):
    def __init__(self, file_path, vocab, tokenizer):
        self.file_path = file_path
        self.data = []
        self.vocab = vocab
        self.tokenizer = tokenizer
        f = open(self.file_path, 'r', encoding='utf-8')
        datasets = []
        while True:
            line = f.readline()
            datasets.append([line])
            if not line:
                break
        f.close()

        print("tokenizer ending")
        for line in datasets:
            if not line[0]:
                break
            if len(line[0]) < 3:
                continue

            index_of_words = [tokenizer.bos_token_id] + tokenizer.encode(line[0][:-1]) + [tokenizer.eos_token_id]

            if len(index_of_words) > 382:
                continue

            index_of_words += [tokenizer.pad_token_id] * (384 - len(index_of_words))
            # print(index_of_words)

            if len(index_of_words) > 1024:
                continue
            elif len(index_of_words) < 10:
                continue

            self.data.append(index_of_words)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item