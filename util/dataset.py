from torch.utils.data import Dataset
import csv

class essay_dataset(Dataset):
    def __init__(self, file_path, vocab, tokenizer):
        self.file_path = file_path
        self.data = []
        self.vocab = vocab
        self.tokenizer = tokenizer

        f = open(self.file_path,'r')
        rdr = csv.reader(f)
        datasets = []
        for line in rdr:
            datasets.append(line)
            print(line)
        f.close()

        print("tokenizer ending")
        for line in datasets:
            if not line[0]:
                break
            if len(line[0]) < 3:
                continue

            ids = tokenizer.encode(line[0][:-1])
            max_len = 384
            while True:     ## sliding window technic
                if len(ids) > max_len-2:
                    index_of_words = [tokenizer.bos_token_id] + ids[:max_len-2] + [tokenizer.eos_token_id]
                    index_of_words += [tokenizer.pad_token_id] * (max_len - len(index_of_words))
                    self.data.append(index_of_words)

                    ids = ids[max_len-20:]
                else:
                    if len(ids) < 100:
                        break
                    else:
                        index_of_words = [tokenizer.bos_token_id] + ids[:max_len - 2] + [tokenizer.eos_token_id]
                        index_of_words += [tokenizer.pad_token_id] * (max_len - len(index_of_words))
                        self.data.append(index_of_words)
                        break

            self.data.append(index_of_words)
            print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item