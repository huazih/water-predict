from torch.utils.data import Dataset


class ExportData(Dataset):
    def __init__(self, targets, seq_length):
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.targets) - self.seq_length

    def __getitem__(self, idx):
        return (self.targets[idx-self.seq_length:idx], self.targets[idx+1])