from torch.utils.data import Dataset


class WaterLevelDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.valid_indices = [i for i in range(seq_length, len(targets) - 1)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]

        # 获取特征和目标
        feature_data = self.features[valid_idx - self.seq_length:valid_idx]
        target_data = self.targets[valid_idx + 1]
        return feature_data, target_data