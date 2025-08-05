import numpy as np
import pandas as pd
import os
import torch
from utility.data_utils import DataUtils
from torch.utils.data import TensorDataset, IterableDataset, DataLoader, ConcatDataset



class DataObject:
    def __init__(self, train_loader, val_loader, test_loader, input_df, target_df, in_channels, in_times):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.input_df = input_df
        self.target_df = target_df
        self.in_channels = in_channels
        self.in_times = in_times


def data_loader(config):
    target_df = pd.read_excel(config.Standard_coord_path, index_col=0)
    style, input_df, data_x_dir = load_input_df_and_data(config)


    # TUAB-style data loading
    if style == "tuab":
        tuab_root = f"{config.Dataset_path}/128hz_{config.TU_sequen}seqlen/"
        train_loader, val_loader, test_loader = DataUtils.TUABLoader(
            tuab_root=tuab_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        sample_batch = next(iter(train_loader))[0]
        return DataObject(train_loader, val_loader, test_loader, input_df, target_df,
                          sample_batch.shape[1], sample_batch.shape[2])

    # / BCIC-style data loading (Emotiv style key is different)
    if style in ["emotiv", "bcic_p300", "moabb"]:
        train_data, train_label = data_x_dir['X_train'], data_x_dir['Y_train']
        val_data, val_label = data_x_dir['X_val'], data_x_dir['Y_val']
        test_data, test_label = data_x_dir['X_test'], data_x_dir['Y_test']


    if style == "moabb":
        # === Label Encoding for string labels ===
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        all_labels = np.concatenate([train_label, val_label, test_label])
        label_encoder.fit(all_labels)

        train_label = label_encoder.transform(train_label)
        val_label = label_encoder.transform(val_label)
        test_label = label_encoder.transform(test_label)

        print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


    # Dataloader creation
    train_loader, val_loader, test_loader = DataUtils.EmotivLoader(
        train_data, train_label,
        val_data, val_label,
        test_data, test_label,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    sample_batch = next(iter(train_loader))[0]
    return DataObject(train_loader, val_loader, test_loader, input_df, target_df,
                      sample_batch.shape[1], sample_batch.shape[2])


def load_input_df_and_data(config):
    #if config.data_name in ['Alpha', 'Attention', "STEW", "Crowdsourced"]:
    if config.Dataset_style == "BCICP300":
        style = "bcic_p300"
        input_df = pd.read_excel(config.BCICP300_coord_path, index_col=0)
        path = f"{config.BCICP300_path}/{config.data_name}/{config.data_name}.npy"
    elif config.Dataset_style == "Emotiv":
        style = "emotiv"
        input_df = pd.read_excel(config.Emotiv_coord_path, index_col=0)
        path = f"{config.Emotiv_path}/{config.data_name}/{config.data_name}.npy"
    elif config.Dataset_style =="MOABB":
        style = "moabb"
        if config.data_name == "PhysionetMI":
            input_df = pd.read_excel(config.PhysionetMI_coord_path, index_col=0)
        elif config.data_name == "BNCI2014_001":
            input_df = pd.read_excel(config.BNCI2014_001_coord_path, index_col=0)
        elif config.data_name == "GrosseWentrup2009":
            input_df = pd.read_excel(config.GrosseWentrup2009_coord_path, index_col=0)
        else:
            raise ValueError(f"Unsupported MOABB data_name: {config.data_name}")
        path = f"{config.MOABB_path}/{config.data_name}/{config.data_name}.npy"
    elif config.Dataset_style == "TUAB":
        style = "tuab"
        input_df = pd.read_excel(config.TUAB_coord_path, index_col=0)
        path = None  # TUAB handled differently later
    else:
        raise ValueError(f"Unsupported Dataset_path: {config.Dataset_path}")
    if path:
        data_x_dir = np.load(path, allow_pickle=True)

    else:
        data_x_dir = None

    return style, input_df, data_x_dir



# sharing same weight during each batch. But some small dataset may copy-paste too much times causing bias.
# -> Example: If Crowdsourced only has 12k and DriverDistraction has 66k,
#    but weights are uniform, Crowdsourced will be over-sampled (reused many times).
# -> This helps improve linear probe accuracy in low-SNR settings like EEG,
#    but also makes the model biased toward smaller datasets.
class MultiSourceEEGDataset(IterableDataset):
    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        self.keys = list(datasets.keys())
        self.weights = weights or [1.0 / len(self.keys)] * len(self.keys) #[0.25, 0.25, 0.25, 0.25]

    def __iter__(self):
        iterators = {k: iter(self.datasets[k]) for k in self.keys}
        while True:
            source = np.random.choice(self.keys, p=self.weights)
            try:
                yield next(iterators[source])
            except StopIteration:
                iterators[source] = iter(self.datasets[source])
                yield next(iterators[source])

    def __len__(self):
        '''
        Example Sizes:
            - Crowdsourced: 12,296
            - STEW: 28,512
            - Attention: 21,894
            - DriverDistraction: 66,197
        Total sample pool = sum of all datasets = ~128,000

        Since sampling uses fixed weights (default uniform), small datasets will be 
        reused multiple times => Crowdsourced could appear ~30k times (repeated 3x).
            => Model eventually tend to Crowdsourced, due to EEG is low SNR, repetition is helpful
            => Prove Data augmentation and Data Aggregation is very useful for EEG.
        '''
        return sum(len(ds) for ds in self.datasets.values())


# strictly balanced sampling: each dataset contributes at most its own size
# => No sample is reused; when the smallest dataset is exhausted, iteration ends.
# => Prevents bias toward small dataset repetition, but reduces total training volume.
class TruncatedMultiSourceDataset(IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.keys = list(datasets.keys())
        # Each dataset can contribute only min_len samples
        self.min_len = min(len(ds) for ds in datasets.values())

    def __iter__(self):
        iterators = {k: iter(self.datasets[k]) for k in self.keys}
        used_count = {k: 0 for k in self.keys}

        while all(used_count[k] < self.min_len for k in self.keys):
            source = np.random.choice(self.keys)
            if used_count[source] < self.min_len:
                try:
                    sample = next(iterators[source])
                    used_count[source] += 1
                    yield sample
                except StopIteration:
                    continue  # One source exhausted prematurely

    def __len__(self):
        '''
        Total samples = min(dataset sizes) x number of datasets

        For example, if min_len = 12,296 (Crowdsourced),
        and there are 4 datasets => total = 4 x 12,296 = 49,184

        No sample repetition → Better for fair representation, 
        but may under-utilize large datasets like DriverDistraction (66k).
        '''
        return self.min_len * len(self.datasets)


def fused_emotiv_loader(config):
    target_df = pd.read_excel(config.Standard_coord_path, index_col=0)
    input_df = pd.read_excel(config.Emotiv_coord_path, index_col=0)

    dataset_names = ['DREAMER', 'Alpha','DriverDistraction', 'Attention', 'Crowdsource', 'STEW']
    root = config.Emotiv_path

    def load_dataset(name):
        path = os.path.join(root, name, f"{name}.npy")
        data = np.load(path, allow_pickle=True)
        x = data['X_train']
        y = data['Y_train']
        return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    datasets_dict = {name: load_dataset(name) for name in dataset_names}

    # === Select fusion strategy ===
    mode = getattr(config, "emotiv_fusion_mode", "random_concat")  # default
    if mode == "random_concat":
        all_data = ConcatDataset(list(datasets_dict.values()))
        g = torch.Generator().manual_seed(3407)
        train_loader = DataLoader(all_data, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, generator=g)
    elif mode == "multi_source_weighted":
        multi_dataset = MultiSourceEEGDataset(datasets_dict)
        train_loader = DataLoader(multi_dataset, batch_size=config.batch_size,
                                  shuffle=False, num_workers=config.num_workers)
    elif mode == "multi_source_truncated":
        truncated_dataset = TruncatedMultiSourceDataset(datasets_dict)
        train_loader = DataLoader(truncated_dataset, batch_size=config.batch_size,
                                  shuffle=False, num_workers=config.num_workers)
    else:
        raise ValueError(f"Unsupported emotiv_fusion_mode: {mode}")

    # === Validation and Test Data (Merged) ===
    val_all, val_label_all = [], []
    test_all, test_label_all = [], []

    for name in dataset_names:
        path = os.path.join(root, name, f"{name}.npy")
        data = np.load(path, allow_pickle=True)
        val_all.append(data['X_val'])
        val_label_all.append(data['Y_val'])
        test_all.append(data['X_test'])
        test_label_all.append(data['Y_test'])

    val_data = np.concatenate(val_all, axis=0)
    val_label = np.concatenate(val_label_all, axis=0)
    test_data = np.concatenate(test_all, axis=0)
    test_label = np.concatenate(test_label_all, axis=0)

    _, val_loader, test_loader = DataUtils.EmotivLoader(
        val_data, val_label,
        val_data, val_label,
        test_data, test_label,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    sample_batch = next(iter(train_loader))[0]
    return DataObject(train_loader, val_loader, test_loader, input_df, target_df,
                      sample_batch.shape[1], sample_batch.shape[2])



