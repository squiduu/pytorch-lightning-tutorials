import json
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer


class CustomDataModule(LightningDataModule):
    def __init__(self, data_dir: str, tokenizer: RobertaTokenizer, args):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.args = args

    def load_dataset(self, run_type: str):
        if run_type == "train":
            with open(self.data_dir + "train.json", mode="r", encoding="utf-8") as f:
                raw_data = json.load(f)
        elif run_type == "val":
            with open(self.data_dir + "val.json", mode="r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            with open(self.data_dir + "test.json", mode="r", encoding="utf-8") as f:
                raw_data = json.load(f)

        return raw_data

    def preprocess_dataset(self, run_type: str) -> list:
        print(f" ----- Read data from {self.data_dir} ----- ")

        dataset = []
        raw_data = self.load_dataset(run_type=run_type)

        return dataset

    def setup(self, stage: str = None):
        if stage == "fit":
            self.dataset_train = Dataset(self.preprocess_dataset(data_dir=self.data_dir, run_type="train"))
            self.dataset_val = Dataset(self.preprocess_dataset(data_dir=self.data_dir, run_type="val"))
        if stage == "test":
            self.dataset_test = Dataset(self.preprocess_dataset(data_dir=self.data_dir, run_type="test"))

    def my_collate_fn(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.args.batch_size_per_gpu,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.my_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            collate_fn=self.my_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            collate_fn=self.my_collate_fn,
            pin_memory=True,
        )
