import itertools
import torch
from torch.optim.adamw import AdamW
import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup


class Losey(pl.LightningModule):
    def __init__(self, tokenizer: RobertaTokenizer, roberta: RobertaForSequenceClassification, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.tokenizer = tokenizer
        self.roberta = roberta
        self.accuracy = Accuracy()

    def training_step(self, batch, batch_idx):
        outputs = self.roberta.forward(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
        )
        self.log(name="train_loss", value=outputs.loss, on_epoch=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        self.roberta.eval()
        outputs = self.roberta.forward(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
        )
        _, val_pred = torch.max(outputs.logits, dim=1)

        return {"val_pred": val_pred, "val_labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}
        val_acc = self.accuracy(outputs["val_pred"], outputs["val_labels"])

        self.log(name="val_acc", value=val_acc)

        return {"val_acc": val_acc, "val_pred": outputs["val_pred"], "val_labels": outputs["val_labels"]}

    def test_step(self, batch, batch_idx):
        self.roberta.eval()
        outputs = self.roberta.forward(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
        )
        _, test_pred = torch.max(outputs.logits, dim=1)

        return {"test_pred": test_pred, "test_labels": batch["labels"]}

    def test_epoch_end(self, outputs):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}
        test_acc = self.accuracy(outputs["val_pred"], outputs["val_labels"])

        self.log(name="test_acc", value=test_acc)

        return {"test_acc": test_acc, "test_pred": outputs["test_pred"], "test_labels": outputs["test_labels"]}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_linear_schedule_with_warmup(
                    optimizer, self.args.warmup_steps, self.args.num_training_steps
                ),
                "interval": "step",
            },
        }
