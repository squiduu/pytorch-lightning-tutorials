import pytorch_lightning as pl
import nltk
import json
import time
import itertools
import numpy as np
import torch
import rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import AdamW
from torch.nn.modules.loss import CrossEntropyLoss
from transformers.optimization import get_linear_schedule_with_warmup

from heuristic_converter import get_converter
from evaluation import get_acc, get_template_acc

nltk.download("punkt")


class DS2(pl.LightningModule):
    def __init__(self, args, tokenizer, sum_model, qa_model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.sum_model = sum_model
        if self.args["use_qa_deconverter"]:
            self.qa_model = qa_model
        self.lr = args["lr"]
        self.blank = "____"
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.converter = get_converter(args["state_converter"])
        self.evaluator = rouge.Rouge(
            metrics=["rouge-n"],
            max_n=4,
            limit_length=True,
            length_limit=100,
            length_limit_type="words",
            apply_avg=False,
            apply_best=True,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True,
        )

    def training_step(self, batch, batch_idx):
        self.sum_model.train()

        outputs = self.sum_model(
            input_ids=batch["encoder_input"], attention_mask=batch["attention_mask"], labels=batch["decoder_output"],
        )

        # get slot loss
        slot_masked_labels = batch["decoder_output"].clone()
        slot_masked_labels[batch["slot_mask"] * batch["decoder_output"] == 0] = -100
        loss_slot = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), slot_masked_labels.view(-1))
        # get non-slot loss
        non_slot_masked_labels = batch["decoder_output"].clone()
        non_slot_masked_labels[(1 - batch["slot_mask"]) * batch["decoder_output"] == 0] = -100
        loss_nonslot = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), non_slot_masked_labels.view(-1))

        # get slot-weighted loss
        if torch.isnan(torch.sum(loss_slot)):
            loss = loss_nonslot
        else:
            loss = self.args["slot_loss_ratio"] * loss_slot + self.args["nonslot_loss_ratio"] * loss_nonslot

        # get length-weighted loss for long sequence
        # len_wgt = torch.linspace(start=self.args["len_wgt_ratio"], end=1.0, steps=loss.size(-1)).to(
        #     torch.device("cuda")
        # )
        # loss = torch.matmul(loss, len_wgt)
        self.log("loss", value=loss)
        return {"loss": loss, "log": {"train_loss": loss.detach()}}

    def eval_step(self, batch, batch_idx):
        self.sum_model.eval()

        outputs = self.sum_model(
            input_ids=batch["encoder_input"], attention_mask=batch["attention_mask"], labels=batch["decoder_output"],
        )

        # get slot loss
        slot_masked_labels = batch["decoder_output"].clone()
        slot_masked_labels[batch["slot_mask"] * batch["decoder_output"] == 0] = -100
        loss_slot = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), slot_masked_labels.view(-1))
        # get non-slot loss
        non_slot_masked_labels = batch["decoder_output"].clone()
        non_slot_masked_labels[(1 - batch["slot_mask"]) * batch["decoder_output"] == 0] = -100
        loss_nonslot = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), non_slot_masked_labels.view(-1))

        # get slot-weighted loss
        if torch.isnan(torch.sum(loss_slot)):
            loss = loss_nonslot
        else:
            loss = self.args["slot_loss_ratio"] * loss_slot + self.args["nonslot_loss_ratio"] * loss_nonslot

        # get length-weighted loss for long sequence
        # len_wgt = torch.linspace(start=self.args["len_wgt_ratio"], end=1.0, steps=loss.size(-1)).to(
        #     torch.device("cuda")
        # )
        # loss = torch.matmul(loss, len_wgt)
        self.log("loss", value=loss)

        return loss.item()

    def pred_step(self, batch, batch_idx):
        self.sum_model.eval()
        pred_summary_token = self.sum_model.generate(
            batch["encoder_input"], num_beams=self.args["num_beams"], min_length=5, max_length=100, early_stopping=True,
        )

        return {
            "pred_summary_token": pred_summary_token,
            "gold_state": batch["slot_values"],
            "gold_summary": batch["output_text"],
            "eval_slots": batch["eval_slots"],
        }

    def eval_epoch_end(self, outputs):
        res = {}
        res["loss"] = np.mean(outputs)

        print(res)

        return res

    def pred_epoch_end(self, outputs, mode="val"):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}

        pred_summary = [
            self.tokenizer.decode(_sum, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _sum in outputs["pred_summary_token"]
        ]

        if self.args["use_qa_deconverter"]:
            pred_state = self.qa_model.sum_to_state(pred_summary, outputs["eval_slots"])
        else:
            pred_state = [self.converter.sum_to_state(str.split(_sum, sep="summary: ")[-1]) for _sum in pred_summary]

        res = get_acc(pred_state, outputs["gold_state"], outputs["eval_slots"])

        gold_templates = [
            self.converter.state_to_sum(_ds, is_for_template=True, blank=self.blank) for _ds in outputs["gold_state"]
        ]
        template_acc = get_template_acc(pred_summary, gold_templates, self.blank)
        rouge_score = self.evaluator.get_scores(pred_summary, outputs["gold_summary"])["rouge-4"]["f"]
        bleu_score = [
            sentence_bleu([ref.split()], hyp.split(), smoothing_function=SmoothingFunction().method1)
            for ref, hyp in zip(outputs["gold_summary"], pred_summary)
        ]
        res.update(
            {
                "rouge": round(rouge_score, ndigits=4),
                "bleu": round(np.mean(bleu_score), ndigits=4),
                "template_acc": round(template_acc, ndigits=4),
            }
        )

        samples = {
            "gold_summary": outputs["gold_summary"],
            "gold_state": outputs["gold_state"],
            "pred_summary": pred_summary,
            "pred_state": pred_state,
        }
        self.save_samples(samples, f'{str(res["jga"])}_{mode}')

        print(res)

        return res, samples

    def validation_step(self, batch, batch_idx):
        if self.args["eval_loss_only"]:
            return self.eval_step(batch, batch_idx)
        else:
            return self.pred_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        if self.args["eval_loss_only"]:
            res = {f"val_{k}": v for k, v in self.eval_epoch_end(outputs).items()}
        else:
            res = {f"val_{k}": v for k, v in self.pred_epoch_end(outputs, "val")[0].items()}
        self.log_dict(res)

        return res

    def test_step(self, batch, batch_idx):
        return self.pred_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        # get test set outputs
        res, samples = self.pred_epoch_end(outputs=outputs, mode="test")

        # show test JGA as a log
        acc = {f"test_{k}": round(v, ndigits=4) for k, v in res.items()}
        self.log_dict(acc)

        # save the golden and predicted summaries to compare
        samples_res = {"pseudo_labels": samples["pred_state"]}
        test_res = []
        for pred_state in samples_res["pseudo_labels"]:
            test_res.append({"pseudo_labels": pred_state})

        with open(
            file=f"./out/mwoz{self.args['version']}_1pct_{self.args['iteration']}_pseudo.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(obj=test_res, fp=f, indent=4)

        return res

    def configure_optimizers(self):
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.args["weight_decay"],
        #     },
        #     {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        # ]
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args["lr"])

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": get_linear_schedule_with_warmup(
        #             optimizer, self.args["warmup_steps"], self.args["num_training_steps"]
        #         ),
        #         "interval": "step",
        #     },
        # }
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def save_samples(self, samples, name):
        if self.args["save_samples"] > 0:
            output_fields = ["only_domain", "fewshot", "grad_acc_steps", "train_batch_size", "state_converter"]
            output_name = (
                "_".join([str(self.args[k]) for k in output_fields]) + "_" + name + "_" + str(round(time.time()))
            )
            filename = f"./samples_data/{output_name}.json"
            with open(filename, "w") as f:
                json.dump({k: v[: self.args["save_samples"]] for k, v in samples.items()}, f)
