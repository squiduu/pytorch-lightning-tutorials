import os
import torch
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger
from transformers import RobertaForSequenceClassification
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

from myxai.my_datamodule import CustomDataModule
from myxai.my_config import get_args
from myxai.my_model import Losey


def main(args):
    # make or set the log and config files
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.exp_dir + "log/", exist_ok=True)
    args.log_dir = args.exp_dir + "log/"
    os.makedirs(args.exp_dir + "config/", exist_ok=True)
    args.config_dir = args.exp_dir + "config/"

    # set logger
    logger = CSVLogger(save_dir=args.log_dir, name=f"{args.exp_dir}")
    # record hparams
    logger.log_hyperparams(args)

    # set seed
    seed_everything(args.seed)

    # get pre-trained tokenizer
    print(f" ----- Get pre-trained tokenizer : {args.model_name} ----- ")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    # get pre-trained backbone
    print(f" ----- Get pre-trained backbone : {args.model_name}----- ")
    roberta = RobertaForSequenceClassification.from_pretrained(args.model_name)

    # get dataloaders for train and val
    print(" ----- Get customized datamodule ----- ")
    custom_datamodule = CustomDataModule(data_dir=args.data_dir, tokenizer=tokenizer, args=args)

    print(f" ----- Get pre-trained model : {args.model_name} ----- ")
    # get runner for fine-tuning
    losey = Losey(tokenizer=tokenizer, roberta=roberta, args=args)

    # set model checkpoint callback
    if not args.do_test_only:
        es_callback = EarlyStopping(monitor="val_acc", patience=args.patience, verbose=True, mode="max")
        ckpt_callback = ModelCheckpoint(
            dirpath=args.exp_dir,
            filename="{epoch:02d}-{val_acc:.4f}",
            monitor="val_acc",
            verbose=True,
            save_top_k=1,
            mode="max",
        )
        callbacks = [es_callback, ckpt_callback]
    else:
        callbacks = None

    # set trainer
    pl_trainer = Trainer(
        logger=logger if not args.do_test_only else None,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        strategy=args.strategy,
        precision=args.precision,
    )

    if not args.do_test_only:
        print(" ----- Start training ----- ")
        custom_datamodule.setup(stage="fit")
        pl_trainer.fit(model=losey, datamodule=custom_datamodule)

    if not args.do_train_only:
        print(" ----- Start testing ----- ")
        custom_datamodule.setup(stage="test")
        if not args.ckpt_path:
            pl_trainer.test(
                model=losey, datamodule=custom_datamodule, ckpt_path=ckpt_callback.best_model_path, verbose=True
            )
        else:
            pl_trainer.test(model=losey, datamodule=custom_datamodule, ckpt_path=args.ckpt_path, verbose=True)

    logger.save()


if __name__ == "__main__":
    args = get_args()
    main(args)

    torch.cuda.empty_cache()
