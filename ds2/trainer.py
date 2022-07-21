import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from data_loader_v1 import prepare_data
from config import get_args
from modeling import DS2


def fine_tune(args):
    args = vars(args)
    seed_everything(args["seed"])
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args["model_checkpoint"])
    model = AutoModelForSeq2SeqLM.from_pretrained(args["model_checkpoint"])

    dataloaders, _ = prepare_data(args, tokenizer)
    print("Created dataloaders")

    # set max training steps
    if args["max_steps"] > 0:
        args["num_training_steps"] = args["max_steps"]
        args["num_epochs"] = args["max_steps"] // len(dataloaders["train"]) + 1
    else:
        args["num_training_steps"] = int(len(dataloaders["train"]) * args["num_epochs"] / args["num_gpus"])

    # set warm-up steps
    if args["warmup_steps"] < 0:
        args["warmup_steps"] = int(len(dataloaders["train"]) / args["num_gpus"] * args["num_epochs"] * 0.2)

    # logging
    exp_name = args["exp_name"]
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    log_path = f"./logs/{exp_name}"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if args["load_pretrained"]:
        pretrain_ckpt_path = os.path.join(args["load_pretrained"], "pretrain")
        pretrain_ckpts = [_ckpt for _ckpt in os.listdir(pretrain_ckpt_path) if ".ckpt" in _ckpt]
        assert len(pretrain_ckpts) == 1
        ckpt = pretrain_ckpts[0]
        print("load pretrained model from: ", os.path.join(pretrain_ckpt_path, ckpt))
        dst_model = DS2.load_from_checkpoint(
            os.path.join(pretrain_ckpt_path, ckpt),
            args=args,
            tokenizer=tokenizer,
            sum_model=model,
            qa_model=None,
        )
    else:
        dst_model = DS2(args, tokenizer, model, None)

    print("Created Model")

    dir_path = os.path.join(log_path, args["mode"])
    if not args["do_test_only"]:
        earlystopping_callback = EarlyStopping(
            monitor="val_loss" if args["eval_loss_only"] else "val_jga",
            # min_delta=0.00,
            patience=args["patience"],
            verbose=False,
            mode="min" if args["eval_loss_only"] else "max",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_path,
            filename="{val_loss:.3f}" if args["eval_loss_only"] else "{val_jga:.3f}",
            save_top_k=1,
            monitor="val_loss" if args["eval_loss_only"] else "val_jga",
            mode="min" if args["eval_loss_only"] else "max",
        )
        callbacks = [earlystopping_callback, checkpoint_callback]
    else:
        callbacks = None

    # profiler = PyTorchProfiler(export_to_chrome=True)
    trainer = Trainer(
        accumulate_grad_batches=args["grad_acc_steps"],
        gradient_clip_val=args["max_norm"],
        max_epochs=args["num_epochs"],
        callbacks=callbacks,
        gpus=args["num_gpus"],
        deterministic=True,
        strategy="ddp",
        val_check_interval=args["val_check_interval"],
        logger=CSVLogger(dir_path, f"seed_{args['seed']}") if not args["do_test_only"] else None,
        resume_from_checkpoint=args["resume_from_ckpt"],
        # limit_val_batches=0.05,
    )

    if not args["do_test_only"]:
        trainer.fit(dst_model, dataloaders["train"], dataloaders["dev"])

    if not args["do_train_only"]:
        print("test start...")
        # evaluate model
        args["num_beams"] = args["test_num_beams"]
        if args["do_test_only"]:
            ckpts = [_ckpt for _ckpt in os.listdir(dir_path) if ".ckpt" in _ckpt]
            assert len(ckpts) == 1
            ckpt = ckpts[0]
            print("load pretrained model from: ", os.path.join(dir_path, ckpt))
            ckpt_path = os.path.join(dir_path, ckpt)
        else:
            ckpt_path = checkpoint_callback.best_model_path

        dst_model = DS2.load_from_checkpoint(
            checkpoint_path=ckpt_path, args=args, tokenizer=tokenizer, sum_model=model, qa_model=None
        )
        # trainer.test(dst_model, dataloaders["test"])
        trainer.test(dst_model, dataloaders["test"])


if __name__ == "__main__":
    args = get_args()
    fine_tune(args)
