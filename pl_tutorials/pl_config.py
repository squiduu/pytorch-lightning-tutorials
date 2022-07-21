import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # set required arguments for dataloading
    parser.add_argument("--data_dir", default="./data/", type=str, required=True)
    parser.add_argument("--batch_size_per_gpu", default=1, type=int, required=True)
    parser.add_argument("--num_workers", default=4, type=int, required=True)
    parser.add_argument("--seed", default=42, type=int, required=True)

    # set required arguments for modeling
    parser.add_argument("--model_name", default="roberta-base", type=str, required=True)

    # set required arguments for training
    parser.add_argument("--strategy", default="ddp", type=str, required=True)
    parser.add_argument("--lr", default=2e-5, type=float, required=True)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, required=True)
    parser.add_argument("--weight_decay", default=0.01, type=float, required=True)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, required=True)
    parser.add_argument("--warmup_steps", default=-1, type=int, required=True)
    parser.add_argument("--max_steps", default=-1, type=int, required=True)
    parser.add_argument("--max_epochs", default=100, type=int, required=True)
    parser.add_argument("--exp_dir", default="./bios00/", type=str, required=True)
    parser.add_argument("--gpus", default=1, type=int, required=True)
    parser.add_argument("--patience", default=10, type=int, required=True)
    parser.add_argument("--ckpt_path", default="./ckpt/", type=str, required=False)
    parser.add_argument("--do_test_only", action="store_true")
    parser.add_argument("--do_train_only", action="store_true")
    parser.add_argument("--precision", default=16, type=int, required=True)

    args = parser.parse_args()

    return args
