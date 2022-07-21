import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, required=True, help="exp name for logging")
    parser.add_argument("--model_checkpoint", type=str, default="t5-large", help="Path, url or short name of the model")
    parser.add_argument(
        "--state_converter", type=str, default="mwz", choices=["mwz", "wo_para", "wo_concat", "vanilla", "open_domain"]
    )
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--dev_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for test")
    parser.add_argument("--grad_acc_steps", type=int, default=64, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--slot_loss_ratio", type=float, default=1.0, help="A relative ratio for slot loss.")
    parser.add_argument("--nonslot_loss_ratio", type=float, default=1.0, help="A relative ratio for non-slot loss.")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search during eval")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader.")
    parser.add_argument("--test_num_beams", type=int, default=10, help="Number of beams for beam search during test")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--num_gpus", type=int, default=1, help="how many gpu to use")
    parser.add_argument("--model_name", type=str, default="t5", help="use t5 or bart?")
    parser.add_argument("--fewshot", type=float, default=1.0, help="data ratio for few shot experiment")
    parser.add_argument("--mode", type=str, default="finetune", choices=["finetune", "pretrain"])
    parser.add_argument("--fix_label", default=True)
    parser.add_argument("--except_domain", type=str, choices=["hotel", "train", "restaurant", "attraction", "taxi"])
    parser.add_argument("--only_domain", type=str, choices=["hotel", "train", "restaurant", "attraction", "taxi"])
    parser.add_argument("--version", type=str, default="21", help="version of multiwoz")
    parser.add_argument(
        "--ignore_or", type=bool, default=True, help="ignore slot with value |. if False, consider only previous one."
    )
    parser.add_argument("--save_samples", type=int, default=0, help="save # false case samples.")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="ratio of train data that should be learned to check validation performance",
    )
    parser.add_argument("--dialogue_filter", type=str, default="min", choices=["max", "min"])
    parser.add_argument(
        "--train_control",
        type=str,
        default="none",
        choices=["selective_rough", "selective_exactly", "previous", "none"],
    )
    parser.add_argument("--load_pretrained", type=str, help="Path to the pretrained CD model")
    parser.add_argument("--debug_code", action="store_true")
    parser.add_argument("--eval_loss_only", action="store_true")
    parser.add_argument("--do_train_only", action="store_true")
    parser.add_argument("--do_test_only", action="store_true")
    parser.add_argument(
        "--resume_from_ckpt", type=str,
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_qa_deconverter", action="store_true")
    parser.add_argument("--qa_model_path", type=str)
    parser.add_argument("--balanced_sampling", action="store_true")
    parser.add_argument("--filtered_sampling", action="store_true")
    parser.add_argument("--max_steps", default=-1, type=int, required=True)
    parser.add_argument("--weight_decay", default=0.01, type=float, required=True)
    parser.add_argument("--warmup_steps", default=-1, type=int, required=True)
    parser.add_argument("--len_wgt_ratio", default=1.5, type=float, required=True)
    parser.add_argument("--iteration", default="00", type=str, required=True)

    args = parser.parse_args()

    return args
