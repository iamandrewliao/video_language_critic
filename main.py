from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import sys
from video_language_critic.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from video_language_critic.util import (
    get_args,
    set_seed_logger,
    init_device,
    freeze_model,
    init_model,
    get_val_test_dataloaders,
    do_train,
    eval_epoch,
)

torch.distributed.init_process_group(backend="nccl")

global logger


def main():
    global logger
    args = get_args()
    
    confirm = input("Do you have the right captions file in dataloader_vlm_retrieval.py? (use raw-captions-fails-relabeled if you are using failures as positives, use raw-captions if using failures as negatives only) [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Exiting.")
        sys.exit(0)
    args = set_seed_logger(args)
    logger = args.logger
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    model = init_model(args, device, n_gpu, args.local_rank)

    freeze_model(model, args)

    test_dataloader, test_length, val_dataloader, val_length = get_val_test_dataloaders(
        args, tokenizer
    )

    if args.local_rank == 0:
        if test_length is not None:
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))
        if val_length is not None:
            logger.info("***** Running val *****")
            logger.info("  Num examples = %d", val_length)

    if args.do_train:
        best_ckpt = do_train(
            args,
            tokenizer,
            model,
            device,
            n_gpu,
            val_dataloader,
            test_dataloader,
        )
    elif args.do_eval:
        if args.local_rank == 0:
            dataloader = val_dataloader if args.eval_on_val else test_dataloader
            eval_epoch(args, model, dataloader, device, n_gpu, save_eval_result=True)


if __name__ == "__main__":
    main()
