from rf2aa.train_multi_EMA import Trainer
from rf2aa.evaluate import Evaluator


def trainer_factory(args, dataset_param, model_param, loader_param, loss_param):
    dataloader_kwargs = {
        "shuffle": args.shuffle_dataloader,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": not args.dont_pin_memory,
    }
    trainer_class = Trainer
    if args.mode == "eval":
        trainer_class = Evaluator
        args.eval = True

    trainer_object = trainer_class(
        model_name=args.model_name,
        n_epoch=args.num_epochs,
        step_lr=args.step_lr,
        lr=args.lr,
        l2_coeff=1.0e-2,
        port=args.port,
        model_param=model_param,
        loader_param=loader_param,
        loss_param=loss_param,
        batch_size=args.batch_size,
        accum_step=args.accum,
        maxcycle=args.maxcycle,
        eval=args.eval,
        interactive=args.interactive,
        out_dir=args.out_dir,
        wandb_prefix=args.wandb_prefix,
        model_dir=args.model_dir,
        dataset_param=dataset_param,
        dataloader_kwargs=dataloader_kwargs,
        debug_mode=args.debug,
        skip_valid=args.skip_valid,
        start_epoch=args.start_epoch,
    )
    return trainer_object
