import logging
import time

import torch
from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter


def _loss_weights(args):
    return {
        "loss_itc": 1.0,
        "loss_clfe": args.top_weight,
        "loss_slfe": args.pool_weight,
        "loss_gfa": args.pred_weight,
    }


def do_train(
    start_epoch,
    args,
    model,
    train_loader,
    evaluator,
    optimizer,
    scheduler,
    checkpointer,
    best_instance_mr=float("-inf"),
    best_epoch=0,
):
    logger = logging.getLogger("Herb-LEGA.train")
    logger.info("Start training")
    writer = None
    if get_rank() == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "tensorboard is required for training. Install requirements.txt first."
            ) from exc
        writer = SummaryWriter(log_dir=args.output_dir)

    weights = _loss_weights(args)
    meter_names = ["loss"] + list(weights)
    meters = {name: AverageMeter() for name in meter_names}

    for epoch in range(start_epoch, args.num_epoch + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for iteration, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(args.device, non_blocking=True)
                for key, value in batch.items()
            }
            ret = model(batch, epoch)
            available_losses = {
                name: ret[name] for name in weights if name in ret
            }
            if not available_losses:
                raise RuntimeError("No loss was produced. Check --loss_names.")
            total_loss = sum(weights[name] * value for name, value in available_losses.items())

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            batch_size = batch["images"].shape[0]
            meters["loss"].update(total_loss.detach().item(), batch_size)
            for name, value in available_losses.items():
                meters[name].update(value.detach().item(), batch_size)

            if iteration % args.log_period == 0 or iteration == len(train_loader):
                parts = [
                    f"Epoch[{epoch}/{args.num_epoch}]",
                    f"Iter[{iteration}/{len(train_loader)}]",
                ]
                parts.extend(
                    f"{name}: {meter.avg:.4f}"
                    for name, meter in meters.items()
                    if meter.count > 0
                )
                parts.append(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                logger.info(", ".join(parts))

        scheduler.step()
        synchronize()

        if get_rank() == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d finished in %.1fs (%.3fs/batch)",
                epoch,
                elapsed,
                elapsed / max(len(train_loader), 1),
            )
            if writer is not None:
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
                for name, meter in meters.items():
                    if meter.count > 0:
                        writer.add_scalar(f"train/{name}", meter.avg, epoch)
                if "temperature_ori" in ret:
                    writer.add_scalar(
                        "train/temperature_ori", ret["temperature_ori"].item(), epoch
                    )
                if "temperature_gen" in ret:
                    writer.add_scalar(
                        "train/temperature_gen", ret["temperature_gen"].item(), epoch
                    )

        if epoch % args.eval_period == 0 and get_rank() == 0:
            logger.info("Validation Results - Epoch %d", epoch)
            results = evaluator.eval(model, epoch)
            instance_mr = results["INSTANCE"]["mR"]
            if writer is not None:
                writer.add_scalar("val/INSTANCE_mR", instance_mr, epoch)
            if instance_mr > best_instance_mr:
                best_instance_mr = instance_mr
                best_epoch = epoch
                checkpointer.save(
                    "best",
                    epoch=epoch,
                    best_instance_mr=best_instance_mr,
                )
                logger.info(
                    "New best checkpoint: validation INSTANCE mR %.3f at epoch %d",
                    best_instance_mr,
                    epoch,
                )
            torch.cuda.empty_cache()
        synchronize()

    if get_rank() == 0:
        logger.info(
            "Best validation INSTANCE mR: %.3f at epoch %d",
            best_instance_mr,
            best_epoch,
        )
    if writer is not None:
        writer.close()
    return {"best_instance_mr": best_instance_mr, "best_epoch": best_epoch}
