"""
Stage 1: Pretrain Qwen-VLM projector.
Freeze LLM + vision encoder, train only projector.
"""
import os, sys, argparse, time, warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_qwen_vlm import QwenVLM, QwenVLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_qwen_utils import (
    get_lr, init_qwen_vlm, qwen_vlm_checkpoint,
    SkipBatchSampler, qwen_collate_fn,
)

warnings.filterwarnings('ignore')


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main_process():
        print(msg)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    actual_iters = iters - start_step
    last_step = start_step
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to(args.device)
        last_step = step

        lr = get_lr(step, actual_iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss
            if loss is None or torch.isnan(loss) or not loss.requires_grad:
                del input_ids, labels, pixel_values, res
                continue
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == actual_iters:
            spend_time = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            cur_lr = optimizer.param_groups[-1]['lr']
            eta = spend_time / max(step - start_step, 1) * (actual_iters - step) // 60
            log(f'Pretrain [{epoch+1}/{args.epochs}]({step}/{actual_iters}) '
                f'loss:{cur_loss:.4f} lr:{cur_lr:.6f} eta:{eta:.0f}min')
            if wandb and is_main_process():
                wandb.log({'loss': cur_loss, 'lr': cur_lr})

        if (step % args.save_interval == 0 or step == actual_iters) and is_main_process():
            qwen_vlm_checkpoint(
                vlm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, epoch=epoch, step=step,
                save_dir=args.checkpoint_dir, scaler=scaler, wandb=wandb,
            )

        del input_ids, labels, pixel_values, res, loss

    # Handle remaining gradients
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-VLM Pretrain")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_i2t.parquet")
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--from_weight", default="none", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--save_weight", default="pretrain_qwen", type=str)
    parser.add_argument("--checkpoint_dir", default="../checkpoints", type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Qwen-Pretrain")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)

    vlm_config = QwenVLMConfig(max_seq_len=args.max_seq_len)

    # Resume
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = qwen_vlm_checkpoint(
            vlm_config, weight=args.save_weight, save_dir=args.checkpoint_dir
        )

    model, tokenizer = init_qwen_vlm(
        from_weight=args.from_weight, device=args.device, freeze_llm=2,
    )

    train_ds = VLMDataset(
        args.data_path, tokenizer, preprocess=model.processor,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
        max_length=vlm_config.max_seq_len,
    )

    # Data subsampling
    full_size = len(train_ds)
    if args.data_ratio < 1.0:
        torch.manual_seed(42)
        sample_size = int(full_size * args.data_ratio)
        subset_indices = torch.randperm(full_size)[:sample_size].tolist()
        log(f'Data: {sample_size}/{full_size} samples ({args.data_ratio*100:.0f}%)')
    else:
        subset_indices = list(range(full_size))
        log(f'Data: {full_size} samples (full)')

    # Wandb init
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project,
                   name=f'pretrain-qwen-bs{args.batch_size}-lr{args.learning_rate}')
        log('swanlab enabled')

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        if ckp_data.get('optimizer'):
            optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data.get('scaler', {}))
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = set()
        model = DistributedDataParallel(model, device_ids=[local_rank])

    log(f'Pretrain: {args.epochs} epochs, {args.batch_size} batch, lr={args.learning_rate}')

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        torch.manual_seed(42 + epoch)
        indices = torch.randperm(len(subset_indices)).tolist()
        indices = [subset_indices[i] for i in indices]  # map to original indices
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        iters = len(indices) // args.batch_size
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
            pin_memory=True, collate_fn=qwen_collate_fn,
        )
        if skip > 0:
            log(f'Epoch {epoch+1}: skip {skip} steps, start from {skip+1}')
            train_epoch(epoch, loader, iters + skip, skip, wandb)
        else:
            train_epoch(epoch, loader, iters, 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
