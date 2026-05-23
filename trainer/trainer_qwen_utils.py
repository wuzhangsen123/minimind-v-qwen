"""
Training utilities for Qwen-based VLM.
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from model.model_qwen_vlm import QwenVLM, QwenVLMConfig


def get_model_params(model, ignore_patterns=['vision_encoder']):
    def should_count(n):
        return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    trainable = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and should_count(n)) / 1e6
    print(f'Model Params: {total:.1f}M, Trainable: {trainable:.1f}M')
    return total, trainable


def get_lr(current_step, total_steps, lr):
    """Cosine decay with warmup floor."""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_qwen_vlm(from_weight='none', device='cuda',
                  llm_path='./model/qwen-0.5b',
                  vision_path='./model/siglip2-base-p32-256-ve',
                  save_dir='../out', freeze_llm=2):
    """Initialize Qwen VLM model with optional pretrained weights."""

    # Absolute paths — resolve relative to project root (parent of trainer/)
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(llm_path):
        llm_path = os.path.join(proj_root, llm_path.lstrip('./'))
    if not os.path.isabs(vision_path):
        vision_path = os.path.join(proj_root, vision_path.lstrip('./'))
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(proj_root, save_dir.lstrip('./'))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

    # Add image special token
    image_special_token = '<|image_pad|>'
    special_tokens = {'additional_special_tokens': [image_special_token]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    image_token_id = tokenizer.convert_tokens_to_ids(image_special_token)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    # Init model
    config = QwenVLMConfig(
        image_special_token=image_special_token,
        image_token_len=64,
        image_hidden_size=768,
        projector_hidden=1536,
        llm_path=llm_path,
        vision_path=vision_path,
    )
    model = QwenVLM(config, vision_path=vision_path, llm_path=llm_path)
    model.image_token_id = image_token_id

    # Resize LLM embeddings for new tokens
    model.llm.resize_token_embeddings(len(tokenizer))

    # Load pretrained projector weights
    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}.pth'
        if os.path.exists(weight_path):
            ckp = torch.load(weight_path, map_location='cpu')
            if 'projector' in ckp:
                model.projector.load_state_dict(ckp['projector'])
                print(f'Loaded projector from {weight_path}')
            else:
                # Try loading full state dict
                model.load_state_dict(ckp, strict=False)
                print(f'Loaded full weights from {weight_path}')

    # Freeze strategy
    # 0: all LLM trainable
    # 1: first + last LLM layers + projector
    # 2: only projector (LLM fully frozen)
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Projector always trainable
    for param in model.projector.parameters():
        param.requires_grad = True

    if freeze_llm == 0:
        for name, param in model.llm.named_parameters():
            param.requires_grad = True
    elif freeze_llm == 1:
        n_layers = model.llm.config.num_hidden_layers
        last_idx = n_layers - 1
        for name, param in model.llm.named_parameters():
            if 'layers.0.' in name or f'layers.{last_idx}.' in name:
                param.requires_grad = True
    # freeze_llm == 2: LLM stays fully frozen

    get_model_params(model)

    return model.to(device), tokenizer


def qwen_vlm_checkpoint(config, weight='pretrain_qwen', model=None,
                        optimizer=None, epoch=0, step=0, save_dir='../checkpoints',
                        **kwargs):
    """Save/Load checkpoint for Qwen VLM."""
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{weight}.pth'
    resume_path = f'{save_dir}/{weight}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # Save projector weights
        proj_state = {k: v.half().cpu() for k, v in raw_model.projector.state_dict().items()}
        torch.save({'projector': proj_state, 'config': config.to_dict()}, ckp_path)

        # Save full resume checkpoint
        state_dict = raw_model.state_dict()
        resume_data = {
            'model': {k: v.half().cpu() for k, v in state_dict.items()
                      if not k.startswith('vision_encoder.')},
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'step': step,
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(value, 'state_dict'):
                resume_data[key] = value.state_dict()
            elif not isinstance(value, (type, __import__('types').ModuleType, __import__('types').FunctionType)):
                resume_data[key] = value

        torch.save(resume_data, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        # Load mode
        if os.path.exists(resume_path):
            return torch.load(resume_path, map_location='cpu')
        return None


class SkipBatchSampler:
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)


def qwen_collate_fn(batch):
    """Custom collate to handle dict pixel_values from SigLIP processor."""
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values
