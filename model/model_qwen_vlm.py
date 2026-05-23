"""
MiniMind-V with Qwen2.5-0.5B as LLM backbone.
Architecture: SigLIP2 (frozen) → Projector (trainable) → Qwen2.5 (frozen/LoRA)
"""
import os
import torch
import torch.nn as nn
import warnings
from typing import Optional
from PIL import Image
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    SiglipVisionModel, SiglipImageProcessor,
    PreTrainedModel, GenerationMixin, PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

warnings.filterwarnings('ignore')


class QwenVLMConfig(PretrainedConfig):
    model_type = "qwen-vlm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_special_token = kwargs.get('image_special_token', '<|image_pad|>')
        self.image_token_len = kwargs.get('image_token_len', 64)
        self.image_hidden_size = kwargs.get('image_hidden_size', 768)
        self.projector_hidden = kwargs.get('projector_hidden', 1536)
        self.llm_path = kwargs.get('llm_path', './model/qwen-0.5b')
        self.vision_path = kwargs.get('vision_path', './model/siglip2-base-p32-256-ve')


class MMVisionProjector(nn.Module):
    """768 → projector_hidden → llm_hidden with LayerNorm + GELU"""
    def __init__(self, in_dim=768, hidden_dim=1536, out_dim=896):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class QwenVLM(PreTrainedModel, GenerationMixin):
    config_class = QwenVLMConfig
    _no_split_modules = []

    def __init__(self, config=None, vision_path=None, llm_path=None):
        if config is None:
            config = QwenVLMConfig()
        super().__init__(config)

        vision_path = vision_path or config.vision_path
        llm_path = llm_path or config.llm_path

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        self.vision_encoder, self.processor = self._load_vision_encoder(vision_path)
        self.projector = MMVisionProjector(
            in_dim=config.image_hidden_size,
            hidden_dim=config.projector_hidden,
            out_dim=self.llm.config.hidden_size,
        )
        self.image_token_len = config.image_token_len

        # Will be set after tokenizer is loaded
        self._image_token_id = None

    @staticmethod
    def _load_vision_encoder(path):
        if not os.path.exists(path):
            return None, None
        model = SiglipVisionModel.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        processor = SiglipImageProcessor.from_pretrained(path)
        return model.eval(), processor

    @property
    def image_token_id(self):
        return self._image_token_id

    @image_token_id.setter
    def image_token_id(self, value):
        self._image_token_id = value

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        return processor(images=image, return_tensors="pt")

    def get_vision_features(self, pixel_values):
        if self.vision_encoder is None:
            return None

        # Extract pixel_values tensor from dict if needed
        if hasattr(pixel_values, 'keys'):
            pv = pixel_values['pixel_values']
        else:
            pv = pixel_values

        # Handle dimensions: SigLIP expects [B, C, H, W] (4D)
        # Dataset/collate may produce [B, num_images, C, H, W] (5D)
        if pv.dim() == 5:
            B, N = pv.shape[:2]
            pv = pv.flatten(0, 1)  # [B*N, C, H, W]
        else:
            B, N = pv.shape[0], 1

        vis_device = next(self.vision_encoder.parameters()).device
        pv = pv.to(dtype=torch.bfloat16, device=vis_device)

        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=pv)

        vis = self.projector(outputs.last_hidden_state)  # [B*N, 64, hidden]
        if N > 1:
            vis = vis.view(B, N, self.image_token_len, -1)
        return vis

    def _splice_embeddings(self, text_embeds, input_ids, vision_embeds):
        """Replace image token embeddings with vision features."""
        if vision_embeds is None or self.image_token_id is None:
            return text_embeds

        B = text_embeds.shape[0]
        tid = self.image_token_id

        if vision_embeds.dim() == 3:
            vision_embeds = vision_embeds.unsqueeze(1)  # [B, 1, 64, H]

        out_list = []
        for b in range(B):
            emb = text_embeds[b]
            ids = input_ids[b].tolist()
            vi = 0
            i = 0
            while i < len(ids):
                if ids[i] == tid:
                    s = i
                    while i < len(ids) and ids[i] == tid:
                        i += 1
                    n = i - s
                    if vi < vision_embeds.shape[1]:
                        vt = vision_embeds[b, vi, :n, :].to(dtype=emb.dtype, device=emb.device)
                        emb = torch.cat([emb[:s], vt, emb[i:]], dim=0)
                        vi += 1
                else:
                    i += 1
            out_list.append(emb)

        # Pad to max length
        max_len = max(o.shape[0] for o in out_list)
        padded = torch.zeros(B, max_len, text_embeds.shape[-1],
                             dtype=text_embeds.dtype, device=text_embeds.device)
        for b, emb in enumerate(out_list):
            padded[b, :emb.shape[0]] = emb
        return padded

    def forward(self, input_ids, attention_mask=None, pixel_values=None,
                labels=None, past_key_values=None, use_cache=False, **kwargs):
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        if pixel_values is not None and past_key_values is None:
            vision_embeds = self.get_vision_features(pixel_values)
            text_embeds = self._splice_embeddings(text_embeds, input_ids, vision_embeds)

        # If using past_key_values during generation, only pass new tokens
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
        )
        return outputs

    def prepare_inputs_for_generation(self, input_ids, pixel_values=None,
                                      past_key_values=None, attention_mask=None, **kwargs):
        # Only pass pixel_values on first generation step
        if past_key_values is not None:
            pixel_values = None
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'past_key_values': past_key_values,
            'attention_mask': attention_mask,
            'use_cache': True,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm._reorder_cache(past_key_values, beam_idx)

    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save({
            'projector': self.projector.state_dict(),
            'config': self.config.to_dict(),
        }, os.path.join(save_directory, 'projector_weights.pth'))

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)
