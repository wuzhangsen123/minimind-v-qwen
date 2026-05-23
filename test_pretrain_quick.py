"""Quick test of pretrained Qwen-VLM projector."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image
from transformers import AutoTokenizer
from model.model_qwen_vlm import QwenVLM, QwenVLMConfig

device = "cuda:0"
torch.set_grad_enabled(False)

config = QwenVLMConfig()
tokenizer = AutoTokenizer.from_pretrained(config.llm_path, trust_remote_code=True)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})

model = QwenVLM(config)
model.image_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
model.llm.resize_token_embeddings(len(tokenizer))

ckp = torch.load('checkpoints/pretrain-2e.pth', map_location='cpu')
model.projector.load_state_dict(ckp['projector'])
print('Loaded pretrained projector\n')

model = model.to(device).eval()

image_dir = 'dataset/eval_images'
for img_file in sorted(os.listdir(image_dir)):
    img_path = os.path.join(image_dir, img_file)
    image = Image.open(img_path).convert('RGB')
    pixel_values = model.image2tensor(image, model.processor)

    image_tokens = '<|image_pad|>' * 64
    prompt = f'<|im_start|>user\n{image_tokens}请描述这张图片。<|im_end|>\n<|im_start|>assistant\n'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # Manually splice vision embeddings into text embeddings for first step
    text_embeds = model.llm.model.embed_tokens(input_ids)
    vision_embeds = model.get_vision_features(pixel_values)
    inputs_embeds = model._splice_embeddings(text_embeds, input_ids, vision_embeds)

    output = model.llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if 'assistant\n' in response:
        response = response.split('assistant\n')[-1].strip()

    print(f'[{img_file}]')
    print(f'  {response}')
    print()
