"""
MMBench evaluation for QwenVLM.
Downloads MMBench_DEV_CN and scores on multiple-choice questions.
"""
import os, sys, io, json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from trainer.trainer_qwen_utils import init_qwen_vlm
from trainer.train_sft_qwen import apply_lora

device = "cuda:0"
torch.set_grad_enabled(False)

# Load SFT model
model, tokenizer = init_qwen_vlm(
    from_weight='pretrain-2e', device=device, freeze_llm=2,
)
model = apply_lora(model, lora_rank=16, lora_alpha=32, lora_dropout=0.05)
ckp = torch.load('checkpoints/sft-lora-1e-resume.pth', map_location='cpu')
model.load_state_dict(ckp['model'], strict=False)
model = model.to(device).eval()
print("Model loaded\n")

# Download dataset
data_path = "dataset/mmbench_dev_cn.parquet"
if not os.path.exists(data_path):
    print("Downloading MMBench_DEV_CN...")
    from huggingface_hub import hf_hub_download
    os.makedirs("dataset", exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="lmms-lab/MMBench",
        filename="cn/dev-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir="dataset",
        local_dir_use_symlinks=False,
    )
    os.rename(downloaded, data_path)
    print("Downloaded.\n")

df = pd.read_parquet(data_path)
print(f"Total: {len(df)} questions\n")

correct, total = 0, 0
results = []
letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

for _, row in tqdm(df.iterrows(), total=len(df)):
    # Parse image
    img_data = row['image']
    if isinstance(img_data, dict):
        img_bytes = img_data['bytes']
    else:
        img_bytes = img_data
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        pixel_values = model.image2tensor(image, model.processor)
    except Exception as e:
        continue

    question = row['question']
    hint = row.get('hint', '')

    # Build options (some may be NaN for fewer-than-4 choices)
    option_labels = []
    option_lines = []
    for letter in ['A', 'B', 'C', 'D']:
        val = row[letter]
        if pd.notna(val) and str(val).strip():
            option_labels.append(letter)
            option_lines.append(f"{letter}. {val}")

    options_str = '\n'.join(option_lines)
    gt_letter = row['answer']
    if gt_letter not in option_labels:
        continue

    if pd.notna(hint) and str(hint).strip():
        hint_text = f"提示：{hint}\n"
    else:
        hint_text = ""

    prompt = (
        f"<|im_start|>user\n{'<|image_pad|>' * 64}{hint_text}{question}\n{options_str}"
        f"\n请直接回答选项字母{'、'.join(option_labels)}。<|im_end|>\n<|im_start|>assistant\n"
    )

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    text_embeds = model.llm.get_input_embeddings()(input_ids)
    vision_embeds = model.get_vision_features(pixel_values)
    inputs_embeds = model._splice_embeddings(text_embeds, input_ids, vision_embeds)

    output = model.llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if 'assistant\n' in response:
        response = response.split('assistant\n')[-1].strip()

    # Extract answer letter
    pred_letter = '?'
    for ch in response.upper():
        if ch in option_labels:
            pred_letter = ch
            break

    total += 1
    if pred_letter == gt_letter:
        correct += 1

    results.append({
        'index': row.get('index', _),
        'category': row['category'],
        'question': str(question)[:80],
        'pred': pred_letter,
        'gt': gt_letter,
        'correct': pred_letter == gt_letter,
    })

    if total % 200 == 0:
        acc = correct / total * 100
        tqdm.write(f"  [{total}] Acc: {acc:.1f}% ({correct}/{total})")

acc = correct / total * 100
print(f"\n{'='*50}")
print(f"MMBench_DEV_CN Accuracy: {acc:.2f}% ({correct}/{total})")
print(f"{'='*50}")

# Per-category
cat = {}
for r in results:
    c = r['category']
    if c not in cat:
        cat[c] = {'correct': 0, 'total': 0}
    cat[c]['total'] += 1
    cat[c]['correct'] += r['correct']

print("\nCategory breakdown:")
for c in sorted(cat.keys()):
    print(f"  {c}: {cat[c]['correct']}/{cat[c]['total']} = {cat[c]['correct']/cat[c]['total']*100:.1f}%")

os.makedirs('out', exist_ok=True)
with open('out/mmbench_results.json', 'w') as f:
    json.dump({'accuracy': acc, 'correct': correct, 'total': total, 'results': results}, f, ensure_ascii=False, indent=2)
print(f"\nSaved to out/mmbench_results.json")
