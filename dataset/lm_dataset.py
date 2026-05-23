import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model.model_vlm import MiniMindVLM
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512, image_special_token='<|image_pad|>', image_token_len=64):
        super().__init__()
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len

        # Qwen uses <|im_start|>assistant\n, MiniMind uses <|bos|>assistant\n
        if tokenizer.bos_token is None:
            # Qwen-style tokenizer
            self.bos_id = self.tokenizer.encode('\n<|im_start|>assistant\n', add_special_tokens=False)
            self.eos_id = self.tokenizer.encode('<|im_end|>\n', add_special_tokens=False)
        else:
            self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
            self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) if turn.get('role') != 'system' else turn['content']
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _get_prefix_len(self, conversations):
        """Get token count of everything before the last assistant response."""
        prefix_convs = [c for c in conversations if c.get('role') != 'assistant']
        if not prefix_convs:
            return 0
        # Replace <image> placeholder with pad tokens in prefix conversations
        prefix_convs = [
            {**c, 'content': c['content'].replace('<image>', self.image_special_token)}
            if c.get('role') != 'system' else c
            for c in prefix_convs
        ]
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        prefix_prompt = self.tokenizer.apply_chat_template(
            prefix_convs, tokenize=False, add_generation_prompt=True, tools=tools
        )
        prefix_prompt = post_processing_chat(prefix_prompt)
        return len(self.tokenizer(prefix_prompt).input_ids)

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]

        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)

        # Tokenize full conversation
        full_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        prefix_len = self._get_prefix_len(conversations)

        # Labels: -100 for prompt prefix, copy input_ids for assistant response
        labels = [-100] * min(prefix_len, self.max_length)
        response_ids = full_ids[prefix_len:self.max_length]
        labels += response_ids

        # Pad to max_length
        input_ids = full_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(full_ids))
        labels = (labels + [-100] * (self.max_length - len(labels)))[:self.max_length]

        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data

# 测试parquet数据读取和可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        pf = pq.ParquetFile(path); n = pf.num_row_groups; t = pa.concat_tables([pf.read_row_group(i * n // 5).slice(0, 1) for i in range(5)]); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            img_data = t['image_bytes'][i].as_py(); img_data = img_data[0] if isinstance(img_data, list) else img_data
            ax[i].imshow(Image.open(io.BytesIO(img_data))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{pf.metadata.num_rows}条')
