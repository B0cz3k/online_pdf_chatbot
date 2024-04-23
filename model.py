import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig

TOKEN = os.environ.get('TOKEN')

class ChatModel():
    def __init__(self, model_id, device) -> None:
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_id, token=TOKEN)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id, 
                                                          device_map='auto',
                                                          torch_dtype=torch.float16,
                                                          token=TOKEN)
        self.model.eval()
        self.chat = []

    def generate(self, question: str, context: str, max_new_tokens: int = 250):
        if context == None or context == '':
            prompt = f'Give a detailed answer to the following question: {question}'
        else:
            prompt = f'Here is the context: {context}\nGive a detailed answer to the following question: {question}'
        
        chat = [{'role': 'user', 'content': prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer.encode(
            formatted_prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        answer = answer[len(formatted_prompt):].replace('<eos>', '')
        return answer