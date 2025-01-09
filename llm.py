# import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LanguageModel:
    def generate(self, prompt):
        """
        根据提示生成响应。
        :param prompt: 输入的提示文本。
        :return: 生成的响应。
        """
        raise NotImplementedError
    
class OpenAILanguageModel(LanguageModel):
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "你是一个游戏助手。"},
                      {"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    
class QwenLanguageModel(LanguageModel):
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    
    def generate(self, prompt, max_length=300, temperature=0.2):
        messages = [
            {"role": "system", "content": "你是一个游戏助手。"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1000
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response