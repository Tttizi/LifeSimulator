import os
import sys
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import random
from llm import OpenAILanguageModel, QwenLanguageModel
import json
from tqdm import tqdm

def replace_fullwidth_symbols(text):
    """
    将全角符号替换为半角符号。
    """
    fullwidth_to_halfwidth = {
        "：": ":",  # 全角冒号 -> 半角冒号
        "，": ",",  # 全角逗号 -> 半角逗号
        "。": ".",  # 全角句号 -> 半角句号
        "！": "!",  # 全角感叹号 -> 半角感叹号
        "？": "?",  # 全角问号 -> 半角问号
        "“": "\"",  # 全角左引号 -> 半角引号
        "”": "\"",  # 全角右引号 -> 半角引号
        "（": "(",  # 全角左括号 -> 半角括号
        "）": ")",  # 全角右括号 -> 半角括号
        "+": "",
    }
    for fullwidth, halfwidth in fullwidth_to_halfwidth.items():
        text = text.replace(fullwidth, halfwidth)
    return text

def clean_response(response):
    """
    清理模型返回内容，移除多余字符并替换全角符号。
    """
    response = response.strip()  # 去除首尾空白字符
    response = replace_fullwidth_symbols(response)  # 替换全角符号
    return response

def parse_dynamic_question(response):
    """
    解析语言模型生成的问题内容。
    确保返回值是合法的 JSON，并包含必要的字段。
    """
    try:
        # 直接解析 JSON
        response = clean_response(response)
        parsed_response = json.loads(response)
        
        # 验证结构完整性
        if "question" not in parsed_response or "options" not in parsed_response:
            raise ValueError("返回值缺少 'question' 或 'options' 字段。")
        
        for option in parsed_response["options"]:
            if not all(key in parsed_response["options"][option] for key in ["description", "health", "wealth", "happiness"]):
                return False
        
        return parsed_response
    except json.JSONDecodeError as e:
        return False
    
def parse_long_term_memory(response):
    """
    解析语言模型生成的长期记忆内容。
    :param response: 语言模型返回的文本内容。
    :return: Python 列表格式的长期记忆。
    """
    try:
        # 尝试直接解析为 JSON
        response = clean_response(response)
        return json.loads(response)
    except json.JSONDecodeError:
        # 如果不是 JSON 格式，提取列表部分
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end != -1:
            cleaned_response = response[start:end]
            return json.loads(cleaned_response)
        return False
    
def growing_bar(ages, total_steps=10, sleep_time=0.2):
    for step in range(total_steps + 1):
        # 成长部分
        grown = "🌱" * step  # 已经成长的部分
        # 未成长部分
        remaining = " " * (total_steps - step)  # 剩余的空白
        # 百分比
        percent = (step / total_steps) * 100
        
        # 输出进度条
        sys.stdout.write(f"\rGrowing: [{grown}{remaining}] {percent:.0f}%")
        sys.stdout.flush()
        time.sleep(sleep_time)
    print(f"\n你现在{ages}岁了! 🌳")  # 换行并结束提示

class LifeSimulator:
    def __init__(self, language_model):
        """
        初始化游戏模拟器。
        :param language_model: 语言模型接口对象，需要实现 `generate` 方法。
        """
        self.language_model = language_model
        self.state = {
            "health": random.randint(50, 100),
            "wealth": random.randint(10, 100),
            "happiness": random.randint(10, 100),
            "age": 3,
            "fixed": {
                "gender": random.choice(["男", "女"]),
                "birthplace": random.choice(["城市", "农村", "小镇"])
            },
            "dynamic": {
                "long_term_memory": [],
                "short_term_memory": "",
                "memory_capacity": 1  # 初始记忆容量
            }
        }

    def update_memory_capacity(self):
        """
        使用语言模型动态调整记忆力容量。
        """
        prompt = f"""
        玩家当前状态：
        年龄：{self.state['age']}，健康：{self.state['health']}，财富：{self.state['wealth']}，幸福：{self.state['happiness']}
        当前长期记忆：{self.state['dynamic']['long_term_memory']}
        
        请根据玩家当前状态，决定记忆力的容量（0到10之间的整数）。考虑健康、幸福感、财富以及可能的压力给出新的记忆力容量。
        输出一个整数，例如：2
        确保输出不包含多余字符。
        """
        response = self.language_model.generate(prompt)
        content = response.split("\n")[0]
        new_capacity = int(content.replace("新的记忆力容量：", "").strip())
        self.state["dynamic"]["memory_capacity"] = max(0, min(new_capacity, 10))

    def update_long_term_memory(self, new_event):
        """
        使用语言模型更新长期记忆。
        """
        # 将 JSON 数据转为字符串，避免格式冲突
        long_term_memory_str = json.dumps(self.state['dynamic']['long_term_memory'], ensure_ascii=False)
        new_event_str = json.dumps(new_event, ensure_ascii=False)
        memory_num = json.dumps(self.state['dynamic']['memory_capacity'], ensure_ascii=False)
        prompt = f"""
        玩家当前状态：
        年龄：{self.state['age']}，健康：{self.state['health']}，财富：{self.state['wealth']}，幸福：{self.state['happiness']}
        当前长期记忆：{long_term_memory_str}
        新事件：{new_event_str}

        根据玩家状态和事件的重要性，选择保留哪些长期记忆（最多 {memory_num} 条），如果没有变化，按照指定格式返回原记忆即可。
        返回格式必须是一个合法的 JSON 列表，例如：
        {{
            {{"age": 18, "description": "参加大学，学习计算机科学。"}},
            {{"age": 25, "description": "结婚，进入人生新阶段。"}}
        }}
        注意：
        - 每个元素必须包含 "age" 和 "description" 字段。
        - "age" 是整数，表示年龄。
        - "description" 是字符串，简要描述事件。
        - 确保输出不包含多余字符，仅保留合法的 JSON 格式内容。
        """
        Genrate = True
        try_time = 5
        while Genrate and try_time > 0:
            response = self.language_model.generate(prompt)
            chosen_memories = parse_long_term_memory(response)
            if chosen_memories:
                Genrate = False
            try_time -= 1
        self.state["dynamic"]["long_term_memory"] = chosen_memories

    def update_short_term_memory(self, new_event_summary):
        """
        使用语言模型更新短期记忆。
        """
        prompt = f"""
        玩家当前状态：
        年龄：{self.state['age']}，健康：{self.state['health']}，财富：{self.state['wealth']}，幸福：{self.state['happiness']}
        最近的短期记忆：{self.state['dynamic']['short_term_memory']}
        新事件总结：{new_event_summary}

        请生成一段短期记忆总结，突出情感倾向。
        """
        response = self.language_model.generate(prompt)
        self.state["dynamic"]["short_term_memory"] = response.strip()

    def generate_dynamic_question(self):
        """
        使用语言模型生成动态问题。
        """
        prompt = f"""
        玩家当前状态：
        年龄：{self.state['age']}，健康：{self.state['health']}，财富：{self.state['wealth']}，幸福：{self.state['happiness']}（数值范围是0~100）
        玩家固定属性：性别：{self.state['fixed']['gender']}，出生地：{self.state['fixed']['birthplace']}
        玩家的年龄数值决定了玩家的年龄大小，例如：1表示1岁；玩家的健康数值决定了玩家的身体状况，数值越大，健康情况越好，反之健康情况较差；玩家财富情况决定了玩家的总体资产，数值越大，玩家越有钱；玩家的信服情况决定了玩家对当前生活的满意程度，数值越高，玩家越满意当前生活。
        请基于当前玩家状态生成一个选择型问题，这个问题应该符合玩家年龄、健康、财富、幸福状态部分问题和选择应该与家人、朋友、同事、爱人等相关，并提供两个选项（A 和 B），每个选项描述该选择对玩家状态的影响。注意：问题和选项要保证合理性，确定是在玩家年龄会遇到的问题和该年龄会做的思考，在12岁之前应该有父母的介入。
        返回值必须是严格的 JSON 格式，例如：
        {{
            "question": "终于大学毕业了，你是否愿意冒险创业？",
            "options": {{
                "A": {{"description": "创业成功，获得丰厚财富。", "health": -1, "wealth": +5, "happiness": +2}},
                "B": {{"description": "保持现状，选择稳定工作。", "health": 0, "wealth": +1, "happiness": 0}}
            }}
        }}
        确保输出不包含多余字符。
        """
        Generate = True
        try_time = 5
        while Generate and try_time > 0:
            response = self.language_model.generate(prompt)
            response = parse_dynamic_question(response)
            if response:
                Generate = False
            try_time -= 1
        return response  # 假设返回合法的 Python 字典格式

    def generate_final_summary(self):
        """
        使用语言模型生成最终人生总结。
        """
        prompt = f"""
        现在正在进行一个人生模拟的游戏，
        玩家当前状态：
        年龄：{self.state['age']}，健康：{self.state['health']}，财富：{self.state['wealth']}，幸福：{self.state['happiness']}
        长期记忆：{self.state['dynamic']['long_term_memory']}
        短期记忆总结：{self.state['dynamic']['short_term_memory']}
        记忆力容量变化：玩家的记忆力容量从游戏开始时的 1，逐步调整到当前的 {self.state['dynamic']['memory_capacity']}。

        请基于上述信息，生成一段人生总结，并加入对玩家的一段人生反思或祝福。
        """
        return self.language_model.generate(prompt).strip()

    def handle_event(self, choice_effect):
        """
        处理玩家选择的事件，更新状态和记忆。
        """
        # 新事件
        new_event = {
            "description": choice_effect["description"],
            "age": self.state["age"],
            "is_major": random.random() < 0.2
        }

        # 更新长期记忆
        self.update_long_term_memory(new_event)

        # 更新短期记忆
        new_event_summary = f"你在{self.state['age']}岁时选择了: {choice_effect['description']}。"
        self.update_short_term_memory(new_event_summary)

        self.state["health"] += choice_effect["health"]
        self.state["wealth"] += choice_effect["wealth"]
        self.state["happiness"] += choice_effect["happiness"]
        self.state["age"] += random.randint(1, 10)

        # 更新记忆力
        self.update_memory_capacity()

    def play(self):
        """
        游戏主循环。
        """
        print("\n" + "这是一个人生模拟器，在这里你会随机出生在不同的地区，有着不同的健康、财富和幸福状态，你会面对着各种选择，它们会影响你在游戏中的人生，所有的问题都是双选题，用A或者B来开始人生模拟吧~")
        print(f"现在的你{self.state['age']}岁")
        while self.state["health"] > 0 and self.state["age"] < 80:
            question_data = self.generate_dynamic_question()
            print(question_data["question"])
            for option, effect in question_data["options"].items():
                print(f"{option}: {effect['description']}")

            choice = input("请选择 A 或 B: ").strip().upper()
            while choice not in question_data["options"]:
                choice = input("无效选择，请重新选择 A 或 B: ").strip().upper()
            self.handle_event(question_data["options"][choice])
            print(self.state['dynamic']['short_term_memory']+"\n")
            if self.state['dynamic']['long_term_memory']:
                for i in range(len(self.state['dynamic']['long_term_memory'])):
                    print("回忆过去，你对这些记忆印象深刻：")
                    print(f"在{self.state['dynamic']['long_term_memory'][i]['age']}岁的时候，{self.state['dynamic']['long_term_memory'][i]['description']}")
                print("\n")
            growing_bar(self.state['age'])
        print("\n游戏结束，正在生成总结...\n")
        print(self.generate_final_summary())

if __name__ == "__main__":
    model_choice = input("请选择语言模型（1: OpenAI, 2: Qwen-2.5）：").strip()
    # while model_choice not in [1, 2]:
    #     model_choice = input("无效选择，请重新选择 1 或 2: ").strip().upper()
    if model_choice == "1":
        print("您选择了 OpenAI 模型，请输入您的 OpenAI API 密钥：")
        api_key = input("请输入 API key: ").strip()
        model = OpenAILanguageModel(api_key=api_key)
    elif model_choice == "2":
        print("您选择了 Qwen 模型")
        model = QwenLanguageModel(model_name="/path/to/yourmodel")
    game = LifeSimulator(language_model=model)
    game.play()