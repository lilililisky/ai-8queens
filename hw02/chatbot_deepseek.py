"""
DeepSeek Chatbot 示例 - 官方API版
已充值10元，可以正常调用
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class DeepSeekChatbot:
    def __init__(self, api_key=None):
        """初始化DeepSeek聊天机器人"""
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请在.env文件中设置DEEPSEEK_API_KEY")
        
        # 初始化OpenAI客户端（DeepSeek兼容OpenAI格式）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
    
    def chat(self, user_message, system_prompt="你是一个有帮助的助手"):
        """发送消息给DeepSeek并获取回复"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 使用deepseek-chat模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"调用失败: {str(e)}"
    
    def check_balance(self):
        """查询账户余额"""
        try:
            response = self.client.get(f"{self.client.base_url}/user/balance")
            return response.json()
        except Exception as e:
            return f"余额查询失败: {str(e)}"


def main():
    print("=" * 60)
    print("          DeepSeek Chatbot 官方API版")
    print("=" * 60)
    print("💰 已充值10元，可正常调用")
    print("-" * 60)
    
    try:
        # 创建机器人实例
        bot = DeepSeekChatbot()
        print("✅ 初始化成功")
        print("-" * 60)
        
        # 测试对话列表
        test_questions = [
            "你好，请用一句话介绍自己",
            "什么是Transformer模型？简单说明",
            "深度学习和机器学习有什么区别？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 测试 {i}/3")
            print(f"👤 问: {question}")
            print("🤖 AI思考中...")
            
            answer = bot.chat(question)
            print(f"🤖 答: {answer}")
            print("-" * 60)
        
        # 交互模式
        print("\n💬 进入交互模式（输入'退出'结束）")
        while True:
            user_input = input("\n👤 请输入问题: ")
            if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                print("👋 再见！")
                break
            
            if user_input.strip() == "":
                print("请输入有效问题")
                continue
            
            print("🤖 AI思考中...")
            response = bot.chat(user_input)
            print(f"🤖 AI: {response}")
            
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        print("请检查.env文件中的API Key是否正确")
    except Exception as e:
        print(f"❌ 未知错误: {e}")


if __name__ == "__main__":
    main()
    