import whisper
import os
import time

def transcribe_audio(audio_path, model_size="base"):
    """使用Whisper识别音频文件"""
    print(f"加载 {model_size} 模型...")
    model = whisper.load_model(model_size)
    
    print("开始识别...")
    start_time = time.time()
    result = model.transcribe(audio_path)
    end_time = time.time()
    
    print(f"识别完成！耗时: {end_time - start_time:.2f}秒")
    return result["text"]

# 主程序
print("=" * 50)
print("语音识别测试 - OpenAI Whisper")
print("=" * 50)

# 获取当前目录下的voice.mp4
current_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(current_dir, "voice.mp4")

print(f"音频文件路径: {audio_path}")

# 检查文件是否存在
if not os.path.exists(audio_path):
    print(f"错误：找不到音频文件 {audio_path}")
    print("请确保 voice.mp4 文件在 hw04 目录下")
else:
    print(f"音频文件大小: {os.path.getsize(audio_path)} 字节")
    print("-" * 50)
    
    # 测试 tiny 模型
    print("\n使用模型: tiny")
    try:
        text = transcribe_audio(audio_path, "tiny")
        print(f"识别结果: {text}")
        print("-" * 50)
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试 base 模型
    print("\n使用模型: base")
    try:
        text = transcribe_audio(audio_path, "base")
        print(f"识别结果: {text}")
        print("-" * 50)
    except Exception as e:
        print(f"错误: {e}")

print("\n程序结束")