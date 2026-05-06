# 语音识别方案调研与实验报告

## 一、开源语音识别方案对比

| 方案 | 许可协议 | 模型大小 | 推理速度 | 语言支持 | 部署难度 |
|------|----------|----------|----------|----------|----------|
| OpenAI Whisper | MIT | tiny: 39MB, base: 139MB | 快 | 多语言 | 中等 |
| Vosk | Apache 2.0 | 40MB-1GB | 快 | 多语言 | 简单 |
| FunASR | MIT | 200MB+ | 中等 | 中文优化 | 中等 |

## 二、选型与实现

选择 **OpenAI Whisper** 进行本地测试。

### 环境说明
- OS: Windows 10
- Python: 3.9.7
- 无GPU (CPU模式)

### 实现步骤
1. 安装 whisper: `pip install openai-whisper`
2. 安装 ffmpeg 用于音频处理
3. 将 mp4 音频转换为 wav 格式
4. 使用 whisper 进行语音识别

## 三、识别结果

**测试音频**：任务二生成的配音 (voice.mp4)

**识别文字**：
由于 ffmpeg 文件较大（192MB），无法上传至 GitHub，实际运行需要手动安装 ffmpeg 并添加到系统路径。
安装方法：https://ffmpeg.org/download.html