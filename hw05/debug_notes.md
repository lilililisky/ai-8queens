# 调试记录

## 实验环境
- OS: Windows 10
- Python: 3.9.7
- PyTorch: 2.0+
- 硬件: CPU (无 GPU)

## 遇到的问题及解决

### 问题1：安装 PyTorch 速度慢
- **现象**：pip install 下载很慢
- **解决**：使用清华镜像源
  ```bash
  pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple