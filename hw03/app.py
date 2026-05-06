"""
人脸检测Web应用 - 使用OpenCV Haar Cascade
功能：上传图片，检测人脸位置，标注人脸框
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

# 设置页面
st.set_page_config(
    page_title="人脸检测系统",
    page_icon="👤",
    layout="wide"
)

st.title("👤 人脸检测Web应用")
st.markdown("基于 OpenCV Haar Cascade 的人脸检测系统")


def load_face_cascade():
    """加载OpenCV人脸检测器"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


def detect_faces_opencv(image, face_cascade):
    """使用OpenCV检测人脸"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def draw_face_boxes(image, faces):
    """在图片上绘制人脸框"""
    img_copy = image.copy()
    
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Face {i+1}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_copy, (x, y - label_size[1] - 5), 
                      (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_copy


# 侧边栏
with st.sidebar:
    st.header("📁 功能设置")
    
    uploaded_file = st.file_uploader(
        "上传图片",
        type=["jpg", "jpeg", "png"],
        help="支持jpg、jpeg、png格式"
    )
    
    st.markdown("---")
    st.markdown("### 参数调整")
    
    scale_factor = st.slider(
        "scaleFactor (缩放比例)",
        min_value=1.01,
        max_value=1.5,
        value=1.1,
        step=0.01
    )
    
    min_neighbors = st.slider(
        "minNeighbors (最小邻居数)",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    st.markdown("---")
    st.markdown("### 功能说明")
    st.markdown("- 人脸检测：框出图片中所有人脸")
    st.markdown("- 支持多张人脸同时检测")


def main():
    """主函数"""
    face_cascade = load_face_cascade()
    image = None
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.success("✅ 图片上传成功！")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("图片宽度", image.shape[1])
        with col2:
            st.metric("图片高度", image.shape[0])
        with col3:
            st.metric("颜色通道", image.shape[2] if len(image.shape) > 2 else 1)
    
    if image is not None:
        st.subheader("📷 原始图片")
        # 修复：use_container_width 改为 use_column_width
        st.image(image, use_column_width=True)
        
        # 转换为BGR（OpenCV格式）
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        with st.spinner("🔍 正在检测人脸..."):
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=int(min_neighbors),
                    minSize=(30, 30)
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("👤 检测到的人脸", len(faces))
                with col2:
                    if len(faces) > 0:
                        st.success(f"✅ 成功检测到 {len(faces)} 张人脸")
                    else:
                        st.warning("⚠️ 未检测到人脸")
                
                if len(faces) > 0:
                    result_image = draw_face_boxes(image_bgr, faces)
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    st.subheader("🎯 检测结果")
                    # 修复：use_container_width 改为 use_column_width
                    st.image(result_image_rgb, use_column_width=True)
                    
                    with st.expander("📊 人脸详细信息"):
                        for i, (x, y, w, h) in enumerate(faces):
                            st.write(f"**人脸 {i+1}**")
                            st.write(f"- 位置：x={x}, y={y}, 宽度={w}, 高度={h}")
                            
            except Exception as e:
                st.error(f"处理图片时出错：{str(e)}")
    
    else:
        st.info("👈 请在左侧上传图片开始检测")
        
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 使用步骤
            1. 在左侧边栏点击"上传图片"
            2. 选择包含人脸的图片
            3. 系统会自动检测并标记人脸位置
            
            ### 功能特点
            - 自动检测图片中所有人脸
            - 用绿色框标记人脸位置
            - 可调整检测参数
            """)
    
    st.markdown("---")
    st.markdown("💡 **技术栈**：Streamlit + OpenCV (Haar Cascade)")


if __name__ == "__main__":
    main()