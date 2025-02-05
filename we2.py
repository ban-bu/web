import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image as PILImage

# 新增的依赖库
from pyngrok import ngrok
import qrcode
from io import BytesIO
import socket
import os

def get_local_ip():
    """获取本机局域网IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def generate_qr(url):
    """生成二维码图片"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

# 初始化模型
model = load_model()

# 侧边栏 - 网络访问信息
with st.sidebar:
    st.header("🌐 访问方式")
    
    # 获取网络信息
    local_ip = get_local_ip()
    port = 8501  # Streamlit默认端口
    
    # 显示局域网访问信息
    st.subheader("局域网访问")
    st.markdown(f"`http://{local_ip}:{port}`")
    
    # 显示公网访问信息
    st.subheader("公网访问")
    
    try:
        # 检查ngrok配置
        if "NGROK_AUTHTOKEN" not in os.environ:
            st.warning("请先设置NGROK_AUTHTOKEN环境变量")
        else:
            ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
            public_url = ngrok.connect(port, proto="http").public_url
            
            # 生成并显示二维码
            qr_img = generate_qr(public_url)
            st.image(qr_img, caption="扫描二维码访问", use_column_width=True)
            st.markdown(f"[{public_url}]({public_url})")
            
    except Exception as e:
        st.error(f"公网访问初始化失败: {str(e)}")

# 主界面
st.title("📷 图片上传与分析")
st.markdown("支持格式：JPEG/PNG，图片将被自动调整为224x224像素")

# 文件上传组件
uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # 处理图片
        img = PILImage.open(uploaded_file).convert('RGB')
        st.image(img, caption="上传的图片", use_column_width=True)
        
        # 预处理
        img = img.resize((224, 224))
        img_array = np.array(img)
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # 预测
        with st.spinner('正在分析图片...'):
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=3)[0]

        # 显示结果
        st.success("分析完成！")
        st.subheader("识别结果（置信度）:")
        
        cols = st.columns(3)
        for idx, (_, label, score) in enumerate(decoded_predictions):
            with cols[idx]:
                st.metric(label=f"Top {idx+1}", 
                          value=f"{label.capitalize()} ({score:.1%})")

    except Exception as e:
        st.error(f"处理出错: {str(e)}")

# 运行提示
st.markdown("""
---
**运行提示**  
请使用以下命令启动服务：  
`streamlit run app.py --server.address 0.0.0.0 --server.port 8501`
""")