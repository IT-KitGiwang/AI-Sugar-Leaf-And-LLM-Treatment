import streamlit as st
import os
import shutil
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from google import genai
from google.genai import types
import torch.nn.functional as F

# ==================== CẤU HÌNH API & BIẾN TOÀN CỤC ====================
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
API_KEY = os.getenv("GOOGLE_API_KEY")

# Khởi tạo Session State chỉ cho các biến cần thiết
if 'feedback_sent' not in st.session_state:
    st.session_state.feedback_sent = False
if 'current_image_hash' not in st.session_state:
    st.session_state.current_image_hash = None

# Khởi tạo Gemini Client
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo Gemini Client: {e}. Vui lòng kiểm tra GOOGLE_API_KEY.")
else:
    st.error("❌ Lỗi API: Không tìm thấy GOOGLE_API_KEY trong biến môi trường.")

# Danh sách các loại bệnh
CLASSES = {
    'Healthy': 'Khỏe mạnh (Healthy)',
    'Mosaic': 'Bệnh khảm lá (Mosaic Virus)',
    'RedRot': 'Bệnh thối đỏ (Red Rot)',
    'Rust': 'Bệnh gỉ sắt (Rust)',
    'Yellow': 'Vàng lá - Thiếu dinh dưỡng (Yellow Leaf)'
}

CONFIDENCE_THRESHOLD = 0.85

# ==================== HÀM HỖ TRỢ CƠ BẢN ====================
def set_seed(seed=42):
    """Đặt seed cho tính tái lập kết quả"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms():
    """Trả về transform để xử lý ảnh đầu vào"""
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@st.cache_resource
def load_model():
    """Tải mô hình AI từ file hoặc tạo mô hình mặc định"""
    model_path = 'models/sugarcane_disease_model.pth'
    
    if os.path.exists(model_path):
        try:
            model = torch.jit.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            return model
        except Exception as e:
            st.error(f"❌ Lỗi tải mô hình: {e}")

    st.warning("⚠️ Không tìm thấy mô hình. Đang tạo mô hình ResNet18 mặc định...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model.eval()
    return model

def predict_disease(image, model):
    transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device("cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        pred_idx = preds.item()
        confidence_value = confidence.item()
    class_keys = list(CLASSES.keys())
    predicted_class_key = class_keys[pred_idx]
    predicted_class_name = CLASSES[predicted_class_key]
    is_confident = confidence_value >= CONFIDENCE_THRESHOLD
    return predicted_class_name, confidence_value, is_confident

def save_feedback(image_path, predicted_class, is_correct):
    base_dir = 'feedback'
    split = 'True' if is_correct else 'False'
    target_dir = os.path.join(base_dir, split, predicted_class)
    os.makedirs(target_dir, exist_ok=True)
    new_filename = f"{int(time.time())}_{os.path.basename(image_path)}"
    shutil.copy(image_path, os.path.join(target_dir, new_filename))
    return target_dir

def get_image_hash(image):
    import hashlib
    return hashlib.md5(image.tobytes()).hexdigest()

# ==================== LỚP HỖ TRỢ GEMINI ====================
class GeminiHelper:
    def __init__(self, client):
        self.client = client
        # Khởi tạo phiên chat mới mỗi lần reload
        self.chat_session = self.client.chats.create(model="gemini-2.0-flash-exp")

    def consult_treatment(self, query):
        system_instruction = """
        Bạn là chuyên gia nông nghiệp cây mía Việt Nam. Trả lời NGẮN GỌN, RÕ RÀNG, CHUẨN CHUYÊN MÔN.
        🎯 NGUYÊN TẮC:
        1️⃣ Sử dụng kiến thức chuyên ngành cập nhật.
        2️⃣ Có thể dùng Google Search để kiểm tra thông tin mới nhất, và TRÍCH DẪN nguồn tin cậy.
        🧾 ĐỊNH DẠNG TRẢ LỜI:
        • **Triệu chứng:**
        • **Nguyên nhân:**
        • **Cách chữa:**
        • **Lưu ý:**
        ⚠️ Cuối cùng: thêm phần `📚 Nguồn:` ghi rõ nếu từ web.
        """
        try:
            response = self.chat_session.send_message(
                [types.Part(text=query)],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[{"google_search": {}}],
                ),
            )
            text = response.text or "Không có phản hồi rõ ràng."
            citations = ""
            gm = getattr(response.candidates[0], "grounding_metadata", None)
            if gm and getattr(gm, "web_search_queries", None):
                citations = "🔎 **Nguồn web:** " + ", ".join(gm.web_search_queries)
            return text, citations
        except Exception as e:
            return f"⚠️ Lỗi tư vấn (Gemini): {e}", ""

# ==================== KẾ HOẠCH ĐIỀU TRỊ CƠ BẢN ====================
def get_treatment_plan(disease_name):
    """Trả về kế hoạch điều trị cơ bản cho từng loại bệnh"""
    plans = {
        'Khỏe mạnh (Healthy)': """✅ CÂY MÍA KHỎE MẠNH

🧪 **Đặc điểm chung:**
Cây mía khỏe mạnh có lá xanh bóng, thân đứng vững, rễ phát triển mạnh, không có dấu hiệu héo, thối hay biến dạng. Đây là trạng thái lý tưởng giúp cây quang hợp tối đa và cho năng suất cao.

🧴 **Hướng dẫn chăm sóc định kỳ:**
- Theo dõi sinh trưởng 7–10 ngày/lần, chú ý sâu bệnh, độ ẩm và độ pH đất.
- Bón phân **NPK 16-16-8** với liều lượng **500 kg/ha/vụ**, chia 2–3 lần bón trong mùa sinh trưởng.
- Tưới nước đều đặn, duy trì khoảng **20 mm/tuần**; tránh úng ngập kéo dài.
- Giữ luống thoáng, làm cỏ và xới xáo nhẹ quanh gốc để đất thông thoáng.

🛡 **Biện pháp phòng ngừa:**
- Sử dụng giống mía **kháng bệnh và năng suất cao**, có nguồn gốc rõ ràng.
- Cân đối phân bón: tránh bón thừa đạm, tăng hữu cơ và kali.
- Quản lý nước hợp lý, không để ruộng quá khô hoặc úng.

🌾 **Mẹo dành cho nhà nông:**
- Bổ sung **chế phẩm vi sinh Trichoderma** để tăng sức đề kháng rễ.
- Ghi nhật ký chăm sóc (phân, nước, thời tiết) giúp đánh giá và cải tiến vụ sau.""",

        'Bệnh khảm lá (Mosaic Virus)': """🦠 BỆNH KHẢM LÁ (MOSAIC VIRUS)

🧪 **Nguyên nhân thường gặp:**
- Do **virus Sugarcane mosaic virus (SCMV)** gây ra.
- Virus lây lan chủ yếu qua **rệp muội (Aphis spp.)** chích hút, hoặc từ **cây giống bị nhiễm bệnh**.
- Thời tiết ẩm nóng, trồng dày, chăm sóc kém khiến bệnh phát triển mạnh.

🧴 **Hướng dẫn điều trị tại nhà (tham khảo):**
- Phun **Imidacloprid 0,5 ml/lít nước**, phun 2 lần cách nhau 7 ngày để trừ rệp.
- **Nhổ bỏ và tiêu hủy** cây bị bệnh nặng, tránh lây lan sang khu vực khác.
- Sau khi xử lý, rửa sạch dụng cụ cắt tỉa bằng **cồn 70° hoặc dung dịch Cloramin B** để diệt virus bám dính.

🛡 **Biện pháp phòng ngừa:**
- Trồng giống **mía kháng Mosaic**, lấy giống từ ruộng sạch bệnh.
- Không trồng liên tục nhiều vụ cùng giống ở cùng một khu vực.
- Vệ sinh ruộng thường xuyên, kiểm tra rệp định kỳ.

🌾 **Mẹo dành cho nhà nông:**
- Dùng **bẫy dính màu vàng** để giám sát mật độ rệp.
- Có thể **trồng xen cúc vạn thọ** hoặc **húng quế** để xua rệp tự nhiên.""",

        'Bệnh thối đỏ (Red Rot)': """🍄 BỆNH THỐI ĐỎ (RED ROT)

🧪 **Nguyên nhân thường gặp:**
- Gây ra bởi nấm **Colletotrichum falcatum Went**.
- Nấm phát triển mạnh trong điều kiện **ẩm độ cao**, thoát nước kém và khi **giống mía yếu hoặc trồng dày**.
- Cây bị tổn thương do côn trùng, dao cắt hoặc sau mưa dài ngày dễ nhiễm bệnh.

🧴 **Hướng dẫn điều trị tại nhà (tham khảo):**
- Dùng thuốc **Carbendazim 50%** với liều **500 g/ha**, pha loãng tưới đều quanh gốc 1–2 lần.
- Cắt bỏ thân bệnh, đốt tiêu hủy toàn bộ phần bị thối.
- Nếu vùng bệnh lan rộng, tiến hành **luân canh với cây họ đậu 1 vụ** để cắt mầm nấm.

🛡 **Biện pháp phòng ngừa:**
- Chọn **giống kháng nấm**, không sử dụng hom từ ruộng có tiền sử bệnh.
- Trồng trên đất cao, **cải tạo rãnh thoát nước** tốt.
- Không trồng mía liên tục trên cùng ruộng quá 3 vụ liên tiếp.

🌾 **Mẹo dành cho nhà nông:**
- Sau mỗi vụ, **cày phơi ải đất ít nhất 3 tuần** để nắng diệt bào tử nấm.
- Bón **vôi bột 300 kg/ha** sau thu hoạch để trung hòa pH và diệt khuẩn.""",

        'Bệnh gỉ sắt (Rust)': """🍂 BỆNH GỈ SẮT (SUGARCANE RUST)

🧪 **Nguyên nhân thường gặp:**
- Tác nhân gây bệnh là nấm **Uromyces scitamineus**.
- Phát triển mạnh khi nhiệt độ từ **25–30°C** và độ ẩm không khí cao.
- Gió, mưa và dụng cụ nông nghiệp là nguồn lây lan chính.

🧴 **Hướng dẫn điều trị tại nhà (tham khảo):**
- Phun **Mancozeb 80WP 2 kg/ha**, pha đúng liều khuyến cáo, phun 3 lần cách nhau 7–10 ngày.
- Cắt bỏ toàn bộ lá bệnh nặng và **đốt tiêu hủy**.
- Kết hợp **bổ sung phân Kali** để tăng sức đề kháng cho cây.

🛡 **Biện pháp phòng ngừa:**
- Trồng **giống kháng gỉ sắt** đã được Viện Mía Đường khuyến nghị.
- Giữ ruộng thoáng, tránh trồng quá dày.
- Không bón thừa đạm – dễ làm lá non yếu và dễ nhiễm nấm.

🌾 **Mẹo dành cho nhà nông:**
- Phun thuốc vào **buổi sáng sớm hoặc chiều mát**, khi không có gió để đạt hiệu quả cao.
- Có thể **luân phiên thuốc gốc đồng và Mancozeb** để tránh kháng thuốc.""",

        'Vàng lá - Thiếu dinh dưỡng (Yellow Leaf)': """🌱 HIỆN TƯỢNG VÀNG LÁ (THIẾU DINH DƯỠNG)

🧪 **Nguyên nhân thường gặp:**
- Thiếu **đạm (N)** là nguyên nhân phổ biến nhất, ngoài ra còn do thiếu **lưu huỳnh (S)** hoặc **sắt (Fe)**.
- Đất chua (pH < 5,5) làm giảm khả năng hấp thu dinh dưỡng.
- Rửa trôi phân bón do mưa nhiều hoặc tưới quá mức.

🧴 **Hướng dẫn điều trị tại nhà (tham khảo):**
- Bón **Urê 200 kg/ha**, chia 2–3 lần trong vụ, kết hợp **phân hữu cơ vi sinh** để giữ ẩm.
- Phun dung dịch **Urê 5%** hoặc **phân bón lá chứa Fe, Zn** giúp lá phục hồi nhanh.
- Kiểm tra pH đất, nếu thấp thì **bón vôi 100–200 kg/ha** để nâng pH.

🛡 **Biện pháp phòng ngừa:**
- Duy trì **pH đất 6,0–7,0** và cân đối phân NPK hợp lý.
- Sử dụng **phân chuồng hoai, phân xanh**, tăng mùn và vi sinh vật có lợi.
- Theo dõi lá thường xuyên để phát hiện sớm tình trạng thiếu dinh dưỡng.

🌾 **Mẹo dành cho nhà nông:**
- Sau mưa lớn, nên **bổ sung phân bón lá nhẹ** để tránh rửa trôi.
- Dùng **than sinh học (biochar)** trộn đất để giữ ẩm và dinh dưỡng lâu dài."""
    }
    return plans.get(disease_name, "❓ LIÊN HỆ CHUYÊN GIA!")

# ==================== GIAO DIỆN CHÍNH ====================
def main():
    st.set_page_config(layout="wide", page_title="🌾 AI Cây Mía Nâng Cao")

    # CSS tùy chỉnh
    st.markdown("""
    <style>
    .chat-message {
        margin: 10px 0;
        margin-bottom: 0.5cm;
        padding: 10px;
        border-radius: 10px;
        max-width: 85%;
        word-wrap: break-word;
    }
    .chat-user {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        margin-left: auto;
        text-align: right;
        padding: 8px 12px;
        border-radius: 12px;
        max-width: 85%;
        width: fit-content;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }
    .st-emotion-cache-12j140x p, .st-emotion-cache-12j140x ol, .st-emotion-cache-12j140x ul, .st-emotion-cache-12j140x dl, .st-emotion-cache-12j140x li {
    font-size: 18px;
    line-height: 1.6;
    align-items: justify;
    }
    .chat-assistant {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        margin-right: auto;
        text-align: left;
        padding: 8px 12px;
        border-radius: 12px;
        max-width: 85%;
        width: fit-content;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }
    .treatment-plan {
    font-size: 20px;
    line-height: 1.5;
    text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tải mô hình AI
    model = load_model()

    # Khởi tạo Gemini Helper
    gemini = None
    if client:
        gemini = GeminiHelper(client)

    # Tiêu đề ứng dụng
    st.markdown('<h1 style="text-align: center;">🌾 AI NHẬN DIỆN & TƯ VẤN BỆNH CÂY MÍA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Tích hợp GenAI, Tìm kiếm Web và Độ Tin Cậy</p>', unsafe_allow_html=True)

    # Thanh bên (Sidebar)
    with st.sidebar:
        tab1, tab2 = st.tabs(["📖 Hướng dẫn", "ℹ️ Thông tin"])
        with tab1:
            st.markdown("### 📖 Hướng dẫn sử dụng")
            st.markdown("""
            1. **Nhận diện bệnh:** Tải ảnh hoặc chụp từ webcam
            2. **Xem kết quả:** Chỉ hiển thị nếu độ tin cậy ≥ 85%
            3. **Tư vấn chuyên sâu:** Hỏi chatbot về bệnh cây mía
            4. **Phản hồi:** Gửi feedback 1 lần/ảnh để cải thiện mô hình
            """)
        with tab2:
            st.markdown("### ℹ️ Thông tin đề tài")
            st.markdown("""
            **Đề tài:** AI Nhận Diện & Tư Vấn Bệnh Cây Mía
            **Mô tả:** Ứng dụng AI nhận diện bệnh trên lá cây mía với độ tin cậy cao (≥85%) và tư vấn điều trị chuyên sâu.
            **Công nghệ:** Streamlit, PyTorch, Google GenAI
            **Năm:** 2025
            """)

    # Chia layout thành 2 cột
    col1, col2 = st.columns(2)

    # ========== CỘT 1: NHẬN DIỆN BỆNH ==========
    with col1:
        st.markdown('<h3 style="text-align: center;color:white; background-color: #7f69f4; padding: 10px; border-radius: 5px; margin-bottom:1cm;">🔍 NHẬN DIỆN VÀ ĐIỀU TRỊ</h3>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 23px; font-weight: bold; margin:0 0 0.5cm 0;">Chọn phương thức nhập ảnh:</div>', unsafe_allow_html=True)
        input_method = st.radio("Chọn phương thức nhập ảnh:", ["Tải ảnh", "Chụp từ webcam"], key="input_method", label_visibility="collapsed")
        image = None
        if input_method == "Tải ảnh":
            st.markdown('<div style="font-size: 23px; color: red; font-weight:bold;margin-bottom:0.5cm;">📸 Tải ảnh lá cây mía</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("📸 Tải ảnh lá cây mía", type=['png', 'jpg', 'jpeg'], key="image_uploader", label_visibility="collapsed")
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
        else:
            st.markdown('<div style="font-size: 18px;">📷 Chụp ảnh từ webcam</div>', unsafe_allow_html=True)
            camera_input = st.camera_input("📷 Chụp ảnh từ webcam", key="camera_input", label_visibility="collapsed")
            if camera_input:
                image = Image.open(camera_input).convert("RGB")

        if image:
            st.image(image, width=200, caption="Ảnh được nhập")
            current_hash = get_image_hash(image)
            if st.session_state.current_image_hash != current_hash:
                st.session_state.feedback_sent = False
                st.session_state.current_image_hash = current_hash
            with st.spinner("🔬 AI đang phân tích..."):
                disease_name, confidence, is_confident = predict_disease(image, model)
            if is_confident:
                st.markdown(f"""
                <div style="background-color: #dc3545; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <p style="margin: 0; font-size:25px; font-weight: bold;"> 🎯 Cảnh báo: {disease_name}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color: blue; color: white; padding: 5px; border-radius: 10px; text-align: center; margin: 10px 0;">
                    <p style="margin: 0; font-size:20px; font-weight: bold;">📊 Độ chính xác dự đoán: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                st.subheader("💡 **KẾ HOẠCH ĐIỀU TRỊ CƠ BẢN**")
                with st.container():
                    st.markdown(f'<div class="treatment-plan">{get_treatment_plan(disease_name)}</div>', unsafe_allow_html=True)
                st.subheader("📝 **PHẢN HỒI (FEEDBACK)**")
                if st.session_state.feedback_sent:
                    st.success("✅ Bạn đã gửi phản hồi cho ảnh này rồi!")
                else:
                    st.markdown('<div style="font-size: 20px;">Kết quả dự đoán có đúng không?</div>', unsafe_allow_html=True)
                    correct = st.radio("", ["Đúng", "Sai"], key="feedback_radio", label_visibility="collapsed")
                    col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
                    with col_center2:
                        if st.button("💾 Lưu Feedback", use_container_width=True):
                            img_path = f"temp_{int(time.time())}_feedback.jpg"
                            image.save(img_path)
                            save_feedback(img_path, disease_name, correct=="Đúng")
                            st.success("✅ Đã lưu phản hồi! Dữ liệu sẽ được dùng để cải thiện mô hình.")
                            os.remove(img_path)
                            st.session_state.feedback_sent = True
                            time.sleep(0.5)
                            st.rerun()
            else:
                st.warning("### ⚠️ KHÔNG THỂ XÁC ĐỊNH CHÍNH XÁC")
                st.info(f"📊 **Độ tin cậy:** {confidence*100:.2f}% (Cần ≥ {CONFIDENCE_THRESHOLD*100}%)")
                st.markdown("""
                **HÌNH ẢNH NÀY TÔI KHÔNG CHẮC CHẮN VÌ CÓ NHIỀU YẾU TỐ:**
                - 📸 Ảnh mờ hoặc không rõ nét
                - 💡 Ánh sáng không đủ hoặc quá sáng
                - 🍃 Chụp nhiều lá cùng
                - 🚫 Ảnh không liên quan đến lá cây mía
                - 🔄 Góc chụp không phù hợp
                **💡 ĐỀ XUẤT:**
                - Chụp lại ảnh với ánh sáng tốt hơn
                - Chụp 1 lá riêng biệt, rõ nét
                - Đảm bảo ảnh là lá cây mía thật
                """)

    # ========== CỘT 2: CHATBOT TƯ VẤN ==========
    with col2:
        st.markdown('''
        <h3 style="text-align: center; background-color: #249adc; color: white; padding: 10px; border-radius: 5px; margin-bottom: 1cm;">
            <img src="https://cdn-icons-png.flaticon.com/512/8943/8943377.png" 
                alt="Icon" 
                style="width: 30px; height: 30px; vertical-align: middle; margin-right: 10px;">
            AI TƯ VẤN ĐIỀU TRỊ THAM KHẢO
        </h3>
        ''', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message chat-assistant">Xin chào tôi là trợ lý ảo do nhóm HS ... tạo ra!</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message chat-assistant">Tôi có thể đồng hành với bạn để hướng dẫn bạn điều trị các bệnh trên cây mía.</div>', unsafe_allow_html=True)

        if gemini:
            # Tạo biến tạm để lưu lịch sử chat trong phiên hiện tại
            if 'temp_chat_history' not in st.session_state:
                st.session_state.temp_chat_history = []

            # Hiển thị lịch sử chat tạm thời
            for message in st.session_state.temp_chat_history:
                role_class = "chat-user" if message['role'] == "user" else "chat-assistant"
                st.markdown(f'<div class="chat-message {role_class}">{message["text"]}</div>', unsafe_allow_html=True)

            # Ô nhập câu hỏi
            query = st.chat_input("Hỏi chuyên gia về các loại bệnh")
            if query:
                # Thêm câu hỏi của người dùng vào lịch sử tạm
                st.session_state.temp_chat_history.append({"role": "user", "text": query})
                with st.spinner("🤖 Chuyên gia Gemini đang trả lời..."):
                    response_text, citations = gemini.consult_treatment(query)
                    st.session_state.temp_chat_history.append({"role": "assistant", "text": response_text + (f"\n{citations}" if citations else "")})
                st.rerun()
        else:
            st.warning("⚠️ Chatbot bị vô hiệu hóa do lỗi API Key. Vui lòng kiểm tra cấu hình API.")

    # Footer
    st.markdown("---")
    st.markdown('<p style="text-align: center;">🌾 AI Nông Nghiệp Việt Nam 2025 - Sử dụng Google GenAI & PyTorch</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('feedback/True', exist_ok=True)
    os.makedirs('feedback/False', exist_ok=True)
    main()
