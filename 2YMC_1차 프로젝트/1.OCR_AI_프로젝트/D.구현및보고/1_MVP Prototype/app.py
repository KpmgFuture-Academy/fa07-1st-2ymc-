import os, io, json, time, re, uuid, shutil, requests, cv2
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 환경/설정
# =========================
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
DATA_ROOT       = Path(os.getenv("DATA_ROOT", "./data/users")).resolve()
OCR_ENDPOINT    = "https://api.upstage.ai/v1/document-digitization"
GPT_MODEL       = "gpt-4o-mini"   # 필요시 다른 모델로 교체

assert UPSTAGE_API_KEY, "환경변수 UPSTAGE_API_KEY가 없습니다(.env 확인)."
assert OPENAI_API_KEY,  "환경변수 OPENAI_API_KEY가 없습니다(.env 확인)."

client = OpenAI(api_key=OPENAI_API_KEY)

DOC_TYPES = [
    "보험금청구서",
    "진단서",
    "입퇴원확인서",
    "처방전",
    "사고경위서",
    "수리견적서",
    "사망진단서",
    "기타 문서"
]

# =========================
# 유틸
# =========================
def ensure_user_folder(customer_id: str) -> Path:
    """고객별/업로드회차별 폴더 생성"""
    safe_id = re.sub(r"[^0-9A-Za-z가-힣_-]", "_", customer_id.strip())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = DATA_ROOT / safe_id / ts
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def save_uploaded_file(file, dst_path: Path) -> Path:
    with open(dst_path, "wb") as f:
        f.write(file.read())
    return dst_path

def preprocess_image_to_png(src_path: Path) -> Path:
    """간단 전처리(그레이/오츠 이진화) PNG로 저장"""
    img = cv2.imdecode(
        np.fromfile(str(src_path), dtype="uint8"),
        cv2.IMREAD_COLOR
    ) if src_path.suffix.lower() not in [".png", ".jpg", ".jpeg"] else cv2.imread(str(src_path))
    if img is None:
        # PIL fallback
        pil = Image.open(src_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out = src_path.with_suffix(".proc.png")
    cv2.imwrite(str(out), thr)
    return out

def run_upstage_ocr(image_path: Path) -> dict:
    """Upstage OCR 호출 -> {text, ...}"""
    with open(image_path, "rb") as f:
        files = {"document": f}
        data = {"model": "ocr"}
        headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}
        res = requests.post(OCR_ENDPOINT, headers=headers, files=files, data=data, timeout=60)
        res.raise_for_status()
        return res.json()

def classify_with_gpt(ocr_text: str) -> dict:
    """
    GPT로 문서유형 분류 + 핵심 필드 추출.
    JSON으로 강제 반환.
    """
    system_prompt = (
        "당신은 보험 문서 분류/추출 전문가입니다. "
        "사용자가 제공한 OCR 텍스트를 바탕으로 문서 유형을 다음 중 하나로 분류하세요: "
        f"{', '.join(DOC_TYPES)}. "
        "가능하면 핵심 필드(이름, 생년월일, 날짜, 금액, 증권/계약번호, 병원/기관명, 계좌정보 등)를 추출하세요. "
        "정확도(confidence)는 0~1 사이 실수로 주세요. 반드시 JSON으로만 답변하세요."
    )
    user_prompt = f"""
[OCR_TEXT BEGIN]
{ocr_text[:20000]}
[OCR_TEXT END]

반환 형식(JSON):
{{
  "doc_type": "<위 목록 중 하나>",
  "confidence": 0.0,
  "key_fields": {{
      "name": "...",
      "dob": "...",
      "date": "...",
      "amount": "...",
      "policy_number": "...",
      "hospital": "...",
      "account": "..."
  }},
  "rationale": "간단 근거"
}}
    """.strip()

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0.1,
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_prompt}
        ]
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        # JSON 파싱 실패 시 안전 래퍼
        data = {"doc_type":"기타","confidence":0.0,"key_fields":{},"rationale":"parse_failed","raw":content}
    return data

# numpy import (지연 사용 대비)
import numpy as np

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="보험 청구 서류 업로드 · OCR · 분류", page_icon="🧾", layout="wide")

st.title("🧾 YMC 보험 청구")
st.caption("빠른 청구는 YMC 보험")

with st.form("uploader", clear_on_submit=False):
    col1, col2 = st.columns([1,2])
    with col1:
        customer_id = st.text_input("고객 ID 또는 이름", placeholder="예: 홍길동_ID")
    with col2:
        uploaded = st.file_uploader("보험 청구 서류 업로드", type=["png","jpg","jpeg","pdf","tif","tiff"])
    submitted = st.form_submit_button("업로드 및 분류 실행")

if submitted:
    # ✅ 0) 유효성 검사: 파일 없으면 즉시 종료(어떤 기록/폴더 생성도 X)
    if uploaded is None:
        st.warning("파일을 업로드해 주세요.")
        st.stop()  # ← 아래 코드 전부 실행되지 않음 (로그/폴더 생성 방지)

    if not customer_id.strip():
        st.warning("고객 ID 또는 이름을 입력해 주세요.")
        st.stop()


    # 1) 고객 폴더 생성
    user_folder = ensure_user_folder(customer_id)
    st.info(f"📂 고객 폴더 생성: {user_folder}")

    # 2) 원본 저장
    original_path = user_folder / uploaded.name
    save_uploaded_file(uploaded, original_path)
    st.success(f"원본 저장 완료: {original_path.name}")

    # 3) 이미지/페이지 전처리 (단순화: 이미지형만 처리, PDF는 그대로 OCR 시도)
    targets = []
    if original_path.suffix.lower() in [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]:
        proc = preprocess_image_to_png(original_path)
        targets = [proc]
    else:
        # PDF/기타는 전처리 생략하고 그대로 보냄 (Upstage가 내부 처리)
        targets = [original_path]

    # 4) Upstage OCR 호출(복수 page/파일이면 합치기)
    ocr_texts = []
    for t in targets:
        try:
            res = run_upstage_ocr(t)
            ocr_texts.append(res.get("text",""))
        except Exception as e:
            st.error(f"OCR 실패: {e}")
            st.stop()

    merged_text = "\n\n".join(ocr_texts).strip()

    # 5) GPT 분류
    with st.spinner("GPT로 문서 유형 분류 중..."):
        classify = classify_with_gpt(merged_text)

    # 6) 결과 저장
    text_path = user_folder / "ocr_text.txt"
    json_path = user_folder / "classification.json"
    text_path.write_text(merged_text, encoding="utf-8")
    json_path.write_text(json.dumps(classify, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7) 결과 표시
    st.subheader("분류 결과")
    st.json(classify)
    st.download_button("OCR 텍스트 다운로드", data=merged_text, file_name="ocr_text.txt")
    st.download_button("분류 JSON 다운로드", data=json.dumps(classify, ensure_ascii=False, indent=2), file_name="classification.json")

    # 간단 미리보기
    if original_path.suffix.lower() in [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]:
        st.image(str(original_path), caption="원본 미리보기", use_column_width=True)