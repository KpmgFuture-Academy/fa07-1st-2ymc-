import os, io, json, time, re, uuid, shutil, requests, cv2
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

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
    """
    간단 전처리(그레이/오츠 이진화) 후 PNG로 저장 (유니코드 경로 안전)
    - 모든 포맷을 np.fromfile + cv2.imdecode로 읽음
    - 저장은 cv2.imencode(...).tofile(...)로 처리
    """
    # 1) 읽기 (cv2.imread 대신 imdecode 사용: 한글 경로 안전)
    data = np.fromfile(str(src_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # PIL 폴백
    if img is None:
        try:
            pil = Image.open(src_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"이미지 로드 실패: {src_path} ({e})")

    # 2) 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) 저장 (imwrite 대신 imencode+tofile: 한글 경로 안전)
    out = src_path.with_suffix(".proc.png")
    ok, buf = cv2.imencode(".png", thr)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")

    # 실제 쓰기
    buf.tofile(str(out))

    # 4) 존재/크기 확인
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"전처리된 파일 저장 실패: {out}")

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

with st.sidebar:
    st.divider()
    st.subheader("이동 메뉴")

    st.markdown(
        "<a href='http://localhost:8502/' target='_blank'>🗂️ 새 탭에서 관리자용 열기</a>",
        unsafe_allow_html=True
    )

with st.form("uploader", clear_on_submit=False):
    col1, col2 = st.columns([1,2])
    with col1:
        customer_id = st.text_input("고객 ID", placeholder="예: 홍길동_31")
    with col2:
        uploads = st.file_uploader(
    "보험 청구 서류 업로드",
    type=["png","jpg","jpeg","pdf","tif","tiff","doc","docx","hwp","bmp"],
    accept_multiple_files=True
)
    submitted = st.form_submit_button("업로드 및 분류 실행")

if submitted:
    # ✅ 0) 유효성 검사
    if not uploads or len(uploads) == 0:
        st.warning("파일을 1개 이상 업로드해 주세요.")
        st.stop()

    if not customer_id.strip():
        st.warning("고객 ID 또는 이름을 입력해 주세요.")
        st.stop()

    # ✅ 1) 입력 형식 검증 (이름+_숫자 형식만 허용)
    # 예시: 홍길동_1, 김민수_23 등
    if not re.match(r"^[가-힣A-Za-z]+_[0-9]+$", customer_id.strip()):
        st.error("❌ 형식 오류: '이름_숫자' 형태로 입력해 주세요. (예: 홍길동_31)")
        st.stop()

    # 2) 고객 폴더 생성
    user_folder = ensure_user_folder(customer_id)

    results = []
    summary = []

    for uploaded in uploads:
        st.markdown(f"---\n**파일:** {uploaded.name}")
        original_path = user_folder / uploaded.name
        save_uploaded_file(uploaded, original_path)
        st.success(f"원본 저장 완료: {original_path.name}")

        ext = original_path.suffix.lower()
        if ext in [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]:
            proc = preprocess_image_to_png(original_path)
            targets = [proc]
        else:
            if ext in [".doc",".docx",".hwp"]:
                st.info("ℹ️ .docx / .hwp는 전처리 없이 OCR API로 전송합니다. 가능하면 PDF 권장.")
            targets = [original_path]

        ocr_texts = []
        for t in targets:
            try:
                with st.spinner(f"OCR 인식 중... ({t.name})"):
                    res = run_upstage_ocr(t)
                ocr_texts.append(res.get("text",""))
            except Exception as e:
                st.error(f"OCR 실패: {e}")
                continue

        merged_text = "\n\n".join(ocr_texts).strip()

        with st.spinner("GPT로 문서 유형 분류 중..."):
            classify = classify_with_gpt(merged_text)

        stem = original_path.stem
        text_path = user_folder / f"{stem}.ocr_text.txt"
        json_path = user_folder / f"{stem}.classification.json"
        text_path.write_text(merged_text, encoding="utf-8")
        json_path.write_text(json.dumps(classify, ensure_ascii=False, indent=2), encoding="utf-8")

        results.append({
            "file": uploaded.name,
            "doc_type": classify.get("doc_type","기타 문서"),
            "confidence": classify.get("confidence",0.0),
            "name": classify.get("key_fields",{}).get("name",""),
            "date": classify.get("key_fields",{}).get("date",""),
            "amount": classify.get("key_fields",{}).get("amount",""),
            "policy_number": classify.get("key_fields",{}).get("policy_number",""),
            "hospital": classify.get("key_fields",{}).get("hospital",""),
        })
        summary.append({
            "file": uploaded.name,
            "ocr_text_path": text_path.name,
            "classification_path": json_path.name,
            "classification": classify
        })

    # 🔸 회차 요약 저장 (관리자 화면이 읽음)
    (user_folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        from db import exec_tx
        import json
        from datetime import datetime

        exec_tx("""
            INSERT INTO documents
            (customer_id, uploaded_at, doc_type, confidence, key_fields,
            original_path, ocr_text_path, classification_json_path, source_ext)
            VALUES
            (:cid, :uploaded_at, :doc_type, :confidence, CAST(:key_fields AS JSON),
            :orig, :ocr, :cls, :ext)
            """, {
            "cid": customer_id,
            "uploaded_at": datetime.now(),
            "doc_type": classify.get("doc_type"),
            "confidence": float(classify.get("confidence") or 0),
            "key_fields": json.dumps(classify.get("key_fields") or {}, ensure_ascii=False),
            "orig": str(original_path),
            "ocr": str(text_path),
            "cls": str(json_path),
            "ext": original_path.suffix.lower().lstrip(".")
        })
        st.success("✅ DB에 저장 완료!")

    except Exception as e:
        st.error(f"❌ DB 저장 실패: {e}")
