import os, json, re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# =========================
# 환경 설정
# =========================
load_dotenv()
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data/users")).resolve()

st.set_page_config(page_title="YMC보험사 직원용 – 고객 문서 조회", page_icon="🗂️", layout="wide")
st.title("🗂️ 고객 문서 조회/확인 대시보드")

# =========================
# 유틸 & 인덱서
# =========================
SYSTEM_FILES = {"ocr_text.txt", "classification.json"}

def parse_ts(ts_folder: Path) -> datetime:
    # 폴더명: YYYYMMDD_HHMMSS
    try:
        return datetime.strptime(ts_folder.name, "%Y%m%d_%H%M%S")
    except Exception:
        # 폴더명이 다르면 폴백: 수정 시각
        return datetime.fromtimestamp(ts_folder.stat().st_mtime)

def read_classification(fp: Path) -> Dict[str, Any]:
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}

def first_user_file(folder: Path) -> Path | None:
    # 고객이 올린 '원본 파일' (시스템 생성물 제외)
    for p in folder.iterdir():
        if p.is_file() and p.name not in SYSTEM_FILES:
            return p
    return None

def safe_read_text(fp: Path) -> str:
    if not fp.exists():
        return ""
    try:
        return fp.read_text(encoding="utf-8")
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def build_index(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(rows)

    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        customer_id = user_dir.name
        for ts_dir in sorted(user_dir.iterdir(), key=parse_ts, reverse=True):
            if not ts_dir.is_dir():
                continue
            ts = parse_ts(ts_dir)
            cls = read_classification(ts_dir / "classification.json")
            doc_type = cls.get("doc_type", "")
            confidence = cls.get("confidence", None)
            key_fields = cls.get("key_fields", {}) or {}
            name = key_fields.get("name", "")
            date = key_fields.get("date", "")
            amount = key_fields.get("amount", "")
            policy = key_fields.get("policy_number", "")

            user_file = first_user_file(ts_dir)
            ocr_text = safe_read_text(ts_dir / "ocr_text.txt")
            rows.append({
                "고객ID": customer_id,
                "업로드시각": ts,
                "문서유형": doc_type,
                "신뢰도": confidence,
                "고객명(추출)": name,
                "일자(추출)": date,
                "원본파일": str(user_file) if user_file else "",
                "OCR텍스트경로": str((ts_dir / "ocr_text.txt").resolve()),
                "분류JSON경로": str((ts_dir / "classification.json").resolve()),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["업로드시각"], ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

# =========================
# 인덱싱 & 사이드바 필터
# =========================
with st.spinner("인덱싱 중..."):
    df = build_index(DATA_ROOT)

if st.sidebar.button("새로고침"):
    build_index.clear()
    st.rerun()

if df.empty:
    st.warning("표시할 데이터가 없습니다. (루트 폴더에 업로드 기록이 있는지 확인하세요)")
    st.stop()

with st.sidebar:
    st.header("🔎 필터")
    custs = ["(전체)"] + sorted(df["고객ID"].unique().tolist())
    sel_cust = st.selectbox("고객 선택", custs)

    doc_types = ["(전체)"] + sorted([x for x in df["문서유형"].dropna().unique().tolist() if x])
    sel_type = st.selectbox("문서 유형", doc_types)

    st.caption("신뢰도(0~1)")
    min_conf, max_conf = st.slider("신뢰도 범위", 0.0, 1.0, (0.0, 1.0), step=0.05)

    q = st.text_input("🔍 키워드 검색 (고객명/일자)")

    refresh = st.button("🔄 인덱스 새로고침", use_container_width=True)
    if refresh:
        build_index.clear()
        df = build_index(DATA_ROOT)
        st.rerun()  # ✅ 최신 Streamlit에서는 이렇게 변경

fdf = df.copy()
if sel_cust != "(전체)":
    fdf = fdf[fdf["고객ID"] == sel_cust]
if sel_type != "(전체)":
    fdf = fdf[fdf["문서유형"] == sel_type]
fdf = fdf[(fdf["신뢰도"].fillna(0) >= min_conf) & (fdf["신뢰도"].fillna(0) <= max_conf)]

if q:
    q = q.strip()
    mask = pd.Series(False, index=fdf.index)
    for col in ["고객명(추출)", "일자(추출)"]:
        mask |= fdf[col].fillna("").str.contains(q, case=False, regex=False)
    fdf = fdf[mask]

st.subheader(f"목록 ({len(fdf):,}건)")
st.dataframe(
    fdf[["고객ID", "업로드시각", "문서유형", "신뢰도", "고객명(추출)", "일자(추출)"]],
    use_container_width=True, height=400
)

# =========================
# 상세 보기
# =========================
st.subheader("상세 보기")
if len(fdf) == 0:
    st.info("좌측 필터를 조정해 문서를 선택하세요.")
else:
    # 최신 1건 기본 선택
    options = [f"[{r['고객ID']}] {r['업로드시각']} · {r['문서유형']} · {Path(r['원본파일']).name if r['원본파일'] else '원본없음'}"
               for _, r in fdf.iterrows()]
    idx = st.selectbox("문서 선택", range(len(options)), format_func=lambda i: options[i])
    row = fdf.iloc[idx]

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**메타/분류 정보**")
        meta = {
            "고객ID": row["고객ID"],
            "업로드시각": row["업로드시각"],
            "문서유형": row["문서유형"],
            "신뢰도": row["신뢰도"],
            "고객명(추출)": row["고객명(추출)"],
            "일자(추출)": row["일자(추출)"],
            "금액(추출)": row["금액(추출)"],
            "증권번호(추출)": row["증권번호(추출)"],
        }
        st.table(pd.DataFrame(meta, index=["값"]).T)

        # 분류 JSON 프리뷰
        try:
            cls_json = json.loads(Path(row["분류JSON경로"]).read_text(encoding="utf-8"))
            with st.expander("분류 Raw JSON 보기", expanded=False):
                st.json(cls_json)
        except Exception:
            st.info("분류 JSON을 읽을 수 없습니다.")

    with c2:
        st.markdown("**원본/텍스트 미리보기**")
        if row["원본파일"] and Path(row["원본파일"]).exists():
            p = Path(row["원본파일"])
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                st.image(str(p), caption=p.name, use_column_width=True)
            else:
                st.caption(f"원본: {p.name} (미리보기 미지원 확장자)")
                st.download_button("원본 다운로드", data=p.read_bytes(), file_name=p.name)
        else:
            st.info("원본 파일이 없습니다.")

        # OCR 텍스트
        txt_path = Path(row["OCR텍스트경로"])
        if txt_path.exists():
            with st.expander("OCR 텍스트 열기", expanded=False):
                st.text_area("OCR 텍스트", txt_path.read_text(encoding="utf-8"), height=250)
                st.download_button("OCR 텍스트 다운로드", data=txt_path.read_text(encoding="utf-8"), file_name="ocr_text.txt")
        else:
            st.info("OCR 텍스트가 없습니다.")

# =========================
# 내보내기 (CSV)
# =========================
st.divider()
csv = fdf.to_csv(index=False).encode("utf-8-sig")
st.download_button("📥 필터된 목록 CSV 다운로드", data=csv, file_name="claims_admin_list.csv")