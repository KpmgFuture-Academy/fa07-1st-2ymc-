import os, json, re, shutil, stat
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
# 간단한 비밀번호 잠금
# =========================
ADMIN_PASSWORD = "1234"

# 세션 상태 초기화
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    st.title("🔐 관리자 로그인")
    password = st.text_input("비밀번호를 입력하세요", type="password")

    if st.button("로그인"):
        if password == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.success("✅ 로그인 성공! 관리자 대시보드로 이동합니다.")
            st.rerun()  # ✅ 최신 Streamlit에서는 이렇게!
        else:
            st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()


# =========================
# 유틸 & 인덱서
# =========================
# ✅ 다중 파일 구조(고객용) 및 레거시 구조(단일 파일) 모두 지원
SYSTEM_FILES = {
    "summary.json",               # 회차 요약 (다중 파일)
}
SYSTEM_SUFFIXES = {
    ".ocr_text.txt",              # 파일별 OCR 텍스트
    ".classification.json",       # 파일별 분류 JSON
    ".proc.png",                  # 전처리 산출물
}
LEGACY_FILES = {"ocr_text.txt", "classification.json"}  # 레거시 단일 구조


def find_original_by_stem(ts_dir: Path, stem: str) -> Path | None:
    cand_exts = [".png",".jpg",".jpeg",".bmp",".tif",".tiff",".pdf",".doc",".docx",".hwp"]
    for ext in cand_exts:
        p = ts_dir / f"{stem}{ext}"
        if p.exists() and not is_system_artifact(p):
            return p
    for p in ts_dir.iterdir():
        if p.is_file() and p.stem == stem and not is_system_artifact(p):
            return p
    return None

def parse_ts(ts_folder: Path) -> datetime:
    try:
        return datetime.strptime(ts_folder.name, "%Y%m%d_%H%M%S")
    except Exception:
        return datetime.fromtimestamp(ts_folder.stat().st_mtime)

def read_classification(fp: Path) -> Dict[str, Any]:
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}

def safe_read_text(fp: Path) -> str:
    if not fp.exists():
        return ""
    try:
        return fp.read_text(encoding="utf-8")
    except Exception:
        return ""

def is_system_artifact(p: Path) -> bool:
    if p.name in SYSTEM_FILES or p.name in LEGACY_FILES:
        return True
    for suf in SYSTEM_SUFFIXES:
        if p.name.endswith(suf):
            return True
    return False

def list_original_files(folder: Path) -> list[Path]:
    files = []
    for p in folder.iterdir():
        if p.is_file() and not is_system_artifact(p):
            files.append(p)
    return sorted(files)

def read_summary(fp: Path) -> list[dict]:
    if not fp.exists():
        return []
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def build_index(root: Path) -> pd.DataFrame:
    """
    우선순위
    1) summary.json (다중 파일 회차)
    2) *.classification.json / *.ocr_text.txt 페어 자동 탐색
    3) legacy: classification.json / ocr_text.txt (단일 파일)
    """
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

            # 1) summary.json 우선
            summ = read_summary(ts_dir / "summary.json")
            if summ:
                for item in summ:
                    cls = item.get("classification", {}) or {}
                    kf  = cls.get("key_fields", {}) or {}
                    orig_path = (ts_dir / item.get("file","")).resolve()
                    ocr_path  = (ts_dir / item.get("ocr_text_path","")).resolve()
                    cls_path  = (ts_dir / item.get("classification_path","")).resolve()
                    rows.append({
                        "고객ID": customer_id,
                        "업로드시각": ts,
                        "문서유형": cls.get("doc_type", ""),
                        "신뢰도": cls.get("confidence", None),
                        "고객명(추출)": kf.get("name", ""),
                        "일자(추출)": kf.get("date", ""),
                        "원본파일": str(orig_path) if orig_path.exists() else "",
                        "OCR텍스트경로": str(ocr_path),
                        "분류JSON경로": str(cls_path),
                    })
                continue  # 이 회차 처리 완료

            # 2) 파일별 페어 자동 탐색 (*.classification.json 기준)
            cls_files = list(ts_dir.glob("*.classification.json"))
            if cls_files:
                for cls_fp in cls_files:
                    stem = cls_fp.name[:-len(".classification.json")]
                    ocr_fp = ts_dir / f"{stem}.ocr_text.txt"
                    orig_fp = find_original_by_stem(ts_dir, stem)

                    cls = read_classification(cls_fp)
                    kf  = cls.get("key_fields", {}) or {}
                    rows.append({
                        "고객ID": customer_id,
                        "업로드시각": ts,
                        "문서유형": cls.get("doc_type",""),
                        "신뢰도": cls.get("confidence", None),
                        "고객명(추출)": kf.get("name",""),
                        "일자(추출)": kf.get("date",""),
                        "원본파일": str(orig_fp.resolve()) if orig_fp else "",
                        "OCR텍스트경로": str(ocr_fp.resolve()),
                        "분류JSON경로": str(cls_fp.resolve()),
                    })
                continue  # 이 회차 처리 완료

            # 3) 레거시 단일 파일 폴백
            cls_legacy = ts_dir / "classification.json"
            ocr_legacy = ts_dir / "ocr_text.txt"
            if cls_legacy.exists() or ocr_legacy.exists():
                originals = [p for p in ts_dir.iterdir() if p.is_file() and not is_system_artifact(p)]
                user_file = originals[0] if originals else None

                cls = read_classification(cls_legacy)
                kf  = cls.get("key_fields", {}) or {}
                rows.append({
                    "고객ID": customer_id,
                    "업로드시각": ts,
                    "문서유형": cls.get("doc_type",""),
                    "신뢰도": cls.get("confidence", None),
                    "고객명(추출)": kf.get("name",""),
                    "일자(추출)": kf.get("date",""),
                    "원본파일": str(user_file.resolve()) if user_file else "",
                    "OCR텍스트경로": str(ocr_legacy.resolve()),
                    "분류JSON경로": str(cls_legacy.resolve()),
                })
            # else: 산출물 없으면 스킵

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["업로드시각"], ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

# =========================
# 인덱싱 & 사이드바 필터
# =========================

if st.sidebar.button("새로고침"):
    build_index.clear()
    st.rerun()

with st.sidebar:
    st.divider()
    st.subheader("이동 메뉴")

    CUSTOMER_URL = "http://localhost:8501/"
    if st.button("🧾 고객용 화면으로 이동"):
        st.markdown(
            f"""
            <meta http-equiv="refresh" content="0; url={CUSTOMER_URL}">
            """,
            unsafe_allow_html=True
        )
        st.info("고객용 화면으로 이동 중입니다...")

with st.spinner("인덱싱 중..."):
    df = build_index(DATA_ROOT)


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

    q = st.text_input("🔍 키워드 검색 (고객명/일자/파일명)")

    refresh = st.button("🔄 인덱스 새로고침", use_container_width=True)
    if refresh:
        build_index.clear()
        df = build_index(DATA_ROOT)
        st.rerun()

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
    # 파일명도 검색에 포함
    if "원본파일" in fdf.columns:
        mask |= fdf["원본파일"].fillna("").str.contains(q, case=False, regex=False)
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
            ext = p.suffix.lower()
            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                st.image(str(p), caption=p.name, use_container_width=True)
            else:
                st.caption(f"원본: {p.name} (미리보기 미지원 확장자)")
            st.download_button("원본 다운로드", data=p.read_bytes(), file_name=p.name)
        else:
            st.info("원본 파일이 없습니다.")

        # OCR 텍스트
        txt_path = Path(row["OCR텍스트경로"])
        if txt_path.exists():
            with st.expander("OCR 텍스트 열기", expanded=False):
                txt = txt_path.read_text(encoding="utf-8")
                st.text_area("OCR 텍스트", txt, height=250)
                st.download_button("OCR 텍스트 다운로드", data=txt, file_name=txt_path.name)
        else:
            st.info("OCR 텍스트가 없습니다.")

# =========================
# 내보내기 (CSV)
# =========================
st.divider()
csv = fdf.to_csv(index=False).encode("utf-8-sig")
st.download_button("📥 필터된 목록 CSV 다운로드", data=csv, file_name="claims_admin_list.csv")