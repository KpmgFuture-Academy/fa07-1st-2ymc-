import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# MySQL 연결 URL (예: mysql+pymysql://user:password@localhost:3306/insurance_db?charset=utf8mb4)
DB_URL = os.getenv("MYSQL_URL")

if not DB_URL:
    raise ValueError("❌ MYSQL_URL 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

# SQLAlchemy 엔진 생성
engine = create_engine(DB_URL, pool_recycle=3600, pool_pre_ping=True)


def exec_tx(query: str, params: dict = None):
    """트랜잭션 단위 SQL 실행 (INSERT/UPDATE/DELETE 등)"""
    if params is None:
        params = {}
    with engine.begin() as conn:  # 자동 커밋/롤백 관리
        conn.execute(text(query), params)


def fetch_df(query: str, params: dict = None):
    """SELECT 쿼리 실행 후 Pandas DataFrame 반환"""
    import pandas as pd
    if params is None:
        params = {}
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    return df


def fetch_one(query: str, params: dict = None):
    """SELECT 쿼리 단일 행 반환"""
    if params is None:
        params = {}
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        row = result.fetchone()
    return dict(row._mapping) if row else None
