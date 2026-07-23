import os
import time
import random  
import re
import pandas as pd
import requests
import pymysql
import yfinance as yf
import OpenDartReader
from datetime import datetime
from bs4 import BeautifulSoup
import streamlit as st

# 최신 LangChain 라이브러리로 통일 (app.py와 완벽 호환)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.chains import LLMChain

from k_quant_backtest import run_multi_backtest, insert_backtest_results_to_db

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    TIDB_HOST = st.secrets["TIDB_HOST"]
    TIDB_USER = st.secrets["TIDB_USER"]
    TIDB_PASSWORD = st.secrets["TIDB_PASSWORD"]
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    TIDB_HOST = os.environ.get("TIDB_HOST", "")
    TIDB_USER = os.environ.get("TIDB_USER", "")
    TIDB_PASSWORD = os.environ.get("TIDB_PASSWORD", "")

DART_API_KEY = "0e2903c19927635fc1a59a560b04556c9b414ed8" 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Referer': 'https://dart.fss.or.kr/', 
    'Connection': 'keep-alive'
}

# ==========================================
# ⚙️ 2. DB 연결 도우미 (중복 코드 제거)
# ==========================================
def get_db_connection():
    return pymysql.connect(
        host=TIDB_HOST, 
        port=4000,
        user=TIDB_USER, 
        password=TIDB_PASSWORD,
        database="analysisassistant",
        charset="utf8mb4",
        autocommit=True,
        ssl_verify_cert=True,
        ssl_verify_identity=True
    )

def init_database():
    print("🏗️ 클라우드 DB 인프라 점검 및 테이블 초기화 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS analysisassistant;")
        cursor.execute("USE analysisassistant;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_master (
                stock_code VARCHAR(20) PRIMARY KEY, company_name VARCHAR(100) NOT NULL, sector VARCHAR(100)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS k_quant_metrics (
                stock_code VARCHAR(20) PRIMARY KEY, base_date DATE, per FLOAT, pbr FLOAT, roe FLOAT, 
                quant_score FLOAT, z_score FLOAT, z_status VARCHAR(20),
                FOREIGN KEY (stock_code) REFERENCES company_master(stock_code) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_report_analysis (
                stock_code VARCHAR(20) PRIMARY KEY, report_year INT, report_type VARCHAR(50), 
                liquidity_summary TEXT, risk_summary TEXT,
                FOREIGN KEY (stock_code) REFERENCES company_master(stock_code) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS k_quant_backtest_results (
                stock_code VARCHAR(20) PRIMARY KEY, backtest_date DATE, stock_return FLOAT, 
                benchmark_return FLOAT, alpha FLOAT, evaluation VARCHAR(20),
                FOREIGN KEY (stock_code) REFERENCES company_master(stock_code) ON DELETE CASCADE
            )
        """)
        cursor.close()
        conn.close()
        print("✅ 클라우드 DB 테이블 세팅 완료!\n")
    except Exception as e:
        print(f"⚠️ 테이블 생성 에러: {e}")

# ==========================================
# 📊 3. 퀀트 및 정량 지표 스캐닝
# ==========================================
def fetch_quant_score(ticker_symbol):
    print(f"📈 [1/5] K-Quant 실시간 데이터 수집 중... ({ticker_symbol})")
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        per = info.get('trailingPE') or 0
        pbr = info.get('priceToBook') or 0
        roe = (info.get('returnOnEquity') or 0) * 100
        ocf = info.get('operatingCashflow') or 0
        ni = info.get('netIncomeToCommon') or 0

        score = 0
        if 0 < per <= 10: score += 25
        elif 10 < per <= 15: score += 15
        elif 15 < per <= 20: score += 5
        if 0 < pbr <= 1.0: score += 25
        elif 1.0 < pbr <= 1.5: score += 15
        elif 1.5 < pbr <= 2.0: score += 5
        if roe >= 15: score += 25
        elif roe >= 10: score += 15
        elif roe >= 5: score += 5
        if ocf > ni and ni > 0: score += 25
        elif ocf > 0: score += 10

        if per == 0: per = 15.0
        if pbr == 0: pbr = 1.2

        return round(per, 2), round(pbr, 2), round(roe, 2), round(score, 2)
    except: return 15.0, 1.2, 10.0, 50.0  

def calculate_altman_z_score(ticker_symbol):
    print(f"🚨 [2/5] Z-Score 재무 건전성 스캐닝 중... ({ticker_symbol})")
    try:
        ticker = yf.Ticker(ticker_symbol)
        bs = ticker.balance_sheet
        inc = ticker.financials
        info = ticker.info
        if bs.empty or inc.empty: return 0.0, "데이터 누락"

        total_assets = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else 1
        current_assets = bs.loc['Current Assets'].iloc[0] if 'Current Assets' in bs.index else 0
        current_liabilities = bs.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in bs.index else 0
        retained_earnings = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0

        if 'Total Liabilities Net Minority Interest' in bs.index:
            total_liabilities = bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
        else:
            total_liabilities = total_assets - (bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else 1)

        ebit = inc.loc['EBIT'].iloc[0] if 'EBIT' in inc.index else 0
        total_revenue = inc.loc['Total Revenue'].iloc[0] if 'Total Revenue' in inc.index else 0
        market_cap = info.get('marketCap', 0)

        x1 = (current_assets - current_liabilities) / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
        x5 = total_revenue / total_assets

        z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)

        if z_score >= 2.99: status = "안전"
        elif 1.81 <= z_score < 2.99: status = "주의"
        else: status = "위험"
        return round(z_score, 2), status
    except: return 0.0, "계산 에러"

# ==========================================
# 🕸️ 4. DART 파싱 및 RAG 분석 모듈
# ==========================================
def parse_dart_to_markdown(company_name, report_year, rcept_no):
    print(f"🕸️ [3/5] DART 스마트 파싱 중... ({company_name})")
    dart = OpenDartReader(DART_API_KEY)
    file_name = f"report_{company_name}_{report_year}.txt"
    text_content = f"=== [{company_name}] {report_year}년 사업보고서 ===\n\n"
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        sub_docs = dart.sub_docs(rcept_no)
        targets = ["이사의 경영진단", "사업의 내용", "주요 제품", "원재료", "연구개발", "투자자 보호", "배당"]
        for keyword in targets:
            doc = sub_docs[sub_docs['title'].str.contains(keyword)]
            if not doc.empty:
                title = doc.iloc[0]['title']
                url = doc.iloc[0]['url']
                time.sleep(random.uniform(1.5, 3.5)) 
                try:
                    resp = session.get(url, timeout=15)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        text_content += f"## {title}\n\n"
                        for el in soup.body.children:
                            if el.name == 'table':
                                try:
                                    df = pd.read_html(str(el))[0]
                                    text_content += df.to_markdown(index=False) + "\n\n"
                                except: text_content += el.get_text(separator=' ', strip=True) + "\n\n"
                            elif el.name in ['p', 'div'] and el.text.strip():
                                text_content += re.sub(r'\s+', ' ', el.get_text(strip=True)) + "\n"
                        text_content += "\n\n"
                except: pass
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(text_content)
        return file_name
    except: return None

# 💡 [핵심 최적화] AI 분석 시 '종목코드(stock_code)'를 받아서 올바른 위치에 벡터DB를 저장합니다!
def run_ai_analysis(file_path, stock_code):
    print("🤖 [4/5] Gemini 하이브리드 RAG 분석 가동 중...")
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150) # app.py와 일치시킴
        texts = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

        # 💡 app.py의 챗봇이 찾을 수 있는 'chroma_db/종목코드' 위치에 저장!
        db_dir = f"./chroma_db/{stock_code}"
        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=db_dir)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        q_liquidity = "재무 상태표와 배당 정책을 고려할 때, 회사의 유동성(단기 채무 상환 능력) 및 재무 안정성을 요약해 줘."
        q_risk = "이사의 경영진단 및 투자자 보호 파트에서 언급된 실적 하락 요인이나 중대한 소송/우발채무 리스크를 요약해 줘."

        liq_ans = qa.invoke(q_liquidity)['result']
        time.sleep(10) # API 방어 (30초는 너무 길어서 단축)
        risk_ans = qa.invoke(q_risk)['result']
        return liq_ans, risk_ans
    except Exception as e:
        print(f"⚠️ AI 분석 에러: {e}")
        return "분석 실패", "분석 실패"

# ==========================================
# 💾 5. 클라우드 DB 적재 및 유틸 모듈 
# ==========================================
def save_to_database(stock_code, comp_name, per, pbr, roe, score, z_score, z_status, liq_ans, risk_ans, report_year):
    print(f"☁️ [5/5] TiDB 클라우드 서버에 적재 중... (상태: {z_status})")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("USE analysisassistant;")

        cursor.execute("""
            INSERT IGNORE INTO company_master (stock_code, company_name, sector)
            VALUES (%s, %s, '미분류')
        """, (stock_code, comp_name))

        cursor.execute("""
            INSERT INTO k_quant_metrics (stock_code, base_date, per, pbr, roe, quant_score, z_score, z_status)
            VALUES (%s, CURDATE(), %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE base_date=CURDATE(), per=%s, pbr=%s, roe=%s, quant_score=%s, z_score=%s, z_status=%s;
        """, (stock_code, per, pbr, roe, score, z_score, z_status, per, pbr, roe, score, z_score, z_status))

        cursor.execute("""
            INSERT INTO ai_report_analysis (stock_code, report_year, report_type, liquidity_summary, risk_summary)
            VALUES (%s, %s, '사업보고서', %s, %s)
            ON DUPLICATE KEY UPDATE liquidity_summary=%s, risk_summary=%s, report_year=%s;
        """, (stock_code, report_year, liq_ans, risk_ans, liq_ans, risk_ans, report_year))

        cursor.close()
        conn.close()
        print(f"🎉 [성공] {comp_name} 클라우드 저장 완료!")
    except Exception as e:
        print(f"⚠️ 클라우드 DB 저장 에러: {e}")

def get_latest_biz_report_rcp(company_name):
    dart = OpenDartReader(DART_API_KEY)
    start_date = (datetime.now() - pd.Timedelta(days=730)).strftime('%Y%m%d')
    try:
        filings = dart.list(company_name, start=start_date, kind='A')
        biz_reports = filings[filings['report_nm'].str.contains('사업보고서', na=False)]
        if not biz_reports.empty:
            row = biz_reports.iloc[0]
            return row['rcept_no'], str(row['rcept_dt'])[:4] 
        return None, None 
    except: return None, None

def get_ticker_from_name(company_name):
    try:
        dart = OpenDartReader(DART_API_KEY)
        corp = dart.corp_codes[dart.corp_codes['corp_name'] == company_name]
        if not corp.empty and corp.iloc[0]['stock_code']:
            stock_code = corp.iloc[0]['stock_code']
            temp_hist = yf.Ticker(f"{stock_code}.KS").history(period="1d")
            return f"{stock_code}.KS" if not temp_hist.empty else f"{stock_code}.KQ"
        return None
    except: return None

# ==========================================
# 🎯 6. 메인 파이프라인 컨트롤러
# ==========================================
def run_full_pipeline(target_ticker, target_company, dart_rcept_no, report_year):
    print("=" * 60)
    print(f"🚀 [K-Quant 클라우드 자동화 파이프라인] {target_company} ({report_year}년)")
    print("=" * 60)

    per, pbr, roe, quant_score = fetch_quant_score(target_ticker)
    z_score, z_status = calculate_altman_z_score(target_ticker)

    parsed_file = parse_dart_to_markdown(target_company, report_year, dart_rcept_no)
    db_code = target_ticker.replace('.KS', '').replace('.KQ', '') # 순수 종목코드 추출

    if parsed_file:
        # 💡 [최적화 연동] AI 분석 시 순수 종목코드를 넘겨주어 경로를 맞춤!
        liq_summary, risk_summary = run_ai_analysis(parsed_file, db_code) 

        save_to_database(db_code, target_company, per, pbr, roe, quant_score, z_score, z_status, liq_summary, risk_summary, report_year)

        # ==============================================================
        # 💡 [새로 추가된 부분] 분석 끝난 김에 백테스트도 바로 돌려서 꽂아넣기!
        # ==============================================================
        print(f"📊 [백테스트 연동] {target_company} 1년 성과 분석 및 DB 적재 중...")
        backtest_df = run_multi_backtest([target_ticker], years_back=1)
        insert_backtest_results_to_db(backtest_df, years_back=1)
        # ==============================================================

    else:
        print("❌ 사업보고서 파싱 실패로 중단.")

if __name__ == "__main__":
    init_database()

    # 단독 테스트를 위한 포트폴리오
    portfolio = [
        {"ticker": "005930.KS", "company": "삼성전자"}
    ]

    print("🚀 [클라우드 팩토리 가동 시작]")
    for item in portfolio:
        rcp_no, report_year = get_latest_biz_report_rcp(item["company"])
        if rcp_no:
            run_full_pipeline(item["ticker"], item["company"], rcp_no, report_year)
            time.sleep(5) 

    print("\n" + "=" * 60)
    print("🎉 [팩토리 가동 종료] 모든 데이터가 클라우드에 성공적으로 적재되었습니다.")
    print("=" * 60)
