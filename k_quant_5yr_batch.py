import os
import time
import random
import re
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from bs4 import BeautifulSoup
import OpenDartReader

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =====================================================================
# 🔑 1. 설정 (종은님 환경에 맞게 키 입력)
# =====================================================================
try:
    import streamlit as st
    DART_API_KEY = st.secrets.get("DART_API_KEY", "0e2903c19927635fc1a59a560b04556c9b414ed8")
except:
    DART_API_KEY = "0e2903c19927635fc1a59a560b04556c9b414ed8"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Referer': 'https://dart.fss.or.kr/', 
    'Connection': 'keep-alive'
}

# =====================================================================
# 🕸️ 2. DART 5개년 사업보고서 리스트 확보
# =====================================================================
def get_5yr_biz_reports(company_name):
    print(f"\n🔍 [{company_name}] 최근 5개년 사업보고서 탐색 중...")
    dart = OpenDartReader(DART_API_KEY)

    # 6년 전 데이터부터 넉넉하게 검색
    start_date = (datetime.now() - pd.Timedelta(days=365 * 6)).strftime('%Y%m%d')

    try:
        filings = dart.list(company_name, start=start_date, kind='A')
        # '사업보고서'라는 글자가 포함된 정기공시만 필터링 (반기/분기 제외)
        biz_reports = filings[filings['report_nm'].str.contains('사업보고서', na=False)]

        # 최신순 정렬 후 상위 5개 가져오기
        biz_reports = biz_reports.sort_values(by='rcept_dt', ascending=False).head(5)

        rcp_list = []
        for _, row in biz_reports.iterrows():
            rcp_no = row['rcept_no']
            year = str(row['rcept_dt'])[:4]
            rcp_list.append((rcp_no, year))

        print(f"✅ 총 {len(rcp_list)}건의 사업보고서를 찾았습니다: {[y for _, y in rcp_list]}")
        return rcp_list
    except Exception as e:
        print(f"⚠️ DART 검색 에러: {e}")
        return []

# =====================================================================
# 🕸️ 3. DART 문서 파싱 (텍스트 변환)
# =====================================================================
def parse_and_save_dart(company_name, report_year, rcept_no):
    file_name = f"report_{company_name}_{report_year}.txt"
    if os.path.exists(file_name):
        print(f"  👉 {report_year}년 파일이 이미 존재합니다. 다운로드를 생략합니다.")
        return file_name

    print(f"  ⏳ {report_year}년 사업보고서 원문 다운로드 및 파싱 중...")
    dart = OpenDartReader(DART_API_KEY)
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
                time.sleep(random.uniform(1.0, 2.0)) # DART 서버 차단 방지 (필수)

                try:
                    resp = session.get(url, timeout=15)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        text_content += f"## {report_year}년 {title}\n\n"
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
    except Exception as e:
        print(f"  ⚠️ {report_year}년 파싱 에러: {e}")
        return None

# =====================================================================
# 🧠 4. 5년치 데이터를 하나의 벡터 DB(Chroma)로 압축
# =====================================================================
def build_5yr_vector_db(company_name, stock_code, file_list):
    print(f"\n🧠 [{company_name}] 5년 치 데이터를 AI 벡터 DB로 압축 중...")
    db_dir = f"./chroma_db/{stock_code}"

    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    all_texts = []

    for file_path in file_list:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            # 5년치 데이터이므로 청크를 촘촘하게 자릅니다
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150) 
            texts = splitter.split_documents(docs)
            all_texts.extend(texts)
        except Exception as e:
            print(f"  ⚠️ {file_path} 로드 에러: {e}")

    if all_texts:
        print(f"  📦 총 {len(all_texts)}개의 데이터 조각(Chunks) 생성 완료! DB에 주입합니다.")
        Chroma.from_documents(all_texts, embeddings, persist_directory=db_dir)
        print(f"🎉 [{company_name}] 5개년 AI 지식 베이스 구축 완벽 성공!")
    else:
        print("❌ 벡터 DB를 구축할 데이터가 없습니다.")

# =====================================================================
# 🔥 5. 실행부
# =====================================================================
def get_stock_code(company_name):
    dart = OpenDartReader(DART_API_KEY)
    try:
        corp = dart.corp_codes[dart.corp_codes['corp_name'] == company_name]
        if not corp.empty and corp.iloc[0]['stock_code']:
            return corp.iloc[0]['stock_code']
    except: pass
    return None

if __name__ == "__main__":
    # 💡 5년 치를 모아둘 기업 리스트를 여기에 적으세요! (예: 포트폴리오 종목들)
    target_companies = ["삼성전자", "한미반도체", "한화에어로스페이스"]

    print("🚀 [K-Quant 5개년 딥다이브 데이터 수집기 가동]\n" + "="*60)

    for company in target_companies:
        stock_code = get_stock_code(company)
        if not stock_code:
            print(f"❌ {company}의 종목코드를 찾을 수 없습니다.")
            continue

        rcp_list = get_5yr_biz_reports(company)
        saved_files = []

        for rcp_no, year in rcp_list:
            file_name = parse_and_save_dart(company, year, rcp_no)
            if file_name:
                saved_files.append(file_name)

        if saved_files:
            build_5yr_vector_db(company, stock_code, saved_files)

        print("="*60)
