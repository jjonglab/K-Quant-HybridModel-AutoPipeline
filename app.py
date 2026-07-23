import os
import time
import streamlit as st
import pandas as pd
import pymysql
import yfinance as yf
import numpy as np

# 랭체인 및 AI 관련
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import OpenDartReader
from langchain_classic.chains import LLMChain
from langchain_classic.chains import RetrievalQA

# 팩토리 모델 핵심 함수
from k_quant_factory import get_latest_biz_report_rcp, run_full_pipeline

# ==========================================
# 🔑 1. 환경 설정 (보안 키 완벽 숨김 처리!)
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    TIDB_HOST = st.secrets["TIDB_HOST"]
    TIDB_USER = st.secrets["TIDB_USER"]
    TIDB_PASSWORD = st.secrets["TIDB_PASSWORD"]
    DART_API_KEY = st.secrets["DART_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    TIDB_HOST = os.environ.get("TIDB_HOST", "")
    TIDB_USER = os.environ.get("TIDB_USER", "")
    TIDB_PASSWORD = os.environ.get("TIDB_PASSWORD", "")
    DART_API_KEY = os.environ.get("DART_API_KEY", "")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

st.set_page_config(page_title="K-Quant AI Platform", page_icon="🚀", layout="wide")

# ==========================================
# 🎨 2. UI 디자인 (CSS)
# ==========================================
st.markdown("""
    <style>
    .main-title { background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.8rem; margin-bottom: 0px; padding-bottom: 10px; }
    .hero-banner { background: linear-gradient(135deg, #1e1e26 0%, #2a2a35 100%); border-radius: 16px; padding: 50px 40px; text-align: center; border: 1px solid #3a3a4a; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 30px; }
    .hero-title { font-size: 3rem; font-weight: 900; color: #ffffff; margin-bottom: 15px; letter-spacing: -1px; }
    .hero-subtitle { font-size: 1.2rem; color: #a0a0b0; margin-bottom: 10px; font-weight: 400; }
    .premium-card { background-color: #1e1e26; border-radius: 12px; padding: 25px 20px; text-align: center; border: 1px solid #2d2d3a; box-shadow: 0 4px 12px rgba(0,0,0,0.15); transition: transform 0.2s ease-in-out; height: 100%; }
    .premium-card:hover { transform: translateY(-5px); border-color: #4facfe; }
    .card-title { color: #a0a0b0; font-size: 1.05rem; font-weight: 600; margin-bottom: 12px; }
    .card-value-container { display: flex; justify-content: center; align-items: baseline; gap: 6px; }
    .card-value { color: #ffffff; font-size: 2.5rem; font-weight: 800; margin: 0; line-height: 1; }
    .card-unit { color: #8a8a98; font-size: 1.1rem; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page = "home"
if 'target_company' not in st.session_state: st.session_state.target_company = None

def go_home(): 
    st.session_state.page = "home"
    st.session_state.target_company = None

# ==========================================
# ⚙️ 3. 핵심 백엔드 함수 (DB, API)
# ==========================================
def get_db_connection():
    return pymysql.connect(
        host=TIDB_HOST, user=TIDB_USER, password=TIDB_PASSWORD, 
        db="analysisassistant", charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor
    )

@st.cache_data(ttl=600)
def load_data():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = """
                SELECT m.stock_code, m.company_name, m.sector, q.per, q.pbr, q.roe, q.quant_score, 
                       q.z_score, q.z_status, a.liquidity_summary, a.risk_summary, 
                       b.stock_return, b.benchmark_return, b.alpha, b.evaluation
                FROM company_master m
                LEFT JOIN k_quant_metrics q ON m.stock_code = q.stock_code
                LEFT JOIN ai_report_analysis a ON m.stock_code = a.stock_code
                LEFT JOIN k_quant_backtest_results b ON m.stock_code = b.stock_code
            """
            cursor.execute(sql)
            result = cursor.fetchall()
        conn.close() 
        
        df = pd.DataFrame(result)
        # 💡 [백신] NaN을 0으로 변환하여 화면 깨짐 방지
        if not df.empty:
            df = df.fillna(0)
        return df
    except Exception as e:
        st.error(f"DB 연결 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_market_data():
    market_info = {}
    try:
        ks = yf.Ticker("^KS11").history(period="5d")
        kq = yf.Ticker("^KQ11").history(period="5d")
        usd = yf.Ticker("KRW=X").history(period="5d")
        
        # 💡 [백신] nanpt 방지 처리
        for name, ticker in [("KOSPI", ks), ("KOSDAQ", kq), ("환율(USD/KRW)", usd)]:
            if not ticker.empty:
                val = ticker['Close'].iloc[-1]
                diff = ticker['Close'].iloc[-1] - ticker['Close'].iloc[-2]
                market_info[name] = (0 if pd.isna(val) else val, 0 if pd.isna(diff) else diff)
            else:
                market_info[name] = (0.0, 0.0)
    except: pass
    return market_info

@st.cache_resource
def get_vector_store(company_name, stock_code, target_year="2023"):
    persist_dir = f"./chroma_db/{stock_code}"
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    file_name = f"report_{company_name}_{target_year}.txt"
    if os.path.exists(file_name):
        loader = TextLoader(file_name, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
        texts = splitter.split_documents(docs)
        return Chroma.from_documents(texts, embeddings)
    return None

@st.cache_data
def get_ticker_from_name(company_name):
    try:
        dart = OpenDartReader(DART_API_KEY)
        corp = dart.corp_codes[dart.corp_codes['corp_name'] == company_name]
        if not corp.empty and corp.iloc[0]['stock_code']:
            stock_code = corp.iloc[0]['stock_code']
            # 💡 [백신] 확실한 데이터인지 확인하기 위해 1달치 조회 (코스닥 오류 방지)
            temp_hist = yf.Ticker(f"{stock_code}.KS").history(period="1mo")
            return f"{stock_code}.KS" if not temp_hist.empty and len(temp_hist) > 10 else f"{stock_code}.KQ"
        return None
    except: return None

df = load_data()
if not df.empty:
    df = df[~df['company_name'].str.contains('종목명', na=False)] 
    df = df.drop_duplicates(subset=['stock_code'], keep='last')

# ==========================================
# 🧭 4. 사이드바 네비게이션
# ==========================================
with st.sidebar:
    st.button("🏠 K-Quant 메인 홈", on_click=go_home, use_container_width=True, type="primary")
    st.write("")
    st.markdown("### 🔍 AI 리포트 검색")
    if df.empty:
        st.warning("DB에 분석된 데이터가 없습니다.")
    else:
        company_list = list(df['company_name'] + " (" + df['stock_code'] + ")")
        selected_company = st.selectbox("분석이 완료된 기업 선택", company_list)
        if st.button("📊 상세 리포트 조회", use_container_width=True):
            st.session_state.page = "report"
            st.session_state.target_company = selected_company.split(" (")[0]
            st.rerun()
    st.divider()
    st.caption(f"✓ 현재 수집된 종목: **{len(df)}개**")

# ==========================================
# 🏠 5. 화면 A: K-Quant 메인 홈 
# ==========================================
if st.session_state.page == "home":
    st.markdown("""
        <div class='hero-banner'>
            <div class='hero-title'>K-Quant AI Platform</div>
            <div class='hero-subtitle'>데이터 기반 퀀트 스코어링과 AI 비정형 분석이 결합된 하이브리드 투자 솔루션</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🌐 현재 시장 현황")
    market_data = fetch_market_data()
    if market_data:
        m_cols = st.columns(len(market_data))
        for idx, (key, (val, diff)) in enumerate(market_data.items()):
            with m_cols[idx]:
                unit = "₩" if "환율" in key else "pt"
                st.metric(label=key, value=f"{val:,.2f}{unit}", delta=f"{diff:,.2f}")
    else:
        st.info("야후 파이낸스에서 시장 지수를 불러오고 있습니다...")

    st.divider()
    st.markdown("#### 🚀 On-Demand 실시간 기업 분석")
    main_search = st.text_input("분석하고 싶은 기업명을 정확히 입력하세요 (예: 삼성전자)", placeholder="기업명 입력 후 Enter", key="main_search")

    if main_search:
        db_company_list = df['company_name'].tolist() if not df.empty else []
        if main_search in db_company_list:
            st.success(f"✅ '{main_search}'(은)는 이미 분석이 완료된 기업입니다! 이동합니다.")
            time.sleep(1)
            st.session_state.page = "report"
            st.session_state.target_company = main_search
            st.rerun()
        else:
            with st.status(f"🛠️ '{main_search}' 즉각 AI 분석 공정을 가동합니다!", expanded=True) as status:
                try:
                    st.write("1️⃣ 종목 코드(Ticker) 검색 및 맵핑 중...")
                    target_ticker = get_ticker_from_name(main_search)
                    if not target_ticker:
                        status.update(label="❌ DART 등록 기업을 찾을 수 없습니다.", state="error")
                        st.stop()

                    st.write(f"2️⃣ DART 사업보고서 원문 확보 중... ({target_ticker})")
                    rcp_no, report_year = get_latest_biz_report_rcp(main_search)
                    if not rcp_no:
                        status.update(label="❌ 최신 사업보고서를 찾을 수 없습니다.", state="error")
                        st.stop()

                    st.write(f"3️⃣ Gemini AI가 {report_year}년 사업보고서 정밀 분석 중... (약 1분 소요)")
                    run_full_pipeline(target_ticker, main_search, rcp_no, report_year)
                    load_data.clear() 

                    status.update(label=f"✅ '{main_search}' 분석 완료!", state="complete", expanded=False)
                    time.sleep(1)
                    st.session_state.page = "report"
                    st.session_state.target_company = main_search
                    st.rerun()
                except Exception as e:
                    status.update(label=f"❌ 오류 발생: {e}", state="error")

    st.divider()
    st.markdown("#### 🔎 수집 완료된 기업 스크리너")
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1: screener_query = st.text_input("기업명 검색", "")
    with f_col2: min_score = st.number_input("최소 K-Quant 점수", min_value=0, max_value=100, step=5)
    with f_col3: safe_only = st.checkbox("✅ 재무 안정성 우수/보통만")

    if not df.empty:
        filtered_df = df.copy()
        if screener_query: filtered_df = filtered_df[filtered_df['company_name'].str.contains(screener_query)]
        if min_score > 0: filtered_df = filtered_df[filtered_df['quant_score'] >= min_score]
        if safe_only: filtered_df = filtered_df[filtered_df['z_status'].isin(['안전', '주의'])]

        display_df = filtered_df[['company_name', 'stock_code', 'quant_score', 'z_status', 'per', 'pbr', 'roe']].copy()
        display_df.columns = ['기업명', '종목코드', 'K-Quant 점수', '재무 안정성', 'PER(배)', 'PBR(배)', 'ROE(%)']
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

# ==========================================
# 📈 6. 화면 B: 개별 기업 상세 AI 리포트
# ==========================================
elif st.session_state.page == "report" and st.session_state.target_company:
    target = st.session_state.target_company
    target_df = df[df['company_name'] == target]
    
    if target_df.empty:
        st.error(f"⚠️ '{target}'의 분석 데이터를 찾을 수 없습니다. 처음부터 다시 시도해주세요.")
        if st.button("홈으로 이동"): go_home()
        st.stop()

    data = target_df.iloc[0]
    st.markdown(f"<h1 class='main-title'>{data['company_name']} AI 퀀트 리포트</h1>", unsafe_allow_html=True)
    st.caption(f"종목코드: {data['stock_code']} | 분석 엔진: Gemini & yfinance")
    st.write("")

    z_status = data.get('z_status', '데이터 없음')
    z_score = data.get('z_score', 0.0)

    if pd.isna(z_status) or z_status == '데이터 없음': st.info("💡 Z-Score 분석 미완료")
    elif z_status == '위험': st.error(f"🚨 **[주의] 재무구조 관찰 필요!** Z-Score: **{z_score}점**")
    elif z_status == '주의': st.warning(f"⚠️ **[참고] 재무 건전성 보통.** Z-Score: **{z_score}점**")
    elif z_status == '안전': st.success(f"✅ **[우수] 재무 건전성 훌륭함.** Z-Score: **{z_score}점**")

    st.write("")
    cols = st.columns(5)
    metrics = [
        ("🔥 K-Quant 매력도", data['quant_score'], "점"), ("📈 PER (수익성)", data['per'], "배"),
        ("🏢 PBR (자산가치)", data['pbr'], "배"), ("💰 ROE (자본이익률)", data['roe'], "%"),
        ("⚖️ Z-Score", z_score, "점")
    ]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
                <div class='premium-card'>
                    <div class='card-title'>{metrics[i][0]}</div>
                    <div class='card-value-container'>
                        <div class='card-value'>{metrics[i][1]}</div><div class='card-unit'>{metrics[i][2]}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 실시간 주가 차트", "📑 정밀 분석 리포트", "💬 AI 투자 어시스턴트", "📊 전략 백테스팅 성과"])

    with tab1:
        st.subheader(f"📊 {data['company_name']} 최근 1년 주가 추이")
        try:
            # 💡 [백신] 코스닥 종목도 안전하게 조회
            ticker_symbol = data['stock_code'] + ".KS"
            stock_df = yf.Ticker(ticker_symbol).history(period="1y")
            
            if stock_df.empty or len(stock_df) < 50:
                ticker_symbol = data['stock_code'] + ".KQ"
                stock_df = yf.Ticker(ticker_symbol).history(period="1y")

            if not stock_df.empty:
                st.line_chart(stock_df[['Close']])
                
                # 💡 [백신] NaN 데이터로 인한 에러 방지
                last_price = stock_df['Close'].iloc[-1]
                max_val = stock_df['Close'].max()
                current_price = int(last_price) if not pd.isna(last_price) else 0
                max_price = int(max_val) if not pd.isna(max_val) else 0
                
                st.success(f"**현재가:** {current_price:,}원 | **52주 최고가:** {max_price:,}원")
            else:
                st.warning(f"데이터가 없습니다 (Ticker: {ticker_symbol})")
        except Exception as e: st.warning(f"차트를 불러올 수 없습니다. ({e})")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.subheader("💧 유동성 및 배당 안정성")
                st.write(data['liquidity_summary'] if pd.notna(data['liquidity_summary']) else "데이터 없음")
        with c2:
            with st.container(border=True):
                st.subheader("⚠️ 핵심 리스크 및 위기 요인")
                st.write(data['risk_summary'] if pd.notna(data['risk_summary']) else "데이터 없음")

    with tab3:
        st.subheader(f"🤖 {data['company_name']} 전문 AI 애널리스트")
        if "current_company" not in st.session_state or st.session_state.current_company != target:
            st.session_state.messages = []
            st.session_state.current_company = target

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "source_text" in msg and msg["source_text"]:
                    with st.expander("🔍 AI가 참고한 원문 확인"): st.info(msg["source_text"])

        if prompt := st.chat_input("질문을 입력하세요!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변을 생성 중입니다..."):
                    try:
                        vectorstore = get_vector_store(data['company_name'], data['stock_code'])
                        source_docs_text = ""
                        # 💡 [백신] AI가 문서를 10개까지 넓게 찾도록 시야 확장!
                        if vectorstore:
                            docs = vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(prompt)
                            source_docs_text = "\n\n".join([f"문서 {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

                        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3) 
                        context = f"""
                        [기업: {data['company_name']}, PER: {data['per']}, Z-Score: {z_score}]
                        [AI 요약: {data['liquidity_summary']} / {data['risk_summary']}]
                        [원문: {source_docs_text}]
                        위 데이터를 바탕으로 질문에 답하세요: {prompt}
                        """
                        response = llm.invoke([HumanMessage(content=context)])
                        answer = response.content

                        if isinstance(answer, list):
                            answer = answer[0].get('text', str(answer))

                        st.markdown(answer)

                        if source_docs_text and "답변드리기 어렵습니다" not in answer:
                            with st.expander("🔍 AI가 참고한 원문 확인"):
                                st.info(source_docs_text)

                        st.session_state.messages.append({"role": "assistant", "content": answer, "source_text": source_docs_text})
                    except Exception as e: 
                        st.error(f"오류: {e}")

    with tab4:
        st.subheader(f"📊 {data['company_name']} 백테스트 성과 (vs KOSPI 200)")
        stock_rtn = data.get('stock_return', np.nan)

        if pd.isna(stock_rtn):
            st.warning("⚠️ 백테스트 데이터가 없습니다. 새로운 종목이라면 백테스트 엔진을 돌려주세요!")
        else:
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("1년 수익률", f"{stock_rtn:.2f}%")
            b_col2.metric("벤치마크", f"{data.get('benchmark_return', 0):.2f}%")
            b_col3.metric("초과 수익 (Alpha)", f"{data.get('alpha', 0):.2f}%", delta=data.get('evaluation', ''))

            st.write("")
            st.markdown("##### 📈 1년 누적 수익률 비교 차트")
            with st.spinner("수익률 비교 차트를 생성 중입니다..."):
                try:
                    # 💡 [백신] 코스닥 종목도 안전하게 조회
                    ticker_symbol = data['stock_code'] + ".KS"
                    hist_stock = yf.Ticker(ticker_symbol).history(period="1y")
                    
                    if hist_stock.empty or len(hist_stock) < 50:
                        ticker_symbol = data['stock_code'] + ".KQ"
                        hist_stock = yf.Ticker(ticker_symbol).history(period="1y")
                        
                    bm_symbol = "069500.KS"  
                    hist_bm = yf.Ticker(bm_symbol).history(period="1y")

                    if not hist_stock.empty and not hist_bm.empty:
                        stock_cum = ((hist_stock['Close'] / hist_stock['Close'].iloc[0]) - 1) * 100
                        bm_cum = ((hist_bm['Close'] / hist_bm['Close'].iloc[0]) - 1) * 100

                        chart_df = pd.DataFrame({
                            f"{data['company_name']} 수익률(%)": stock_cum,
                            "KODEX 200 (수익률 %)": bm_cum
                        })
                        st.line_chart(chart_df, color=["#00f2fe", "#555566"])
                except Exception as e:
                    st.error(f"차트 생성 중 오류가 발생했습니다: {e}")
