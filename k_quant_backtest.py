import os
import time
import pandas as pd
import pymysql
import yfinance as yf
from datetime import datetime, timedelta

# =====================================================================
# 🔑 1. DB 설정 (GitHub & Streamlit Secrets 연동 최적화)
# =====================================================================
try:
    import streamlit as st
    TIDB_HOST = st.secrets["TIDB_HOST"]
    TIDB_USER = st.secrets["TIDB_USER"]
    TIDB_PASSWORD = st.secrets["TIDB_PASSWORD"]
except Exception:
    TIDB_HOST = os.environ.get("TIDB_HOST", "")
    TIDB_USER = os.environ.get("TIDB_USER", "")
    TIDB_PASSWORD = os.environ.get("TIDB_PASSWORD", "")

DB_CONFIG = {
    "host": TIDB_HOST,        
    "port": 4000,
    "user": TIDB_USER,        
    "password": TIDB_PASSWORD,
    "database":"analysisassistant", 
    "charset": "utf8mb4",
    "autocommit": True,
    "ssl_verify_cert": True,
    "ssl_verify_identity": True
}

# =====================================================================
# 2. 멀티 백테스트 엔진 (일괄 다운로드 최적화)
# =====================================================================
def run_multi_backtest(target_tickers, benchmark_ticker="069500.KS", years_back=1):
    print(f"\n🚀 K-Quant 백테스팅 엔진 가동 (과거 {years_back}년)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    results = []

    # 벤치마크 계산
    try:
        bm_hist = yf.Ticker(benchmark_ticker).history(start=start_str, end=end_str)
        if not bm_hist.empty:
            bm_past = bm_hist['Close'].iloc[0]
            bm_curr = bm_hist['Close'].iloc[-1]
            benchmark_return = ((bm_curr - bm_past) / bm_past) * 100
            print(f"📊 벤치마크 1년 수익률: {benchmark_return:.2f}%\n")
        else:
            benchmark_return = 0.0
    except Exception as e:
        print(f"벤치마크 오류: {e}")
        benchmark_return = 0.0

    if not target_tickers:
        return None

    # 대용량 일괄 다운로드
    tickers_str = " ".join(target_tickers)
    print(f"📦 주가 데이터 일괄 다운로드 중... ({len(target_tickers)}개 종목)")
    try:
        bulk_data = yf.download(tickers_str, start=start_str, end=end_str)['Close']
    except Exception as e:
        print(f"❌ 데이터 일괄 다운로드 에러: {e}")
        return None

    for ticker in target_tickers:
        try:
            if len(target_tickers) == 1:
                hist_close = bulk_data.dropna()
            else:
                hist_close = bulk_data[ticker].dropna()

            if hist_close.empty:
                continue

            past_price = hist_close.iloc[0]
            current_price = hist_close.iloc[-1]
            return_rate = ((current_price - past_price) / past_price) * 100
            alpha = return_rate - benchmark_return

            results.append({
                "종목코드": ticker,
                "1년 전 주가": f"{past_price:,.0f}",
                "현재 주가": f"{current_price:,.0f}",
                "종목 수익률(%)": round(return_rate, 2),
                "벤치마크(%)": round(benchmark_return, 2),
                "초과수익(Alpha %)": round(alpha, 2),
                "평가": "시장 승리 🏆" if alpha > 0 else "시장 하회 📉"
            })
        except Exception as e:
            print(f"  ❌ 에러 ({ticker}): {e}")

    if results:
        df = pd.DataFrame(results)
        print("\n💡 [K-Quant 백테스팅 결과 요약]")
        print(df[['종목코드', '종목 수익률(%)', '초과수익(Alpha %)', '평가']].to_string(index=False))
        return df
    return None

# =====================================================================
# 3. DB 저장 엔진 (안전장치 포함)
# =====================================================================
def insert_backtest_results_to_db(df_results, years_back=1):
    if df_results is None or df_results.empty: return

    print("\n☁️ TiDB 클라우드 DB 저장 시작...")
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("USE analysisassistant;")

        success_count = 0
        for index, row in df_results.iterrows():
            raw_ticker = row['종목코드']
            stock_code = raw_ticker.replace('.KS', '').replace('.KQ', '')

            cursor.execute("""
                INSERT IGNORE INTO company_master (stock_code, company_name, sector)
                VALUES (%s, %s, '미분류')
            """, (stock_code, f"종목명_{stock_code}"))

            insert_sql = """
                INSERT INTO k_quant_backtest_results 
                (stock_code, backtest_date, stock_return, benchmark_return, alpha, evaluation)
                VALUES (%s, CURDATE(), %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                backtest_date=CURDATE(), stock_return=%s, benchmark_return=%s, alpha=%s, evaluation=%s;
            """
            cursor.execute(insert_sql, (
                stock_code, 
                row['종목 수익률(%)'], row['벤치마크(%)'], row['초과수익(Alpha %)'], row['평가'],
                row['종목 수익률(%)'], row['벤치마크(%)'], row['초과수익(Alpha %)'], row['평가']
            ))
            success_count += 1

        cursor.close()
        conn.close()
        print(f"🎉 총 {success_count}건의 백테스트 데이터 클라우드 적재 완료!")
    except Exception as e:
        print(f"⚠️ DB 연동 실패: {e}")

# =====================================================================
# 4. 🔥 단독 실행 트리거 (테스트용)
# =====================================================================
if __name__ == "__main__":
    portfolio_list = [
        "005930.KS", "298040.KS"
    ]
    report_df = run_multi_backtest(portfolio_list, years_back=1)
    insert_backtest_results_to_db(report_df, years_back=1)
