# 🚀 K-Quant AI Platform

**데이터 기반 퀀트 스코어링과 AI 비정형 분석이 결합된 하이브리드 투자 솔루션**

방대한 기업 사업보고서를 AI가 심층 분석하고, 핵심 재무 지표와 백테스트 결과를 실시간으로 제공하는 스마트 금융 어시스턴트입니다.

<img width="2880" height="1446" alt="image" src="https://github.com/user-attachments/assets/126e1e1f-fdfc-4489-9c9f-e7087a7eb481" />


---

## ✨ 주요 기능 (Key Features)

### 🤖 RAG 기반 AI 애널리스트 챗봇
* DART 사업보고서를 실시간으로 파싱 및 벡터화하여 질문에 답변
* 할루시네이션(거짓 정보) 방지를 위한 원문 교차 검증(Source Check) 기능 제공

### 📊 퀀트 스코어링 & 리스크 스캐닝
* PER, PBR, ROE 기반의 자체 K-Quant 매력도 산출
* Altman Z-Score 모델을 적용한 기업 재무 건전성(파산 위험도) 자동 스캐닝

### 📈 포트폴리오 백테스팅 파이프라인
* KOSPI 200 벤치마크 대비 1년 누적 초과 수익률(Alpha) 실시간 추적 및 시각화

### 🏗️ 완전 자동화 데이터 팩토리
* 정량 데이터(yfinance)와 정성 데이터(OpenDartReader)를 병렬 수집하여 TiDB 클라우드에 자동 적재

---

## 💻 핵심 구현 로직 (Core Implementation)

### 1. DART 사업보고서 자동 파싱 및 벡터화 (Data Factory)
웹 스크래핑 기술(`OpenDartReader`, `BeautifulSoup`)을 활용하여 방대한 DART 사업보고서 원문에서 '이사의 경영진단', '사업의 내용' 등 투자 핵심 섹션만을 정밀하게 추출하고 마크다운(Markdown) 형태로 변환합니다. 정제된 텍스트는 LangChain의 텍스트 스플리터로 의미 단위 청킹(Chunking)을 거친 후, 한국어 임베딩 모델(`ko-sroberta-multitask`)을 통해 Chroma 벡터 데이터베이스(Vector DB)에 저장되어 빠르고 정확한 검색 환경을 제공합니다.

### 2. 하이브리드 RAG 프롬프트 엔지니어링 (AI Assistant)
단순 문답형 챗봇을 넘어, 구조화된 정량 데이터와 비정형 텍스트를 동시에 처리하는 하이브리드 RAG 아키텍처를 구현했습니다. DB에서 실시간으로 조회한 정량적 퀀트 지표(PER, ROE 등)와 재무 안정성 스코어를 1차 컨텍스트로, 벡터 DB에서 검색된 사업보고서 원문을 2차 컨텍스트로 프롬프트에 유기적으로 주입하여, LLM(Google Gemini)이 종합적이고 신뢰도 높은 투자 인사이트를 도출하도록 설계했습니다.

### 3. Altman Z-Score 기반 기업 파산 위험도 진단 (Quant Scoring)
실시간 주가 및 재무 데이터(`yfinance`)를 기반으로 기업의 재무 건전성을 자동 진단하는 알고리즘을 적용했습니다. 대차대조표와 손익계산서에서 유동자산, 잉여금, 총자산, EBIT 등의 핵심 재무 항목을 추출한 뒤, 파산 예측 모델인 Altman Z-Score 공식을 연산하여 해당 기업의 현재 리스크 수준을 3단계(안전/주의/위험)로 직관적으로 스캐닝합니다.

### 4. 클라우드 DB 자동 적재 방어 로직 (Cloud Infrastructure)
대규모 크롤링 및 분석 데이터를 분산 클라우드 DB(TiDB)에 안정적으로 적재하기 위한 방어적 프로그래밍 패턴을 적용했습니다. 백테스트 결과나 정성 분석 데이터를 병합할 때 발생할 수 있는 외래키(Foreign Key) 참조 무결성 에러를 방지하고자, 데이터 삽입 전 마스터 테이블에 미확인 종목을 자동 선등록(INSERT IGNORE)하고 UPSERT(ON DUPLICATE KEY UPDATE) 쿼리를 활용해 데이터 충돌 및 중복을 원천 차단했습니다.

---

## 🛠️ 기술 스택 (Tech Stack)

| Category | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit, CSS |
| **Backend** | Python, LangChain, HuggingFace (ko-sroberta-multitask) |
| **Database** | TiDB Cloud (MySQL), Chroma (Vector DB) |
| **AI / LLM** | Google Gemini 1.5 Flash |
| **APIs** | yfinance, OpenDartReader |

---

## ⚙️ 시스템 아키텍처 (System Architecture)

* **Data Collection**: yfinance를 통한 주가/재무 데이터 및 DART API를 통한 사업보고서 텍스트 수집
* **Processing**: 텍스트 청킹(Chunking) 후 HuggingFace 임베딩을 통해 Chroma DB에 벡터화
* **Storage**: 추출된 정량/정성 요약 데이터를 TiDB 클라우드 마스터 테이블에 병합
* **Serving**: Streamlit UI를 통해 사용자에게 데이터 시각화 및 LLM 심층 챗봇 서비스 제공

---

## 🚀 시작하기 (Getting Started)

### 1. Repository 클론
```bash
git clone [https://github.com/jjonglab/k-quant-platform.git](https://github.com/jjonglab/k-quant-platform.git)
cd k-quant-platform
