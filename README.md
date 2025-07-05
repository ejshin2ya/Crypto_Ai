# Crypto_Ai
# 1. 프로젝트 폴더 생성 및 vs code 열기
```
mkdir crypto_pullback_ai
cd crypto_pullback_ai
code .
```
# 2. vs code 터미널에서 conda 환경 설정 및 생성
```
conda create -n crypto_pullback_ai python=3.9
conda activate crypto_pullback_ai 
```
# 3. 최적화된 프로젝트 구조 만들기
<img width="516" alt="image" src="https://github.com/user-attachments/assets/ad774d93-078a-4c2e-9d7e-d28ff9821866" />


```
cat > setup_project.sh << 'EOF'
#!/bin/bash
mkdir -p {data/{processed,raw,exports},src,notebooks,tests,config,models/{saved_models,configs},sql}
touch main.py requirements.txt .env .gitignore README.md
touch src/{__init__.py,data_collector.py,database_manager.py,models.py,utils.py}
touch notebooks/{01_data_exploration.ipynb,02_model_development.ipynb,03_backtesting.ipynb,04_results_analysis.ipynb}
touch tests/{test_data.py,test_models.py}
touch config/{database.py,settings.py}
touch sql/{create_tables.sql,sample_queries.sql}
echo "프로젝트 구조 생성 완료!"
EOF

chmod +x setup_project.sh
./setup_project.sh
```

# 4. colab 사용시 패키지 설치 전략
## (1)vs code 설치
```
### 데이터 수집 및 PostgreSQL 관련 (로컬에서만 사용)
conda install pandas numpy matplotlib seaborn # 데이터 분석 및 처리 # 수치 계산 # 그래프 생성 # 고급 시각화
pip install ccxt python-binance psycopg2-binary sqlalchemy python-dotenv # 거래소 API 통합 # 바이낸스 전용 API # PostgreSQL 연결 # ORM (데이터베이스 추상화) # .env 파일 읽기 # 작업 스케줄링 # 진행률 표시
``` 
## (2) colab 설치
```
### 각 Colab 노트북 첫 번째 셀에서
!pip install ccxt python-binance psycopg2-binary sqlalchemy
!pip install optuna scikit-optimize
!pip install ta-lib  # 기술적 지표 라이브러리
# 이미 설치된 패키지들 (Colab 기본 제공)
# - pandas, numpy, matplotlib, seaborn
# - jupyter, tensorflow, scikit-learn
```
# 5. 환경변수 파일 설정
### .env 파일 내용(code .env)
```
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crypto_research
DB_USER=postgres
DB_PASSWORD=your_password_here

# Binance API (나중에 실거래용)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Application Settings
LOG_LEVEL=INFO
DATA_UPDATE_INTERVAL=300  # 5분
```
# 6. 기본설정 파일 작성
### config/settings.py 내용(code config/settings.py)
```
"""
기본 설정 파일
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 데이터베이스 설정
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'crypto_research'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# 바이낸스 API 설정
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY', ''),
    'api_secret': os.getenv('BINANCE_API_SECRET', '')
}

# 애플리케이션 설정
APP_CONFIG = {
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'data_update_interval': int(os.getenv('DATA_UPDATE_INTERVAL', 300))
}

# 거래 설정
TRADING_CONFIG = {
    'symbols': ['BTC/USDT', 'ETH/USDT'],
    'timeframe': '5m',
    'max_position_size': 0.1,  # 10%
    'stop_loss': -0.01,        # -1%
    'take_profit': 0.02        # 2%
}
```
# 7. 데이터 수집기 완성
### src/data_collector.py 내용(code src/data_collector.py)
```
"""
바이낸스 데이터 수집기
"""
import ccxt
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from config.settings import BINANCE_CONFIG, TRADING_CONFIG
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinanceDataCollector:
    """바이낸스 선물 데이터 수집기"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_CONFIG['api_key'],
            'secret': BINANCE_CONFIG['api_secret'],
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        logger.info("바이낸스 데이터 수집기 초기화 완료")
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            markets = self.exchange.load_markets()
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            logger.info(f"연결 테스트 성공 - BTC 현재가: ${ticker['last']:,.2f}")
            return True
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return False
    
    def collect_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """과거 데이터 수집"""
        logger.info(f"{symbol} 과거 {days}일 데이터 수집 시작")
        
        try:
            # 수집할 데이터 계산
            total_candles = days * 24 * 12  # 5분봉 기준
            batches = (total_candles // 1000) + 1
            
            all_data = []
            
            for batch in tqdm(range(batches), desc=f"{symbol} 데이터 수집"):
                try:
                    # 시작 시간 계산
                    since = int((datetime.now() - timedelta(days=days) + timedelta(minutes=batch*1000*5)).timestamp() * 1000)
                    
                    # 데이터 요청
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe='5m',
                        since=since,
                        limit=1000
                    )
                    
                    if ohlcv:
                        all_data.extend(ohlcv)
                    
                    # API 제한 준수
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"배치 {batch} 수집 실패: {e}")
                    continue
            
            if all_data:
                # DataFrame 생성
                df = pd.DataFrame(all_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                
                # 중복 제거 및 정렬
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # 시간 변환
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                
                logger.info(f"{symbol} 데이터 수집 완료: {len(df)}개 캔들")
                return df
            else:
                logger.warning(f"{symbol} 데이터 수집 실패")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"{symbol} 데이터 수집 중 오류: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"{symbol} 현재 가격 조회 실패: {e}")
            return 0.0

# 테스트 실행
if __name__ == "__main__":
    collector = BinanceDataCollector()
    
    if collector.test_connection():
        # 테스트 데이터 수집
        df = collector.collect_historical_data('BTC/USDT', days=1)
        if not df.empty:
            print(f"✅ 데이터 수집 성공: {len(df)}개 캔들")
            print(df.head())
            print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
        else:
            print("❌ 데이터 수집 실패")
    else:
        print("❌ 연결 실패")
```
# 8. 첫 번째 데이터 수집 테스트
```
# 데이터 수집 테스트
python src/data_collector.py
```

# 9.PostgreSQL 연결 준비
### config/database.py 내용(code config/database.py)
```
"""
PostgreSQL 데이터베이스 설정
"""
import psycopg2
from sqlalchemy import create_engine
from config.settings import DATABASE_CONFIG
import logging

logger = logging.getLogger(__name__)

def get_database_url():
    """데이터베이스 URL 생성"""
    return f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

def test_database_connection():
    """데이터베이스 연결 테스트"""
    try:
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            database=DATABASE_CONFIG['database'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password']
        )
        conn.close()
        logger.info("PostgreSQL 연결 테스트 성공")
        return True
    except Exception as e:
        logger.error(f"PostgreSQL 연결 테스트 실패: {e}")
        return False

def get_engine():
    """SQLAlchemy 엔진 생성"""
    return create_engine(get_database_url())
```
# 10. 메인 실행 파일 업데이트
### main.py 내용(code main.py)
```
"""
메인 실행 파일
"""
import sys
import os
import logging
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import BinanceDataCollector
from config.database import test_database_connection
from config.settings import TRADING_CONFIG

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    print("🚀 Crypto Pullback AI 시스템 시작!")
    print("=" * 50)
    
    # 1. 환경 확인
    logger.info("시스템 환경 확인 중...")
    
    # 2. 데이터 수집기 초기화
    collector = BinanceDataCollector()
    
    # 3. 바이낸스 연결 테스트
    if not collector.test_connection():
        logger.error("바이낸스 연결 실패")
        return
    
    # 4. 데이터베이스 연결 테스트 (선택사항)
    logger.info("데이터베이스 연결 테스트...")
    if test_database_connection():
        logger.info("데이터베이스 연결 성공")
    else:
        logger.warning("데이터베이스 연결 실패 - 나중에 설정 필요")
    
    # 5. 데이터 수집 시작
    symbols = TRADING_CONFIG['symbols']
    
    print(f"\n📊 데이터 수집 시작 ({len(symbols)}개 심볼)")
    
    collected_data = {}
    
    for symbol in symbols:
        logger.info(f"{symbol} 데이터 수집 시작...")
        
        # 7일치 데이터 수집
        df = collector.collect_historical_data(symbol, days=7)
        
        if not df.empty:
            collected_data[symbol] = df
            logger.info(f"✅ {symbol} 완료: {len(df)}개 캔들")
            
            # 간단한 통계 출력
            current_price = df['close'].iloc[-1]
            price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            
            print(f"   {symbol}: ${current_price:,.2f} ({price_change:+.2f}%)")
        else:
            logger.error(f"❌ {symbol} 데이터 수집 실패")
    
    # 6. 수집 결과 요약
    print(f"\n📈 수집 완료 요약:")
    print(f"   성공: {len(collected_data)}/{len(symbols)} 심볼")
    print(f"   총 데이터: {sum(len(df) for df in collected_data.values())}개 캔들")
    
    # 7. 다음 단계 안내
    print(f"\n🎯 다음 단계:")
    print("   1. PostgreSQL 설정 (선택사항)")
    print("   2. 기본 분석 및 시각화")
    print("   3. 눌림매매 전략 개발")
    
    logger.info("0-1단계 완료!")

if __name__ == "__main__":
    main()
```

# 11. 첫 번째 완전 실행
```
# 전체 시스템 실행
python main.py
```
