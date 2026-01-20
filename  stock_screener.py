import yfinance as yf
import pandas as pd
import streamlit as st

# 1. 設定目標股票池 (這裡先用幾支美股科技股當範例，也可以換成台股代號如 '2330.TW')
tickers = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'INTC']


# 2. 定義抓取與運算函數
def get_stock_data(ticker_list):
    data = []
    for ticker in ticker_list:
        try:
            # 抓取個股資訊
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="6mo")  # 抓過去半年股價

            # --- 這裡運用簡單的會計與數學運算 ---

            # A. 取得現價
            current_price = hist['Close'].iloc[-1]

            # B. 計算技術指標 (移動平均線)
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]

            # C. 取得基本面資料 (本益比)
            # 如果沒有 PE 資料 (例如虧損中)，設為無限大以便過濾
            pe_ratio = info.get('trailingPE', 999)

            data.append({
                '代號': ticker,
                '股價': round(current_price, 2),
                '50日均線': round(ma_50, 2),
                '本益比(PE)': pe_ratio,
                '趨勢': '強勢' if current_price > ma_50 else '弱勢'
            })
        except Exception as e:
            print(f"無法取得 {ticker} 資料: {e}")

    return pd.DataFrame(data)


# 3. 建立 Streamlit 介面
st.title('📈 我的優質選股小幫手')

st.write("正在分析市場數據，請稍候...")
df = get_stock_data(tickers)

# 4. 設定「優質」的篩選條件 (這裡是你發揮創意的地方)
# 例如：我要找「強勢」且「本益比合理(<35)」的股票
st.subheader('篩選條件設定')
pe_filter = st.slider('最高本益比 (P/E)', 0, 100, 35)

# 執行篩選
condition1 = df['趨勢'] == '強勢'
condition2 = df['本益比(PE)'] <= pe_filter
selection = df[condition1 & condition2]

# 5. 顯示結果
st.success(f'篩選出 {len(selection)} 檔優質股票！')
st.dataframe(selection)

# 加分項：顯示原始資料以供對照
with st.expander("查看原始清單"):
    st.dataframe(df)