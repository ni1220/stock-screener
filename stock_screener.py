import yfinance as yf
import pandas as pd
import streamlit as st
import twstock
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from deep_translator import GoogleTranslator
from datetime import datetime
import pickle

# --- 設定網頁佈局 ---
st.set_page_config(page_title="台股戰略操盤室 Pro", page_icon="🧪", layout="wide")

# --- 初始化 Session State ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None 
if 'scan_results_potential' not in st.session_state:
    st.session_state.scan_results_potential = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "ranking"
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = [] 

# --- 1. 定義清單 ---
POTENTIAL_STOCKS = ["3035", "3017", "6274", "8069", "1519"]
MARKET_STOCKS = [
    "2330", "2317", "2454", "2308", "2303", "2881", "2412", "2382", "2882", "2886",
    "2891", "2884", "2357", "2002", "1216", "2892", "2885", "3711", "3008", "2345",
    "3231", "3045", "5880", "2912", "4904", "2880", "2883", "2887", "2603", "3034",
    "2379", "2395", "1101", "3037", "2327", "2408", "2354", "2360", "2207", "1605",
    "2609", "2615", "2610", "2618", "2606", "2605", "2637", "2633", "2634", "5608",
    "1301", "1303", "1326", "6505", "1402", "2105", "1102", "1907", "2014", "2027",
    "2301", "2313", "2324", "2344", "2353", "2356", "2368", "2376", "2377", "2383",
    "2385", "2392", "2393", "2409", "2449", "2451", "2474", "2492", "2498", "3006",
    "3014", "3036", "3042", "3189", "3293", "3406", "3443", "3481", "3532",
    "3563", "3596", "3653", "3661", "3702", "4915", "4919", "4958", "4961", "4968",
    "5269", "5274", "5347", "5483", "5536", "5871", "5876", "6176", "6213", "6269"
]

# --- 2. 輔助功能 ---
def translate_sector(sector_eng):
    mapping = {
        "Technology": "電子科技", "Financial Services": "金融服務", "Industrials": "工業製造",
        "Consumer Cyclical": "非必需消費品", "Basic Materials": "原物料", "Real Estate": "房地產",
        "Communication Services": "通訊服務", "Energy": "能源", "Healthcare": "醫療保健",
        "Semiconductors": "半導體", "Shipping & Ports": "航運", "Electronic Components": "電子零組件"
    }
    return mapping.get(sector_eng, sector_eng)

def translate_summary(text):
    if not text or len(text) < 5: return "暫無詳細描述。"
    try:
        text_to_translate = text[:2000]
        translator = GoogleTranslator(source='auto', target='zh-TW')
        return translator.translate(text_to_translate)
    except: return f"翻譯失敗，顯示原文：\n{text}"

def add_to_history(mode_key, df, note=""):
    if df is not None and not df.empty:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        display_name = f"[{time_str}] {note}"
        st.session_state.scan_history.insert(0, {
            "display_name": display_name,
            "data": df,
            "mode": mode_key,
            "timestamp": datetime.now().timestamp()
        })

# --- 3. 技術指標計算 ---
def calculate_indicators(df):
    df['9_High'] = df['High'].rolling(9).max()
    df['9_Low'] = df['Low'].rolling(9).min()
    df['RSV'] = (df['Close'] - df['9_Low']) / (df['9_High'] - df['9_Low']) * 100
    df['RSV'] = df['RSV'].fillna(50)
    k_list, d_list = [], []
    k, d = 50, 50
    for rsv in df['RSV']:
        k = (2/3) * k + (1/3) * rsv
        d = (2/3) * d + (1/3) * k
        k_list.append(k)
        d_list.append(d)
    df['K'] = k_list
    df['D'] = d_list

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEA']
    return df

# --- 4. 核心分析模組 ---
def analyze_price_action_logic(df):
    close = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma60 = df['Close'].rolling(60).mean().iloc[-1]
    recent_high = df['High'].tail(20).max()
    recent_low = df['Low'].tail(20).min()
    bias_20 = ((close - ma20) / ma20) * 100
    
    k = df['K'].iloc[-1]
    d = df['D'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    prev_macd_hist = df['MACD_Hist'].iloc[-2]

    trend = "盤整震盪"
    trend_color = "orange"
    if close > ma20 and ma20 > ma60:
        trend = "強勢多頭"
        trend_color = "green"
    elif close < ma20 and ma20 < ma60:
        trend = "空頭走勢"
        trend_color = "red"
    elif close > ma20 and ma20 < ma60:
        trend = "反彈格局"
        trend_color = "blue"

    volume = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    vol_status = "量能溫和"
    if volume > avg_vol * 1.5:
        vol_status = "🔥 爆量攻擊" if close > df['Open'].iloc[-1] else "⚠️ 爆量下殺"
    elif volume < avg_vol * 0.6:
        vol_status = "🧊 量縮整理"

    reasons = []
    if close > ma20: reasons.append("✅ 股價站穩月線 (20MA)。")
    if ma20 > ma60: reasons.append("✅ 均線多頭排列。")
    if k > 80: reasons.append("⚠️ KD 過熱 (>80)。")
    elif k < 20: reasons.append("✨ KD 低檔鈍化 (<20)。")
    if k > d and df['K'].iloc[-2] < df['D'].iloc[-2]: reasons.append("🚀 KD 黃金交叉。")
    if macd_hist > 0 and prev_macd_hist < 0: reasons.append("🚀 MACD 翻紅轉強。")
    
    viewpoint = ""
    if bias_20 > 10: viewpoint = "過熱 - 拉回再買"
    elif bias_20 < -10: viewpoint = "超跌 - 搶反彈"
    elif trend == "強勢多頭": viewpoint = "多頭 - 順勢操作"
    else: viewpoint = "觀望 - 等待訊號"

    if trend_color == "green" or trend_color == "blue":
        entry_price = ma20
        stop_loss = ma60 if ma60 < entry_price else entry_price * 0.95 
        take_profit = recent_high if recent_high > entry_price * 1.05 else entry_price * 1.1
    elif trend_color == "red":
        entry_price = 0; stop_loss = 0; take_profit = 0
    else:
        entry_price = recent_low * 1.02
        stop_loss = recent_low * 0.97   
        take_profit = recent_high * 0.98

    rr_ratio = 0
    if entry_price > 0 and stop_loss > 0 and (entry_price - stop_loss) > 0:
        rr_ratio = (take_profit - entry_price) / (entry_price - stop_loss)

    return {
        'trend': trend, 'trend_color': trend_color,
        'vol_status': vol_status, 'ma20': ma20, 'ma60': ma60,
        'k': k, 'd': d, 'macd_hist': macd_hist,
        'entry_price': entry_price, 'stop_loss': stop_loss,
        'take_profit': take_profit, 'rr_ratio': rr_ratio,
        'reasons': reasons, 'viewpoint': viewpoint
    }

# --- 新增：回測引擎 ---
def backtest_strategy(df, initial_capital=100000):
    """
    簡單回測邏輯：KD 黃金交叉(K>D)且K<40買入，死亡交叉(K<D)賣出
    """
    cash = initial_capital
    position = 0 # 持股數
    equity_curve = []
    
    # 紀錄交易點
    buy_signals = []
    sell_signals = []
    
    # 從資料開始處遍歷 (忽略前 20 天以確保有 MA/KD)
    for i in range(20, len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        
        # 指標
        k = df['K'].iloc[i]
        d = df['D'].iloc[i]
        prev_k = df['K'].iloc[i-1]
        prev_d = df['D'].iloc[i-1]
        
        # 買入條件：黃金交叉 且 K 在相對低檔 (40以下)
        if position == 0:
            if k > d and prev_k < prev_d and k < 40:
                position = int(cash / price)
                cash -= position * price
                buy_signals.append((date, price))
        
        # 賣出條件：死亡交叉 
        elif position > 0:
            if k < d and prev_k > prev_d:
                cash += position * price
                position = 0
                sell_signals.append((date, price))
        
        # 計算當日總資產
        current_equity = cash + (position * price)
        equity_curve.append(current_equity)

    # 計算 Buy & Hold 績效
    start_price = df['Close'].iloc[20]
    end_price = df['Close'].iloc[-1]
    bh_return = (end_price - start_price) / start_price * 100
    
    # 計算 策略 績效
    final_equity = equity_curve[-1]
    strategy_return = (final_equity - initial_capital) / initial_capital * 100
    
    return {
        'equity_curve': equity_curve,
        'dates': df.index[20:],
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'strategy_return': strategy_return,
        'bh_return': bh_return,
        'final_equity': final_equity
    }

def plot_backtest(df, bt_result):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 股價
    fig.add_trace(go.Scatter(x=bt_result['dates'], y=df['Close'].iloc[20:], name="股價", line=dict(color='gray', width=1, dash='dot')), secondary_y=True)
    
    # 策略資產曲線
    fig.add_trace(go.Scatter(x=bt_result['dates'], y=bt_result['equity_curve'], name="KD策略總資產", line=dict(color='blue', width=2)), secondary_y=False)
    
    # 買賣點
    for date, price in bt_result['buy_signals']:
        fig.add_annotation(x=date, y=price, text="B", showarrow=True, arrowhead=1, ax=0, ay=10, bgcolor="red", font=dict(color="white"), yref="y2")
    for date, price in bt_result['sell_signals']:
        fig.add_annotation(x=date, y=price, text="S", showarrow=True, arrowhead=1, ax=0, ay=-10, bgcolor="green", font=dict(color="white"), yref="y2")

    fig.update_layout(title="過去一年策略回測 vs 股價走勢", xaxis_title="日期", height=500)
    fig.update_yaxes(title_text="總資產 (TWD)", secondary_y=False)
    fig.update_yaxes(title_text="股價", secondary_y=True)
    return fig

# --- 修復：補回 plot_full_analysis 函式 ---
def plot_full_analysis(df, name, analysis):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{name} 價量分析", "KD", "MACD"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), line=dict(color='orange', width=1), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(60).mean(), line=dict(color='green', width=1), name='季線'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='blue', width=1), name='K'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='orange', width=1), name='D'), row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="gray", row=2, col=1)
    
    colors = ['red' if v < 0 else 'green' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DIF'], line=dict(color='black', width=1), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DEA'], line=dict(color='orange', width=1), name='DEA'), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=False)
    return fig

@st.cache_data(ttl=3600)
def get_stock_detail(code):
    try:
        clean_code = code.upper().replace('.TW', '').replace('.TWO', '').strip()
        yf_ticker = f"{clean_code}.TW"
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period="1y")
        if hist.empty:
             yf_ticker = f"{clean_code}.TWO"
             stock = yf.Ticker(yf_ticker)
             hist = stock.history(period="1y")
        if hist.empty: return None 

        name = clean_code
        if clean_code in twstock.codes: name = twstock.codes[clean_code].name
        
        hist = calculate_indicators(hist)
        pa = analyze_price_action_logic(hist)
        # 進行回測
        bt = backtest_strategy(hist)
        
        info = stock.info
        raw_summary = info.get('longBusinessSummary', '')
        zh_summary = translate_summary(raw_summary)
        return {'info': info, 'history': hist, 'news': stock.news, 'name': name, 'analysis': pa, 'zh_summary': zh_summary, 'backtest': bt}
    except: return None

def scan_tickers_optimized(ticker_list, max_pe, min_bias, investment, min_price, max_price, mode="ranking"):
    results = []
    batch_tickers = [f"{code}.TW" for code in ticker_list]
    try:
        data = yf.download(batch_tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        if data.empty: return pd.DataFrame()

        for code in ticker_list:
            yf_code = f"{code}.TW"
            try:
                if yf_code not in data.columns.levels[0]: continue 
                df = data[yf_code].dropna()
                if df.empty or len(df) < 20: continue

                cp = df['Close'].iloc[-1]
                if not (min_price <= cp <= max_price): continue 

                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                bias = ((cp - ma20) / ma20) * 100
                pe = 999
                should_add = False
                
                if mode == "price":
                    should_add = True
                else:
                    if bias >= min_bias or (len(ticker_list) <= 10):
                        try:
                            t = yf.Ticker(yf_code)
                            pe_val = t.info.get('trailingPE', 999)
                            pe = pe_val if pe_val else 999
                        except: pe = 999
                        if (pe <= max_pe or pe == 999): should_add = True

                if should_add:
                    entry = ma20
                    target = cp * 1.05
                    profit = int(investment * ((target - entry)/entry)) if entry > 0 else 0
                    name = twstock.codes[code].name if code in twstock.codes else code
                    results.append({
                        '代號': code, '名稱': name, '現價': round(cp, 2),
                        '本益比': round(pe, 2) if pe != 999 else "N/A", 
                        '乖離率(%)': round(bias, 2), '預估獲利': f"${profit:,}"
                    })
            except: continue
    except: return pd.DataFrame()
    return pd.DataFrame(results)

# --- 5. 側邊欄控制台 ---

with st.sidebar:
    st.header("🎮 戰情室控制台")
    
    col_input, col_go = st.columns([2, 1])
    with col_input:
        direct_input = st.text_input("代號快搜", placeholder="如 2330", label_visibility="collapsed")
    with col_go:
        if st.button("GO", type="primary"):
            if direct_input:
                st.session_state.selected_stock = direct_input.replace(".TW", "").replace(".TWO", "").strip().upper()
                st.rerun()
    st.markdown("---")
    
    with st.expander("♾️ 歷史總檔管理", expanded=True):
        st.info("💡 流程：開盤讀舊檔 ➡️ 盤中掃描 ➡️ 收盤存新檔")
        
        uploaded_file = st.file_uploader("📂 Step 1: 讀取「歷史總檔.pkl」", type=["pkl"])
        if uploaded_file is not None:
            try:
                restored_history = pickle.load(uploaded_file)
                if isinstance(restored_history, list):
                    existing_names = set(item['display_name'] for item in st.session_state.scan_history)
                    new_count = 0
                    for item in restored_history:
                        if item['display_name'] not in existing_names:
                            if 'timestamp' not in item: item['timestamp'] = datetime.now().timestamp()
                            st.session_state.scan_history.append(item)
                            existing_names.add(item['display_name'])
                            new_count += 1
                    
                    st.session_state.scan_history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                    if new_count > 0:
                        st.success(f"已載入 {new_count} 筆歷史紀錄！")
                else: st.error("檔案格式不符")
            except: pass

        if st.session_state.scan_history:
            history_data = pickle.dumps(st.session_state.scan_history)
            file_name = "歷史總檔_Master.pkl"
            st.download_button(
                "💾 Step 2: 下載更新後的總檔", 
                data=history_data, 
                file_name=file_name, 
                mime="application/octet-stream", 
                type="primary"
            )
        else:
            st.warning("尚無紀錄可下載")

    st.markdown("---")
    
    page_mode = st.radio("📱 選擇功能模式", ["🏆 市場熱門排行", "🪙 價格區間快搜"])
    
    if page_mode == "🏆 市場熱門排行":
        st.session_state.current_mode = "ranking"
        st.info("💡 篩選優質權值股與潛力黑馬")
        with st.expander("⚙️ 進階參數設定 (PE / 乖離)", expanded=True):
            max_pe = st.number_input("本益比上限 (PE)", value=35)
            min_bias = st.number_input("乖離率下限 (%)", value=0.0)
        investment_amount = st.number_input("預估投入金額", value=100000, step=10000)
        
        if st.button("🚀 啟動排行掃描", type="primary"):
            st.session_state.selected_stock = None
            st.session_state.scan_results = None 
            with st.spinner("正在全速下載市場數據..."):
                df_main = scan_tickers_optimized(MARKET_STOCKS, max_pe, min_bias, investment_amount, 0, 9999, mode="ranking")
                if not df_main.empty:
                    df_top20 = df_main.sort_values(by='乖離率(%)', ascending=False).head(20).reset_index(drop=True)
                    df_top20.insert(0, '排名', range(1, 1 + len(df_top20)))
                    st.session_state.scan_results = df_top20
                    add_to_history("ranking", df_top20, f"Top 20 (PE<{max_pe})")
                else: st.session_state.scan_results = pd.DataFrame()

                st.session_state.scan_results_potential = scan_tickers_optimized(POTENTIAL_STOCKS, 9999, -9999, investment_amount, 0, 9999, mode="ranking")
            st.success("掃描完成！")

    else:
        st.session_state.current_mode = "price"
        st.info("💡 尋找特定價位的機會 (如銅板股)")
        with st.expander("💰 價格區間設定", expanded=True):
            c1, c2 = st.columns(2)
            min_p = c1.number_input("最低價", value=0)
            max_p = c2.number_input("最高價", value=50)
        investment_amount = st.number_input("預估投入金額 (本金)", value=50000, step=5000)
        
        if st.button("🔎 搜尋價格區間", type="primary"):
            st.session_state.selected_stock = None
            st.session_state.scan_results = None
            with st.spinner(f"正在搜尋 {min_p}~{max_p} 元的股票..."):
                df_price = scan_tickers_optimized(MARKET_STOCKS, 9999, -9999, investment_amount, min_p, max_p, mode="price")
                if not df_price.empty:
                    df_price = df_price.sort_values(by='現價', ascending=True).reset_index(drop=True)
                    st.session_state.scan_results = df_price
                    add_to_history("price", df_price, f"股價 ${min_p}-${max_p}")
                else:
                    st.session_state.scan_results = pd.DataFrame()
            st.success("搜尋完成！")

# === 主畫面路由 ===
if st.session_state.selected_stock:
    code = st.session_state.selected_stock
    
    if st.button("⬅️ 返回列表"):
        st.session_state.selected_stock = None
        st.rerun()

    with st.spinner(f"正在進行 {code} 全方位戰略分析..."):
        data = get_stock_detail(code)
    
    if data:
        pa = data['analysis']
        info = data['info']
        bt = data['backtest']
        
        st.markdown(f"# {data['name']} ({code}) 戰略分析報告")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 戰略總覽", "📈 技術指標詳解", "💰 獲利與產業", "🧪 策略回測 (驗證)"])

        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{data['history']['Close'].iloc[-1]:.2f}")
            c2.metric("趨勢", pa['trend'], delta_color="normal" if pa['trend_color'] == "green" else "inverse")
            c3.metric("AI 觀點", pa['viewpoint'])
            k_val, d_val = f"{pa['k']:.1f}", f"{pa['d']:.1f}"
            c4.metric("KD 值", k_val, f"D: {d_val}", delta_color="off")
            
            st.markdown("### 🧐 深度觀點")
            col_reason, col_setup = st.columns([1, 1])
            with col_reason:
                st.info("💡 **操作理由**")
                for reason in pa['reasons']: st.write(reason)
            with col_setup:
                st.success("🎯 **交易戰術**")
                if pa['entry_price'] > 0:
                    st.write(f"🔵 **建議進場:** `{pa['entry_price']:.2f}`")
                    st.write(f"🔴 **獲利目標:** `{pa['take_profit']:.2f}`")
                    st.write(f"⚪️ **防守止損:** `{pa['stop_loss']:.2f}`")
                else: st.warning("⛔️ 目前趨勢不佳，建議觀望。")
            st.plotly_chart(plot_full_analysis(data['history'], data['name'], pa), use_container_width=True)

        with tab2:
            st.subheader("📈 技術指標訊號解讀")
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("#### KD 隨機指標")
                st.write(f"**目前 K 值: {pa['k']:.1f} / D 值: {pa['d']:.1f}**")
                if pa['k'] > 80: st.warning("🔥 **超買區**：過熱警戒。")
                elif pa['k'] < 20: st.success("🧊 **超賣區**：低檔鈍化，關注反彈。")
                elif pa['k'] > pa['d']: st.info("📈 **黃金交叉**：短線偏多。")
                else: st.error("📉 **死亡交叉**：短線偏空。")
            with t2:
                st.markdown("#### MACD 趨勢指標")
                st.write(f"**MACD 柱狀體: {pa['macd_hist']:.2f}**")
                if pa['macd_hist'] > 0: st.success("🚀 **多頭趨勢**：柱狀翻紅。")
                else: st.error("🐻 **空頭抵抗**：柱狀翻綠。")

        with tab3:
            st.subheader("💰 獲利預估與產業背景")
            if pa['entry_price'] > 0:
                roi = (pa['take_profit'] - pa['entry_price']) / pa['entry_price']
                est_profit = int(investment_amount * roi)
                st.metric("預估獲利", f"${est_profit:,}", f"報酬率 {roi*100:.1f}%")
            else: st.write("無進場建議，無法計算。")
            st.markdown("---")
            sector_zh = translate_sector(info.get('sector', 'Unknown'))
            st.info(f"**產業**: {sector_zh} | **細分**: {info.get('industry', 'Unknown')}")
            st.write(f"**公司簡介**: {data['zh_summary']}")
            
        with tab4:
            st.subheader("🧪 過去一年策略回測 (驗證)")
            st.caption("模擬策略：KD 低檔黃金交叉買進，死亡交叉賣出。初始資金：10萬")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("KD 策略報酬率", f"{bt['strategy_return']:.1f}%", delta_color="normal")
            m2.metric("買進持有 (Buy&Hold) 報酬率", f"{bt['bh_return']:.1f}%", delta_color="normal")
            m3.metric("期末總資產", f"${int(bt['final_equity']):,}")
            
            if bt['strategy_return'] > bt['bh_return']:
                st.success("🎉 **恭喜！此策略在過去一年勝過單純存股！**")
            else:
                st.warning("⚠️ 注意：此策略績效不如單純持有，建議搭配其他指標判斷。")
                
            st.plotly_chart(plot_backtest(data['history'], bt), use_container_width=True)

    else:
        st.error(f"❌ 找不到代號為 **{code}** 的股票資料。")
        st.markdown("""
        **可能原因：**
        1. **代號輸入錯誤**：請確認是否為正確的台股代號（如 2330）。
        2. **資料源異常**：Yahoo Finance 暫時無法連線。
        3. **股票已下市**：該股票已停止交易。
        
        請按下方的「返回列表」重新輸入。
        """)

else:
    # --- 列表頁 ---
    if st.session_state.current_mode == "ranking":
        st.title("🏆 台股熱門排行模式")
        tab_now, tab_hist = st.tabs(["🚀 最新掃描結果", "📜 排行榜歷史回顧"])
        with tab_now:
            if st.session_state.scan_results is not None:
                st.subheader("Top 20 熱門排行")
                st.caption("💡 依據乖離率排序")
                if not st.session_state.scan_results.empty:
                    event1 = st.dataframe(st.session_state.scan_results, on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True)
                    if len(event1.selection.rows) > 0:
                        st.session_state.selected_stock = st.session_state.scan_results.iloc[event1.selection.rows[0]]['代號']
                        st.rerun()
                else: st.warning("無符合條件股票。")
                st.markdown("---")
                st.subheader("🚀 潛力黑馬股")
                if st.session_state.scan_results_potential is not None:
                    event2 = st.dataframe(st.session_state.scan_results_potential, on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True)
                    if len(event2.selection.rows) > 0:
                        st.session_state.selected_stock = st.session_state.scan_results_potential.iloc[event2.selection.rows[0]]['代號']
                        st.rerun()
            else:
                st.info("👈 請在左側設定參數並按下 **「啟動排行掃描」**")
        with tab_hist:
            ranking_hist = [h for h in st.session_state.scan_history if h['mode'] == 'ranking']
            if not ranking_hist: st.write("尚無排行模式的歷史紀錄。")
            else:
                hist_options = [h['display_name'] for h in ranking_hist]
                selected_hist_name = st.selectbox("選擇存檔點", hist_options, key="rank_hist_select")
                selected_data = next((h['data'] for h in ranking_hist if h['display_name'] == selected_hist_name), None)
                if selected_data is not None:
                    st.subheader(f"📂 {selected_hist_name}")
                    event_h1 = st.dataframe(selected_data, on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True, key="rank_hist_table")
                    if len(event_h1.selection.rows) > 0:
                        st.session_state.selected_stock = selected_data.iloc[event_h1.selection.rows[0]]['代號']
                        st.rerun()

    elif st.session_state.current_mode == "price":
        st.title("🪙 價格區間獵人模式")
        tab_now, tab_hist = st.tabs(["🔎 最新搜尋結果", "📜 價格搜尋歷史"])
        with tab_now:
            if st.session_state.scan_results is not None:
                st.subheader("搜尋結果")
                st.caption("💡 依據股價排序")
                if not st.session_state.scan_results.empty:
                    event3 = st.dataframe(st.session_state.scan_results, on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True)
                    if len(event3.selection.rows) > 0:
                        st.session_state.selected_stock = st.session_state.scan_results.iloc[event3.selection.rows[0]]['代號']
                        st.rerun()
                else: st.warning(f"在該價格區間內找不到熱門股。")
            else:
                st.info("👈 請在左側設定價格範圍並按下 **「搜尋價格區間」**")
        with tab_hist:
            price_hist = [h for h in st.session_state.scan_history if h['mode'] == 'price']
            if not price_hist: st.write("尚無價格模式的歷史紀錄。")
            else:
                hist_options = [h['display_name'] for h in price_hist]
                selected_hist_name = st.selectbox("選擇存檔點", hist_options, key="price_hist_select")
                selected_data = next((h['data'] for h in price_hist if h['display_name'] == selected_hist_name), None)
                if selected_data is not None:
                    st.subheader(f"📂 {selected_hist_name}")
                    event_h2 = st.dataframe(selected_data, on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True, key="price_hist_table")
                    if len(event_h2.selection.rows) > 0:
                        st.session_state.selected_stock = selected_data.iloc[event_h2.selection.rows[0]]['代號']
                        st.rerun()
