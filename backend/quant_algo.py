import os
import time
import pyotp
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
from smartapi_client import SmartAPIClient  # Add this import if not present
from SmartApi import SmartConnect
from functools import wraps
import json
import sys
import os
from claude_nifty_options_ai import NiftyOptionsAI  # Add this import
import yfinance as yf
import ta
import logging

# Import algo_ai functionality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import algo_ai

# Load environment variables
load_dotenv()

# Rate limiting decorator
def rate_limited(max_per_second):
    min_interval = 1.0 / float(max_per_second)
    def decorator(func):
        last_time_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def initialize_angel_broker():
    try:
        api_key = os.getenv('ANGEL_API_KEY')
        client_id = os.getenv('ANGEL_CLIENT_ID')
        password = os.getenv('ANGEL_PASSWORD')
        totp_key = os.getenv('ANGEL_TOTP_KEY')
        
        # Create SmartAPIClient instance instead of SmartConnect
        smart_api = SmartAPIClient(api_key, client_id, password, totp_key)
        
        # Authenticate using the client
        if smart_api.authenticate():
            st.success(f"Login successful for client: {client_id}")
            return smart_api
        
        st.error("Login failed. Please check your credentials.")
        return None
            
    except Exception as e:
        st.error(f"Error connecting to AngelOne: {e}")
        return None

@rate_limited(1)
def fetch_market_data(smart_api, params):
    """Fetch market data with rate limiting"""
    try:
        if isinstance(smart_api, SmartAPIClient):
            return smart_api.get_candle_data(
                exchange=params.get('exchange', 'NSE'),
                token=params.get('symboltoken'),
                interval=params.get('interval'),
                from_date=params.get('fromdate'),
                to_date=params.get('todate')
            )
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching live data: {str(e)}")
        return pd.DataFrame()

@rate_limited(1)
def fetch_option_chain(smart_api, params):
    """Fetch option chain data with rate limiting"""
    try:
        return smart_api.searchScrip(**params)
    except Exception as e:
        return {"status": False, "message": str(e)}

def display_smartapi_details():
    """Display SmartAPI details and functionalities"""
    st.header("SmartAPI Details")
    
    if 'smartapi_client' in st.session_state:
        client = st.session_state.smartapi_client
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Portfolio", "Order Book", "WebSocket"])
        
        with sub_tab1:
            st.subheader("Portfolio & Positions")
            positions = client.get_positions()
            if not positions.empty:
                st.dataframe(positions)
            else:
                st.warning("No positions data available")
        
        with sub_tab2:
            st.subheader("Order Book")
            order_book = client.get_order_book()
            if not order_book.empty:
                st.dataframe(order_book)
            else:
                st.warning("No order book data available")
        
        with sub_tab3:
            st.subheader("WebSocket")
            if client.websocket and client.websocket.wsapp and client.websocket.wsapp.sock:
                st.success("WebSocket Connected")
                st.write(f"Subscribed Tokens: {client.subscribed_tokens}")
            else:
                st.warning("WebSocket Not Connected")
    else:
        st.error("SmartAPI Not Connected")

def process_signals(df, smartapi_client):
    """Process trading signals and execute paper trades"""
    try:
        # Get latest signal
        latest_signal = df[df['Signal'] != 'HOLD'].iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        if latest_signal['Signal'] in ['BUY', 'SELL', 'EXIT']:
            # Default quantity - can be made dynamic based on your strategy
            quantity = 100
            
            # Execute paper trade
            result = smartapi_client.execute_paper_trade(
                signal=latest_signal['Signal'],
                price=current_price,
                quantity=quantity,
                symbol=df['Symbol'].iloc[0] if 'Symbol' in df.columns else 'NIFTY'
            )
            
            # Display trade result
            st.success(result)
            
            # Display updated stats
            stats = smartapi_client.get_paper_trading_stats()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total P&L", f"₹{stats['Total P&L']:.2f}")
            with col2:
                st.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
            with col3:
                st.metric("Paper Balance", f"₹{stats['Current Balance']:.2f}")
            
            # Show trade history
            trades_df = smartapi_client.get_paper_trades()
            if not trades_df.empty:
                st.dataframe(trades_df)
                
    except Exception as e:
        st.error(f"Error processing signals: {str(e)}")

class QuantAlgo:
    def __init__(self):
        self.setup_logging()
        self.initialize_session_state()
        self.setup_strategies()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'quant_positions' not in st.session_state:
            st.session_state.quant_positions = []
        if 'quant_trades' not in st.session_state:
            st.session_state.quant_trades = []
        if 'quant_pnl' not in st.session_state:
            st.session_state.quant_pnl = 0

    def setup_strategies(self):
        """Initialize quantitative trading strategies"""
        self.strategies = {
            'Statistical Arbitrage': self.statistical_arbitrage,
            'Pairs Trading': self.pairs_trading,
            'Factor Investing': self.factor_investing,
            'Risk Parity': self.risk_parity,
            'Machine Learning': self.machine_learning_strategy
        }

    def display_dashboard(self):
        """Main dashboard for Quantitative Trading"""
        st.title("📈 Quantitative Trading")

        # Create main tabs
        tabs = st.tabs([
            "Portfolio Analytics",
            "Strategy Lab",
            "Risk Management",
            "Execution",
            "Research"
        ])

        with tabs[0]:
            self.display_portfolio_analytics()
        with tabs[1]:
            self.display_strategy_lab()
        with tabs[2]:
            self.display_risk_management()
        with tabs[3]:
            self.display_execution()
        with tabs[4]:
            self.display_research()

    def display_portfolio_analytics(self):
        """Display portfolio analytics dashboard"""
        st.subheader("Portfolio Analytics")

        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Value", "₹10,00,000", "↑ 2.5%")
        with col2:
            st.metric("Daily P&L", "₹15,000", "↑ 1.5%")
        with col3:
            st.metric("Sharpe Ratio", "1.8", "↑ 0.2")
        with col4:
            st.metric("Max Drawdown", "-5%", "↓ 1%")

        # Portfolio composition
        self.plot_portfolio_composition()
        
        # Risk metrics
        self.display_risk_metrics()

    def display_strategy_lab(self):
        """Display strategy laboratory"""
        st.subheader("Strategy Laboratory")

        # Strategy selection
        selected_strategy = st.selectbox(
            "Select Strategy",
            list(self.strategies.keys())
        )

        # Strategy parameters
        col1, col2 = st.columns([2, 1])
        with col1:
            self.display_strategy_parameters(selected_strategy)
        with col2:
            self.display_strategy_metrics(selected_strategy)

    def statistical_arbitrage(self, data1, data2, params):
        """Statistical arbitrage strategy"""
        try:
            # Calculate z-score
            spread = data1['Close'] - data2['Close']
            z_score = (spread - spread.mean()) / spread.std()
            
            # Generate signals
            signals = pd.DataFrame(index=data1.index)
            signals['Signal'] = 0
            
            # Entry signals
            signals.loc[z_score > params['upper_threshold'], 'Signal'] = -1  # Short
            signals.loc[z_score < params['lower_threshold'], 'Signal'] = 1   # Long
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in statistical arbitrage: {str(e)}")
            return None

    def pairs_trading(self, data1, data2, params):
        """Pairs trading strategy"""
        try:
            # Calculate correlation and cointegration
            correlation = data1['Close'].corr(data2['Close'])
            
            if correlation > params['min_correlation']:
                # Calculate spread
                spread = data1['Close'] / data2['Close']
                
                # Generate signals based on mean reversion
                signals = pd.DataFrame(index=data1.index)
                signals['Signal'] = 0
                
                # Entry signals
                mean = spread.mean()
                std = spread.std()
                signals.loc[spread > mean + std * params['std_multiplier'], 'Signal'] = -1
                signals.loc[spread < mean - std * params['std_multiplier'], 'Signal'] = 1
                
                return signals
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in pairs trading: {str(e)}")
            return None

    def factor_investing(self, data, params):
        """Factor investing strategy"""
        try:
            # Calculate factors
            data['Momentum'] = data['Close'].pct_change(params['momentum_period'])
            data['Value'] = data['Close'] / data['Close'].rolling(window=params['value_period']).mean()
            data['Quality'] = data['Close'].rolling(window=params['quality_period']).std()
            
            # Generate signals based on factor combination
            signals = pd.DataFrame(index=data.index)
            signals['Signal'] = 0
            
            # Combined factor score
            factor_score = (
                data['Momentum'] * params['momentum_weight'] +
                data['Value'] * params['value_weight'] +
                data['Quality'] * params['quality_weight']
            )
            
            # Generate signals
            signals.loc[factor_score > params['upper_threshold'], 'Signal'] = 1
            signals.loc[factor_score < params['lower_threshold'], 'Signal'] = -1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in factor investing: {str(e)}")
            return None

    def risk_parity(self, data, params):
        """Risk parity portfolio strategy"""
        try:
            # Calculate asset volatilities
            returns = data.pct_change()
            vols = returns.rolling(window=params['vol_window']).std()
            
            # Calculate risk contribution weights
            weights = 1 / vols
            weights = weights / weights.sum()
            
            # Generate rebalancing signals
            signals = pd.DataFrame(index=data.index)
            signals['Weights'] = weights
            signals['Rebalance'] = False
            
            # Mark rebalancing dates
            signals.loc[signals.index.dayofweek == params['rebalance_day'], 'Rebalance'] = True
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in risk parity: {str(e)}")
            return None

    def machine_learning_strategy(self, data, params):
        """Machine learning based trading strategy"""
        try:
            # Feature engineering
            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            data['MACD'] = ta.trend.MACD(data['Close']).macd()
            data['BB_upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
            data['BB_lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
            
            # Prepare features
            features = ['RSI', 'MACD', 'BB_upper', 'BB_lower']
            X = data[features].dropna()
            
            # Generate predictions using the model
            predictions = self.ml_model.predict(X)
            
            # Convert predictions to signals
            signals = pd.DataFrame(index=X.index)
            signals['Signal'] = predictions
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in machine learning strategy: {str(e)}")
            return None

    def display_risk_management(self):
        """Display risk management dashboard"""
        st.subheader("Risk Management")
        
        # Risk parameters
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Position Size (%)", 1, 100, 5)
            st.number_input("Stop Loss (%)", 1, 100, 2)
        with col2:
            st.number_input("Max Drawdown (%)", 1, 100, 10)
            st.number_input("Risk/Reward Ratio", 1.0, 5.0, 2.0)

    def display_execution(self):
        """Display execution dashboard"""
        st.subheader("Strategy Execution")
        
        # Execution controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Trading"):
                self.start_trading()
        with col2:
            if st.button("Stop Trading"):
                self.stop_trading()
        with col3:
            if st.button("Emergency Exit"):
                self.emergency_exit()

    def display_research(self):
        """Display research dashboard"""
        st.subheader("Quantitative Research")
        
        # Research tools
        tab1, tab2, tab3 = st.tabs(["Backtesting", "Factor Analysis", "Market Analysis"])
        
        with tab1:
            self.display_backtesting()
        with tab2:
            self.display_factor_analysis()
        with tab3:
            self.display_market_analysis()

    def plot_portfolio_composition(self):
        """Plot portfolio composition using pie chart"""
        try:
            # Get portfolio data
            portfolio = self.get_portfolio_data()
            
            if portfolio.empty:
                st.info("No portfolio data available")
                return
            
            # Calculate portfolio composition
            composition = portfolio.groupby('sector')['value'].sum()
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=composition.index,
                values=composition.values,
                hole=0.4,
                marker=dict(
                    colors=['#00FF9D', '#00D1FF', '#FF61DC', '#FFC107', '#FF4B4B'],
                    line=dict(color='#1E1E1E', width=2)
                ),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=12, color='white'),
                showlegend=False
            )])
            
            fig.update_layout(
                title="Portfolio Composition by Sector",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display composition table
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sector Allocation")
                composition_df = pd.DataFrame({
                    'Sector': composition.index,
                    'Value (₹)': composition.values,
                    'Allocation (%)': (composition.values / composition.values.sum() * 100).round(2)
                })
                st.dataframe(composition_df)
            
            with col2:
                st.subheader("Risk Metrics")
                self.display_risk_metrics()
            
        except Exception as e:
            st.error(f"Error plotting portfolio composition: {str(e)}")
            if st.checkbox("Show Error Details"):
                st.exception(e)

    def get_portfolio_data(self):
        """Get portfolio data with sector information"""
        try:
            # Get positions from broker
            if self.smart_api:
                holdings = self.smart_api.get_holdings()
                if holdings is not None:
                    # Convert to DataFrame
                    df = pd.DataFrame(holdings)
                    
                    # Add sector information (you would need to implement this)
                    df['sector'] = self.get_sector_info(df['symbol'].tolist())
                    
                    # Calculate position values
                    df['value'] = df['quantity'] * df['ltp']
                    
                    return df
                    
            # Return empty DataFrame if no data
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error getting portfolio data: {str(e)}")
            return pd.DataFrame()

    def get_sector_info(self, symbols):
        """Get sector information for symbols"""
        try:
            # This is a placeholder - implement actual sector lookup
            # You could use yfinance, your broker's API, or a custom database
            sectors = {
                'NIFTY': 'Index',
                'BANKNIFTY': 'Banking',
                'RELIANCE': 'Energy',
                'TCS': 'Technology',
                'HDFCBANK': 'Banking'
            }
            return [sectors.get(symbol, 'Other') for symbol in symbols]
            
        except Exception as e:
            st.error(f"Error getting sector info: {str(e)}")
            return ['Other'] * len(symbols)

    def display_risk_metrics(self):
        """Display portfolio risk metrics"""
        try:
            portfolio = self.get_portfolio_data()
            if not portfolio.empty:
                metrics = {
                    'Beta': self.calculate_portfolio_beta(),
                    'Sharpe Ratio': self.calculate_sharpe_ratio(),
                    'Value at Risk (1d, 95%)': self.calculate_var(),
                    'Max Drawdown': self.calculate_max_drawdown()
                }
                
                for name, value in metrics.items():
                    st.metric(
                        name,
                        f"{value:.2f}" if value is not None else "N/A"
                    )
                    
        except Exception as e:
            st.error(f"Error displaying risk metrics: {str(e)}")

    def calculate_portfolio_beta(self):
        """Calculate portfolio beta"""
        try:
            # Implement beta calculation
            return 1.0  # Placeholder
        except Exception as e:
            st.error(f"Error calculating beta: {str(e)}")
            return None

    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        try:
            # Implement Sharpe ratio calculation
            return 1.5  # Placeholder
        except Exception as e:
            st.error(f"Error calculating Sharpe ratio: {str(e)}")
            return None

    def calculate_var(self):
        """Calculate Value at Risk"""
        try:
            # Implement VaR calculation
            return 2.5  # Placeholder
        except Exception as e:
            st.error(f"Error calculating VaR: {str(e)}")
            return None

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        try:
            # Implement max drawdown calculation
            return 15.0  # Placeholder
        except Exception as e:
            st.error(f"Error calculating max drawdown: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="Quant Algo Trading", page_icon="📈", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; background-color: #f0f2f6; }
        .stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Quant Algo")
     

    # Initialize API connection
    smart_api = initialize_angel_broker()
    if not smart_api:
        st.stop()

    # Create tabs with correct order
    tabs = st.tabs([
        "🏠 Overview",      # index 0 
        "💼 Portfolio",     # index 1
        "📈 Market Data",   # index 2
        "🔄 Orders",        # index 3
        "📊 Analysis",      # index 4
        "🔍 Symbol Search", # index 5
        "📋 Option Chain",  # index 6
        "⚙️ Settings",      # index 7
        "🤖 Algo AI",       # index 8
        "📊 NIFTY Options AI" # index 9
    ])

    # Overview Tab
    with tabs[0]:
        st.header("Account Overview")
        try:
            if isinstance(smart_api, SmartAPIClient):
                # Get RMS limits using SmartAPIClient
                response = smart_api.smart_api.rmsLimit()  # Access the underlying smart_api object
                if response and response.get('status'):
                    data = response.get('data', {})
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        available_balance = float(data.get('net', 0))
                        st.metric(
                            "Available Balance", 
                            f"₹{available_balance:,.2f}"
                        )
                    
                    with col2:
                        used_margin = float(data.get('utilised', {}).get('grossUtilization', 0))
                        st.metric(
                            "Used Margin", 
                            f"₹{used_margin:,.2f}"
                        )
                    
                    with col3:
                        available_margin = float(data.get('net', 0))
                        st.metric(
                            "Available Margin", 
                            f"₹{available_margin:,.2f}"
                        )
                    
                    # Additional account details
                    st.subheader("Account Details")
                    details_col1, details_col2 = st.columns(2)
                    
                    with details_col1:
                        st.write("Trading Segment:", data.get('segment', 'N/A'))
                        st.write("Product Type:", data.get('productType', 'N/A'))
                    
                    with details_col2:
                        st.write("Exchange:", data.get('exchange', 'N/A'))
                        st.write("Last Update:", datetime.now().strftime("%H:%M:%S"))
                else:
                    st.warning("Unable to fetch account details. Please check your connection.")
            else:
                st.error("SmartAPI client not properly initialized")
            
        except Exception as e:
            st.error("Error fetching account details")
            if st.checkbox("Show Error Details"):
                st.exception(e)

    # Portfolio Tab with Dark UI theme
    with tabs[1]:
        st.markdown("""
        <style>
        /* Dark theme colors */
        :root {
            --background-color: #1a1a1a;
            --text-color: #ffffff;
            --card-background: #2d2d2d;
            --border-color: #404040;
            --accent-color: #007AFF;
            --success-color: #4CAF50;
            --danger-color: #FF5252;
        }
        
        /* Card styling */
        .stCard {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Metric styling */
        .metric-card {
            background: linear-gradient(145deg, #2d2d2d, #353535);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Table styling */
        .dataframe {
            background-color: var(--card-background);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }
        
        /* Button styling */
        .stButton>button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Portfolio Header with Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            if isinstance(smart_api, SmartAPIClient):
                # Get positions using SmartAPIClient
                positions_df = smart_api.get_positions()
                
                # Display positions if available
                if not positions_df.empty:
                    st.dataframe(positions_df)
                else:
                    st.info("No positions found")
                    
                # Get portfolio data
                portfolio_df = smart_api.get_portfolio()
                if not portfolio_df.empty:
                    st.subheader("Portfolio Holdings")
                    st.dataframe(portfolio_df)
                else:
                    st.info("No portfolio holdings found")
                    
            else:
                st.error("SmartAPI client not properly initialized")
                
        except Exception as e:
            st.error(f"Error fetching portfolio data: {str(e)}")
            if st.checkbox("Show detailed error"):
                st.exception(e)

    # Market Data Tab
    with tabs[2]:
        st.header("Market Data")
        
        # Create subtabs for Live and Historical data
        market_subtabs = st.tabs(["📊 Live Data", "Historical Data"])
        
        # Live Data Subtab
        with market_subtabs[0]:
            st.subheader("Live Market Data")
            
            col1, col2 = st.columns(2)
            with col1:
                live_index = st.selectbox(
                    "Select Index",
                    ["NIFTY 50", "BANK NIFTY", "FIN NIFTY", "MIDCAP NIFTY"],
                    key="live_market_index"
                )
            
            with col2:
                auto_refresh = st.checkbox("Auto Refresh (5s)", value=False)
            
            # Map index names to tokens for live data
            live_index_token_map = {
                "NIFTY 50": "99926000",
                "BANK NIFTY": "99926009",
                "FIN NIFTY": "99926037",
                "MIDCAP NIFTY": "99926012"
            }
            
            try:
                if isinstance(smart_api, SmartAPIClient):
                    # Get current market date (only on weekdays)
                    current_date = datetime.now()
                    if current_date.weekday() >= 5:  # Saturday or Sunday
                        # Get last Friday
                        days_to_subtract = current_date.weekday() - 4
                        current_date = current_date - timedelta(days=days_to_subtract)
                    
                    # Format dates for market hours (9:15 AM to 3:30 PM)
                    from_date = current_date.strftime("%Y-%m-%d 09:15")
                    to_date = current_date.strftime("%Y-%m-%d 15:30")
                    
                    # Fetch live data
                    live_data = smart_api.get_candle_data(
                        exchange="NSE",
                        token=live_index_token_map[live_index],
                        interval="5",  # 5-minute candles
                        from_date=from_date,
                        to_date=to_date
                    )
                    
                    if not live_data.empty:
                        # Display live chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=live_data.index,
                            open=live_data['open'],
                            high=live_data['high'],
                            low=live_data['low'],
                            close=live_data['close']
                        )])
                        
                        fig.update_layout(
                            title=f"{live_index} Live Data",
                            yaxis_title="Price",
                            xaxis_title="Time",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display latest values
                        latest = live_data.iloc[-1]
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                        with metrics_col1:
                            st.metric("Open", f"₹{latest['open']:,.2f}")
                        with metrics_col2:
                            st.metric("High", f"₹{latest['high']:,.2f}")
                        with metrics_col3:
                            st.metric("Low", f"₹{latest['low']:,.2f}")
                        with metrics_col4:
                            st.metric("Close", f"₹{latest['close']:,.2f}")
                    else:
                        st.warning("No live data available for the selected index")
                    
                    if auto_refresh:
                        time.sleep(5)
                        st.rerun()
                
            except Exception as e:
                st.error(f"Error fetching live data: {str(e)}")
                if st.checkbox("Show Error Details", key="live_data_error"):
                    st.exception(e)
        
        # Historical Data Subtab
        with market_subtabs[1]:
            st.subheader("Historical Market Data")
            
            # Market data filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                market_index = st.selectbox(
                    "Select Index",
                    ["NIFTY 50", "BANK NIFTY", "FIN NIFTY", "MIDCAP NIFTY"],
                    key="market_index"
                )
                
            with col2:
                market_timeframe = st.selectbox(
                    "Select Timeframe",
                    ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "1 Day"],
                    key="market_timeframe"
                )
                
            with col3:
                market_from_date = st.date_input(
                    "From Date",
                    datetime.now() - timedelta(days=7),
                    key="market_from_date"
                )
                
            with col4:
                market_to_date = st.date_input(
                    "To Date",
                    datetime.now(),
                    key="market_to_date"
                )
                
            if st.button("Fetch Historical Data"):
                try:
                    # Validate date range based on timeframe
                    if market_timeframe in ["1 Minute", "5 Minutes"]:
                        if (market_to_date - market_from_date).days > 7:
                            st.warning("For 1 and 5 minute timeframes, maximum period is 7 days. Adjusting date range.")
                            market_from_date = market_to_date - timedelta(days=7)
                    elif market_timeframe in ["15 Minutes", "30 Minutes"]:
                        if (market_to_date - market_from_date).days > 15:
                            st.warning("For 15 and 30 minute timeframes, maximum period is 15 days. Adjusting date range.")
                            market_from_date = market_to_date - timedelta(days=15)
                    
                    # Map index names to tokens
                    index_token_map = {
                        "NIFTY 50": "99926000",
                        "BANK NIFTY": "99926009",
                        "FIN NIFTY": "99926037",
                        "MIDCAP NIFTY": "99926012"
                    }
                    
                    # Map timeframe to API parameters
                    timeframe_map = {
                        "1 Minute": "ONE_MINUTE",
                        "5 Minutes": "FIVE_MINUTE",
                        "15 Minutes": "FIFTEEN_MINUTE",
                        "30 Minutes": "THIRTY_MINUTE",
                        "1 Hour": "ONE_HOUR",
                        "1 Day": "ONE_DAY"
                    }
                    
                    # Prepare parameters for API call
                    params = {
                        "exchange": "NSE",
                        "symboltoken": index_token_map[market_index],
                        "interval": timeframe_map[market_timeframe],
                        "fromdate": market_from_date.strftime("%Y-%m-%d 09:15"),
                        "todate": market_to_date.strftime("%Y-%m-%d 15:30")
                    }
                    
                    # Fetch market data
                    market_data = fetch_market_data(smart_api, params)
                    
                    if market_data.empty:
                        st.error("Failed to fetch market data")
                    else:
                        # Display candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=market_data.index,
                            open=market_data['open'],
                            high=market_data['high'],
                            low=market_data['low'],
                            close=market_data['close']
                        )])
                        
                        fig.update_layout(
                            title=f'{market_index} Price Movement',
                            yaxis_title='Price',
                            xaxis_title='Date',
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Price",
                                f"₹{market_data['close'].iloc[-1]:,.2f}",
                                f"{((market_data['close'].iloc[-1] - market_data['close'].iloc[-2])/market_data['close'].iloc[-2]*100):,.2f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Day High",
                                f"₹{market_data['high'].max():,.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Day Low",
                                f"₹{market_data['low'].min():,.2f}"
                            )
                        
                        with col4:
                            st.metric(
                                "Volume",
                                f"{market_data['volume'].iloc[-1]:,.0f}"
                            )
                        
                        # Display data table
                        st.subheader("Market Data")
                        st.dataframe(
                            market_data.sort_values(ascending=False),
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error fetching historical data: {str(e)}")

    # Orders Tab
    with tabs[3]:
        st.header("Orders")
        
        # Create subtabs for different order types
        order_tabs = st.tabs(["🔄 Active Orders", "📝 Place Order", "📜 Order History"])
        
        # Active Orders Tab
        with order_tabs[0]:
            st.subheader("Active Orders")
            try:
                # Fetch active orders
                active_orders = smart_api.orderBook()
                if active_orders and active_orders.get('status'):
                    if active_orders.get('data'):
                        # Convert to DataFrame
                        orders_df = pd.DataFrame(active_orders['data'])
                        
                        # Format and display orders
                        display_cols = [
                            'orderid', 'tradingsymbol', 'transactiontype',
                            'producttype', 'quantity', 'price', 'orderstatus',
                            'ordertype', 'exchtime'
                        ]
                        
                        st.dataframe(
                            orders_df[display_cols].sort_values('exchtime', ascending=False),
                            use_container_width=True
                        )
                        
                        # Add cancel order functionality
                        order_to_cancel = st.selectbox(
                            "Select Order to Cancel",
                            orders_df['orderid'].tolist()
                        )
                        
                        if st.button("Cancel Selected Order"):
                            try:
                                cancel_response = smart_api.cancelOrder(
                                    order_to_cancel,
                                    orders_df[orders_df['orderid'] == order_to_cancel]['variety'].iloc[0]
                                )
                                if cancel_response and cancel_response.get('status'):
                                    st.success("Order cancelled successfully!")
                                else:
                                    st.error("Failed to cancel order")
                            except Exception as e:
                                st.error(f"Error cancelling order: {e}")
                    else:
                        st.info("No active orders found")
                else:
                    st.error("Failed to fetch active orders")
            except Exception as e:
                st.error(f"Error fetching orders: {e}")
        
        # Place Order Tab
        with order_tabs[1]:
            st.subheader("Place New Order")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Symbol search
                exchange = st.selectbox(
                    "Exchange",
                    ["NSE", "BSE", "NFO"],
                    key="order_exchange"
                )
                
                symbol = st.text_input(
                    "Symbol",
                    placeholder="Enter trading symbol",
                    key="order_symbol"
                ).upper()
                
                selected_symbol = None
                token = None
                
                if symbol:
                    try:
                        search_result = smart_api.searchScrip(exchange, symbol)
                        if search_result and search_result.get('status'):
                            symbols_df = pd.DataFrame(search_result['data'])
                            selected_symbol = st.selectbox(
                                "Select Trading Symbol",
                                symbols_df['tradingsymbol'].tolist()
                            )
                            
                            if selected_symbol:
                                symbol_data = symbols_df[symbols_df['tradingsymbol'] == selected_symbol].iloc[0]
                                token = symbol_data['symboltoken']
                                
                                # Display symbol details
                                st.write("Symbol Details:")
                                st.write(f"Symbol: {selected_symbol}")
                                st.write(f"Token: {token}")
                                st.write(f"Exchange: {exchange}")
                    except Exception as e:
                        st.error(f"Error searching symbol: {e}")
            
            with col2:
                # Order details
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["BUY", "SELL"]
                )
                
                product_type = st.selectbox(
                    "Product Type",
                    ["DELIVERY", "INTRADAY", "MARGIN"]
                )
                
                order_type = st.selectbox(
                    "Order Type",
                    ["MARKET", "LIMIT", "SL", "SL-M"]
                )
            
            col3, col4 = st.columns(2)
            
            with col3:
                quantity = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=1
                )
                
                if order_type in ["LIMIT", "SL"]:
                    price = st.number_input(
                        "Price",
                        min_value=0.05,
                        value=0.0,
                        step=0.05
                    )
                else:
                    price = 0
                
                if order_type in ["SL", "SL-M"]:
                    trigger_price = st.number_input(
                        "Trigger Price",
                        min_value=0.05,
                        value=0.0,
                        step=0.05
                    )
                else:
                    trigger_price = 0
            
            with col4:
                validity = st.selectbox(
                    "Validity",
                    ["DAY", "IOC"]
                )
                
                variety = st.selectbox(
                    "Variety",
                    ["NORMAL", "STOPLOSS", "AMO"]
                )
            
            if st.button("Place Order"):
                if not (selected_symbol and token):
                    st.error("Please select a valid symbol first")
                else:
                    try:
                        order_params = {
                            "variety": variety,
                            "tradingsymbol": selected_symbol,
                            "symboltoken": token,
                            "transactiontype": transaction_type,
                            "exchange": exchange,
                            "ordertype": order_type,
                            "producttype": product_type,
                            "duration": validity,
                            "quantity": quantity
                        }
                        
                        if order_type in ["LIMIT", "SL"]:
                            order_params["price"] = price
                            
                        if order_type in ["SL", "SL-M"]:
                            order_params["triggerprice"] = trigger_price
                        
                        order_response = smart_api.placeOrder(order_params)
                        
                        if order_response and order_response.get('status'):
                            st.success(f"Order placed successfully! Order ID: {order_response['data']['orderid']}")
                        else:
                            st.error("Failed to place order")
                    except Exception as e:
                        st.error(f"Error placing order: {e}")
        
        # Order History Tab
        with order_tabs[2]:
            st.subheader("Order History")
            
            try:
                # Fetch order history
                order_history = smart_api.orderBook()
                if order_history and order_history.get('status'):
                    if order_history.get('data'):
                        history_df = pd.DataFrame(order_history['data'])
                        
                        # Format and display history
                        display_cols = [
                            'orderid', 'tradingsymbol', 'transactiontype',
                            'producttype', 'quantity', 'price', 'orderstatus',
                            'ordertype', 'exchtime'
                        ]
                        
                        st.dataframe(
                            history_df[display_cols].sort_values('exchtime', ascending=False),
                            use_container_width=True
                        )
                        
                        # Add export functionality
                        if st.button("Export Order History"):
                            csv = history_df.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                "order_history.csv",
                                "text/csv",
                                key='download-csv'
                            )
                    else:
                        st.info("No orders in history")
                else:
                    st.error("Failed to fetch order history")
                    
            except Exception as e:
                st.error(f"Error fetching order history: {e}")

    # Analysis Tab
    with tabs[4]:
        st.header("Technical Analysis")
        
        # First row - Index, Timeframe, and Date Selection
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            index_mapping = {
                "NIFTY 50": "99926000",
                "BANK NIFTY": "99926009",
                "FIN NIFTY": "99926037",
                "MIDCAP NIFTY": "99926012"
            }
            selected_index = st.selectbox(
                "Select Index",
                list(index_mapping.keys()),
                index=0
            )
        with col2:
            timeframe_mapping = {
                "1 Minute": "ONE_MINUTE",
                "5 Minutes": "FIVE_MINUTE",
                "15 Minutes": "FIFTEEN_MINUTE",
                "30 Minutes": "THIRTY_MINUTE",
                "1 Hour": "ONE_HOUR",
                "1 Day": "ONE_DAY"
            }
            selected_timeframe = st.selectbox(
                "Select Timeframe",
                list(timeframe_mapping.keys()),
                index=5
            )
        with col3:
            from_date = st.date_input("From Date", datetime.now() - timedelta(days=30))
        with col4:
            to_date = st.date_input("To Date", datetime.now())
            
        # Second row - Moving Average Parameters
        st.subheader("Moving Average Parameters")
        ma_col1, ma_col2, ma_col3, ma_col4 = st.columns(4)
        
        with ma_col1:
            fast_ma = st.number_input("Fast MA Length", min_value=1, value=9)
        with ma_col2:
            slow_ma = st.number_input("Slow MA Length", min_value=1, value=21)
        with ma_col3:
            fast_ema = st.number_input("Fast EMA Length", min_value=1, value=12)
        with ma_col4:
            slow_ema = st.number_input("Slow EMA Length", min_value=1, value=26)
            
        if st.button("Get Historical Data"):
            try:
                # Adjust date range based on timeframe
                if selected_timeframe in ["1 Minute", "5 Minutes"]:
                    if (to_date - from_date).days > 7:
                        st.warning("For 1 and 5 minute timeframes, maximum period is 7 days. Adjusting date range.")
                        from_date = to_date - timedelta(days=7)
                elif selected_timeframe in ["15 Minutes", "30 Minutes"]:
                    if (to_date - from_date).days > 15:
                        st.warning("For 15 and 30 minute timeframes, maximum period is 15 days. Adjusting date range.")
                        from_date = to_date - timedelta(days=15)
                
                params = {
                    "exchange": "NSE",
                    "symboltoken": index_mapping[selected_index],
                    "interval": timeframe_mapping[selected_timeframe],
                    "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
                    "todate": to_date.strftime("%Y-%m-%d 15:30")
                }
                
                hist_data = fetch_market_data(smart_api, params)
                if not hist_data.empty:
                    # Calculate indicators
                    hist_data[f'MA_{fast_ma}'] = hist_data['close'].rolling(window=fast_ma).mean()
                    hist_data[f'MA_{slow_ma}'] = hist_data['close'].rolling(window=slow_ma).mean()
                    hist_data[f'EMA_{fast_ema}'] = hist_data['close'].ewm(span=fast_ema, adjust=False).mean()
                    hist_data[f'EMA_{slow_ema}'] = hist_data['close'].ewm(span=slow_ema, adjust=False).mean()
                    hist_data['RSI'] = calculate_rsi(hist_data['close'])
                    hist_data['MACD'], hist_data['Signal'] = calculate_macd(hist_data['close'])
                    hist_data['MACD_Hist'] = hist_data['MACD'] - hist_data['Signal']
                    
                    # Generate crossover signals
                    hist_data['MA_Cross'] = np.where(hist_data[f'MA_{fast_ma}'] > hist_data[f'MA_{slow_ma}'], 1, -1)
                    hist_data['MA_Signal'] = hist_data['MA_Cross'].diff()
                    hist_data['EMA_Cross'] = np.where(hist_data[f'EMA_{fast_ema}'] > hist_data[f'EMA_{slow_ema}'], 1, -1)
                    hist_data['EMA_Signal'] = hist_data['EMA_Cross'].diff()
                    
                    # Current values
                    current_price = hist_data['close'].iloc[-1]
                    current_rsi = hist_data['RSI'].iloc[-1]
                    current_macd_hist = hist_data['MACD_Hist'].iloc[-1]
                    prev_macd_hist = hist_data['MACD_Hist'].iloc[-2]
                    
                    # Display summary and signals
                    st.subheader(f"{selected_index} Analysis & Signals")
                    
                    # Market metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"₹{current_price:,.2f}", 
                                f"{((current_price - hist_data['close'].iloc[-2])/hist_data['close'].iloc[-2]*100):,.2f}%")
                    with col2:
                        st.metric("RSI", f"{current_rsi:.2f}")
                    with col3:
                        st.metric("MACD Histogram", f"{current_macd_hist:.2f}")
                    
                    # Signal Analysis
                    signal_col1, signal_col2 = st.columns(2)
                    
                    with signal_col1:
                        st.markdown("### Trading Signals")
                        
                        # Combined MA/EMA Crossover Analysis
                        bullish_crossover = (hist_data['MA_Signal'].iloc[-1] == 2 or hist_data['EMA_Signal'].iloc[-1] == 2)
                        bearish_crossover = (hist_data['MA_Signal'].iloc[-1] == -2 or hist_data['EMA_Signal'].iloc[-1] == -2)
                        
                        # RSI Confirmation
                        rsi_bullish = current_rsi < 40
                        rsi_bearish = current_rsi > 60
                        
                        # MACD Confirmation
                        macd_bullish = current_macd_hist > 0 and prev_macd_hist < 0
                        macd_bearish = current_macd_hist < 0 and prev_macd_hist > 0
                        
                        # Generate Strong Buy Signals
                        if bullish_crossover and rsi_bullish and current_macd_hist > 0:
                            st.success("🟢 Strong Buy Signal - Consider Call Options")
                            st.write("- Bullish MA/EMA Crossover")
                            st.write("- RSI showing oversold/bullish")
                            st.write("- MACD confirms upward momentum")
                            
                        # Generate Strong Sell Signals
                        elif bearish_crossover and rsi_bearish and current_macd_hist < 0:
                            st.error("🔴 Strong Sell Signal - Consider Put Options")
                            st.write("- Bearish MA/EMA Crossover")
                            st.write("- RSI showing overbought/bearish")
                            st.write("- MACD confirms downward momentum")
                            
                        # Moderate Signals
                        elif bullish_crossover or (rsi_bullish and macd_bullish):
                            st.info("🔵 Moderate Buy Signal - Watch for confirmation")
                            st.write("- Some bullish indicators present")
                            st.write("- Wait for additional confirmation")
                            
                        elif bearish_crossover or (rsi_bearish and macd_bearish):
                            st.warning("🟡 Moderate Sell Signal - Watch for confirmation")
                            st.write("- Some bearish indicators present")
                            st.write("- Wait for additional confirmation")
                            
                        else:
                            st.write("No clear trading signals at the moment")
                    
                    with signal_col2:
                        st.markdown("### Technical Levels")
                        st.write(f"Current RSI: {current_rsi:.2f}")
                        st.write(f"MACD Histogram: {current_macd_hist:.2f}")
                        st.write(f"Fast MA ({fast_ma}): ₹{hist_data[f'MA_{fast_ma}'].iloc[-1]:,.2f}")
                        st.write(f"Slow MA ({slow_ma}): ₹{hist_data[f'MA_{slow_ma}'].iloc[-1]:,.2f}")
                        st.write(f"Fast EMA ({fast_ema}): ₹{hist_data[f'EMA_{fast_ema}'].iloc[-1]:,.2f}")
                        st.write(f"Slow EMA ({slow_ema}): ₹{hist_data[f'EMA_{slow_ema}'].iloc[-1]:,.2f}")
                    
                    # Candlestick chart with indicators
                    fig = go.Figure()
                    
                    # Add candlestick
                    fig.add_trace(go.Candlestick(
                        x=hist_data.index,
                        open=hist_data['open'],
                        high=hist_data['high'],
                        low=hist_data['low'],
                        close=hist_data['close'],
                        name='Price'
                    ))
                    
                    # Add MAs and EMAs
                    fig.add_trace(go.Scatter(
                        x=hist_data.index, 
                        y=hist_data[f'MA_{fast_ma}'],
                        name=f'MA {fast_ma}',
                        line=dict(color='orange')
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist_data.index, 
                        y=hist_data[f'MA_{slow_ma}'],
                        name=f'MA {slow_ma}',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist_data.index, 
                        y=hist_data[f'EMA_{fast_ema}'],
                        name=f'EMA {fast_ema}',
                        line=dict(color='yellow', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist_data.index, 
                        y=hist_data[f'EMA_{slow_ema}'],
                        name=f'EMA {slow_ema}',
                        line=dict(color='purple', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_index} Price Movement with Indicators',
                        yaxis_title='Price',
                        xaxis_title='Date',
                        template='plotly_dark',
                        yaxis=dict(tickformat=".2f"),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display OHLC data with indicators
                    display_df = hist_data.tail().copy()
                    st.dataframe(display_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching historical data: {e}")

    # Symbol Search Tab
    with tabs[5]:
        st.header("Symbol Search")
        
        col1, col2 = st.columns(2)
        with col1:
            exchange = st.selectbox(
                "Select Exchange",
                ["NSE", "BSE", "NFO", "MCX"],
                key="search_exchange"
            )
        with col2:
            search_query = st.text_input("Search Symbol", 
                placeholder="Enter name like RELIANCE, NIFTY, BANKNIFTY, etc.").upper()
            
        if search_query:
            try:
                search_result = smart_api.searchScrip(exchange, search_query)
                if search_result['status']:
                    df = pd.DataFrame(search_result['data'])
                    if len(df) > 0:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info(f"No symbols found matching '{search_query}' in {exchange}")
            except Exception as e:
                st.error(f"Error searching symbols: {e}")

        with st.expander("Search Tips"):
            st.markdown("""
            - For **Equity**, simply enter the company name or symbol (e.g., RELIANCE, TCS)
            - For **Index**, enter the index name (e.g., NIFTY, BANKNIFTY)
            - For **F&O**, search the underlying (e.g., RELIANCE for Reliance futures/options)
            - For **Commodities**, enter commodity name (e.g., GOLD, SILVER)
            """)

    # Option Chain Tab
    with tabs[6]:
        st.header("Option Chain")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            index = st.selectbox(
                "Select Index",
                ["NIFTY", "BANKNIFTY", "FINNIFTY"],
                key="option_index"
            )
        
        with col2:
            try:
                search_params = {"exchange": "NFO", "searchscrip": index}
                chain_response = fetch_option_chain(smart_api, search_params)
                
                if chain_response['status']:
                    df = pd.DataFrame(chain_response['data'])
                    if 'expiry' in df.columns:
                        expiry_dates = sorted(pd.to_datetime(df['expiry'].unique()))
                        selected_expiry = st.selectbox(
                            "Select Expiry",
                            expiry_dates,
                            key="option_expiry"
                        )
            except Exception as e:
                st.error(f"Error fetching expiry dates: {e}")
        
        with col3:
            spot_price = st.number_input("Spot Price", min_value=0.0, value=0.0, step=0.05)

    # Settings Tab
    with tabs[7]:
        st.header("Settings")
        
        # Create sections for different settings
        st.subheader("API Configuration")
        api_col1, api_col2 = st.columns(2)
        
        with api_col1:
            st.text_input(
                "API Key",
                value=os.getenv('ANGEL_API_KEY', ''),
                type="password",
                key="api_key_input"
            )
            st.text_input(
                "Client ID",
                value=os.getenv('ANGEL_CLIENT_ID', ''),
                key="client_id_input"
            )
        
        with api_col2:
            st.text_input(
                "Password",
                value=os.getenv('ANGEL_PASSWORD', ''),
                type="password",
                key="password_input"
            )
            st.text_input(
                "TOTP Key",
                value=os.getenv('ANGEL_TOTP_KEY', ''),
                type="password",
                key="totp_key_input"
            )

        if st.button("Save API Settings"):
            try:
                # Create or update .env file
                env_file = ".env"
                env_data = {
                    "ANGEL_API_KEY": st.session_state.api_key_input,
                    "ANGEL_CLIENT_ID": st.session_state.client_id_input,
                    "ANGEL_PASSWORD": st.session_state.password_input,
                    "ANGEL_TOTP_KEY": st.session_state.totp_key_input
                }
                
                with open(env_file, "w") as f:
                    for key, value in env_data.items():
                        f.write(f"{key}={value}\n")
                
                st.success("API settings saved successfully! Please restart the application.")
            except Exception as e:
                st.error(f"Error saving settings: {e}")

        # Trading Settings
        st.subheader("Trading Settings")
        trading_col1, trading_col2 = st.columns(2)
        
        with trading_col1:
            default_quantity = st.number_input(
                "Default Trading Quantity",
                min_value=1,
                value=1,
                help="Default quantity for trading orders"
            )
            
            risk_percentage = st.slider(
                "Risk Percentage per Trade",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Maximum risk percentage per trade"
            )

        with trading_col2:
            default_stoploss = st.number_input(
                "Default Stop Loss (%)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Default stop loss percentage"
            )
            
            default_target = st.number_input(
                "Default Target (%)",
                min_value=0.1,
                max_value=20.0,
                value=4.0,
                step=0.1,
                help="Default target percentage"
            )

        if st.button("Save Trading Settings"):
            try:
                # Save trading settings to a JSON file
                trading_settings = {
                    "default_quantity": default_quantity,
                    "risk_percentage": risk_percentage,
                    "default_stoploss": default_stoploss,
                    "default_target": default_target
                }
                
                with open("trading_settings.json", "w") as f:
                    json.dump(trading_settings, f, indent=4)
                
                st.success("Trading settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving trading settings: {e}")

        # Display Settings
        st.subheader("Display Settings")
        display_col1, display_col2 = st.columns(2)
        
        with display_col1:
            chart_theme = st.selectbox(
                "Chart Theme",
                ["plotly_dark", "plotly", "plotly_white"],
                index=0,
                help="Select the theme for charts"
            )
            
            show_indicators = st.multiselect(
                "Default Indicators",
                ["RSI", "MACD", "Moving Averages", "Bollinger Bands"],
                default=["RSI", "MACD"],
                help="Select default indicators to show on charts"
            )

        with display_col2:
            auto_refresh_interval = st.number_input(
                "Auto Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=5,
                help="Interval for auto-refreshing data"
            )
            
            show_debug_info = st.checkbox(
                "Show Debug Information",
                value=False,
                help="Show additional debugging information"
            )

        if st.button("Save Display Settings"):
            try:
                # Save display settings to a JSON file
                display_settings = {
                    "chart_theme": chart_theme,
                    "show_indicators": show_indicators,
                    "auto_refresh_interval": auto_refresh_interval,
                    "show_debug_info": show_debug_info
                }
                
                with open("display_settings.json", "w") as f:
                    json.dump(display_settings, f, indent=4)
                
                st.success("Display settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving display settings: {e}")

        # Add import/export settings functionality
        st.subheader("Backup & Restore")
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            if st.button("Export Settings"):
                try:
                    # Combine all settings
                    all_settings = {
                        "trading_settings": trading_settings,
                        "display_settings": display_settings
                    }
                    
                    # Convert to JSON string
                    settings_json = json.dumps(all_settings, indent=4)
                    
                    # Create download button
                    st.download_button(
                        "Download Settings",
                        settings_json,
                        "trading_bot_settings.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Error exporting settings: {e}")

        with backup_col2:
            uploaded_file = st.file_uploader(
                "Import Settings",
                type="json",
                help="Upload a previously exported settings file"
            )
            
            if uploaded_file is not None:
                try:
                    imported_settings = json.load(uploaded_file)
                    
                    # Save the imported settings
                    with open("trading_settings.json", "w") as f:
                        json.dump(imported_settings["trading_settings"], f, indent=4)
                    
                    with open("display_settings.json", "w") as f:
                        json.dump(imported_settings["display_settings"], f, indent=4)
                    
                    st.success("Settings imported successfully! Please refresh the page.")
                except Exception as e:
                    st.error(f"Error importing settings: {e}")

    # Algo AI Tab
    with tabs[8]:
        st.header("Algorithmic Trading AI")
        
        # Create a container for algo_ai functionality
        algo_container = st.container()
        
        with algo_container:
            # Initialize algo_ai session state if needed
            if 'ma_length' not in st.session_state:
                st.session_state.ma_length = 10
            if 'ema_length' not in st.session_state:
                st.session_state.ema_length = 10
            
            # Use algo_ai functionality
            symbol = st.selectbox(
                "Select Index",
                options=list(algo_ai.INDICES.keys()),
                key='algo_ai_symbol'
            )
            symbol = algo_ai.INDICES[symbol]
            
            # Get data and calculate indicators
            df = algo_ai.get_data(symbol)
            if df is not None:
                df = algo_ai.calculate_indicators(df)
                df = algo_ai.generate_signals(df)
                
                if df is not None:
                    # Process historical signals
                    algo_ai.process_historical_signals(df, symbol)
                    
                    # Create sub-tabs for Algo AI
                    ai_tabs = st.tabs([
                        "Chart & Signals",
                        "Paper Trading",
                        "Portfolio",
                        "Options Chain"
                    ])
                    
                    with ai_tabs[0]:
                        algo_ai.display_metrics(df)
                        fig = algo_ai.plot_chart(df, f"{symbol} - 5min")
                        st.plotly_chart(fig, use_container_width=True)
                        algo_ai.display_strategy_dashboard(df, symbol)
                    
                    with ai_tabs[1]:
                        process_signals(df, smart_api)
                    
                    with ai_tabs[2]:
                        algo_ai.display_portfolio_dashboard(df, symbol)
                    
                    with ai_tabs[3]:
                        algo_ai.display_options_analysis(df, symbol)

    # NIFTY Options AI Tab
    with tabs[9]:
        try:
            nifty_options_ai = NiftyOptionsAI()
            nifty_options_ai.display_dashboard()
        except Exception as e:
            st.error(f"Error in NIFTY Options AI: {str(e)}")
            if st.checkbox("Show Debug Info", key="main_debug_info"):
                st.error(f"Detailed error: {str(e)}")

if __name__ == "__main__":
    main()