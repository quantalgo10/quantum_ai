import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from SmartApi import SmartConnect
import pyotp
import logging
from scipy.stats import norm
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QuantumTrader:
    def __init__(self):
        self.setup_logging()
        self.initialize_session_state()
        self.setup_theme()
        self.connect_broker()
        
        # Enhanced strategies
        self.strategies = {
            'Quantum Momentum': self.quantum_momentum_strategy,
            'Neural Network': self.neural_network_strategy,
            'Options Flow': self.options_flow_strategy,
            'Smart Money': self.smart_money_strategy,
            'Volatility Edge': self.volatility_edge_strategy
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'positions' not in st.session_state:
            st.session_state.positions = []
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'cash' not in st.session_state:
            st.session_state.cash = 100000  # Initial capital
        if 'pnl' not in st.session_state:
            st.session_state.pnl = 0

    def setup_theme(self):
        """Configure modern dark theme for the app"""
        st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --background-color: #0E1117;
            --secondary-bg: #1E1E1E;
            --text-color: #FFFFFF;
            --accent-color: #00FF9D;
            --error-color: #FF4B4B;
            --success-color: #00CC8E;
            --warning-color: #FFA500;
        }
        
        /* Modern card styling */
        .stCard {
            background: linear-gradient(145deg, #1E1E1E, #2D2D2D);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Metric styling */
        .metric-card {
            background: linear-gradient(145deg, #2D2D2D, #1E1E1E);
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        /* Custom button styling */
        .stButton>button {
            background: linear-gradient(45deg, var(--accent-color), #00CC8E);
            color: black;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.3);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: var(--secondary-bg);
            border-radius: 10px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: var(--text-color);
            border-radius: 5px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, var(--accent-color), #00CC8E);
            color: black;
        }
        
        /* Chart styling */
        .js-plotly-plot {
            background-color: var(--secondary-bg);
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Dataframe styling */
        .dataframe {
            background-color: var(--secondary-bg);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    def display_dashboard(self):
        """Enhanced main dashboard display"""
        st.set_page_config(
            layout="wide",
            page_title="Quantum Trader AI",
            page_icon="🧠"
        )

        # Sidebar
        with st.sidebar:
            st.image("logo.png", width=200)  # Add your logo
            st.title("Quantum Trader AI")
            
            # Account Info
            st.subheader("Account Overview")
            account_metrics = self.get_account_metrics()
            
            for metric in account_metrics:
                st.metric(
                    metric['label'],
                    metric['value'],
                    metric['delta'],
                    delta_color=metric['delta_color']
                )
            
            # Strategy Selection
            selected_strategy = st.selectbox(
                "Select Strategy",
                list(self.strategies.keys()),
                format_func=lambda x: f"🤖 {x}"
            )
            
            # Quick Actions
            st.subheader("Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Start Trading"):
                    self.start_trading(selected_strategy)
            with col2:
                if st.button("⏹️ Stop Trading"):
                    self.stop_trading()

        # Main Content
        main_tabs = st.tabs([
            "📊 Dashboard",
            "📈 Live Trading",
            "🧪 Strategy Lab",
            "📉 Risk Analytics",
            "⚙️ Settings"
        ])

        with main_tabs[0]:
            self.display_main_dashboard()
            
        with main_tabs[1]:
            self.display_live_trading()
            
        with main_tabs[2]:
            self.display_strategy_lab()
            
        with main_tabs[3]:
            self.display_risk_analytics()
            
        with main_tabs[4]:
            self.display_settings()

    def display_main_dashboard(self):
        """Enhanced main dashboard with modern UI"""
        # Header Section
        st.header("Trading Dashboard")
        
        # Key Metrics Row
        metrics = self.get_key_metrics()
        cols = st.columns(4)
        for col, metric in zip(cols, metrics):
            with col:
                self.display_metric_card(
                    metric['label'],
                    metric['value'],
                    metric['delta'],
                    metric['icon']
                )

        # Charts Section
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Portfolio Performance")
            self.plot_portfolio_performance()
            
        with chart_col2:
            st.subheader("Strategy Returns")
            self.plot_strategy_returns()

        # Active Positions
        st.subheader("Active Positions")
        positions = self.get_active_positions()
        if positions:
            self.display_positions_table(positions)
        else:
            st.info("No active positions")

        # Recent Trades
        st.subheader("Recent Trades")
        trades = self.get_recent_trades()
        if trades:
            self.display_trades_table(trades)
        else:
            st.info("No recent trades")

    def display_metric_card(self, label, value, delta, icon):
        """Display a modern metric card"""
        html = f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.2rem;">{icon} {label}</span>
                <span style="font-size: 1.5rem; font-weight: bold;">{value}</span>
            </div>
            <div style="color: {'#00CC8E' if float(delta.strip('%')) > 0 else '#FF4B4B'}; 
                        font-size: 0.9rem; margin-top: 0.5rem;">
                {delta}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    def plot_portfolio_performance(self):
        """Create an interactive portfolio performance chart"""
        fig = go.Figure()
        
        # Add portfolio value line
        fig.add_trace(go.Scatter(
            x=self.get_portfolio_history()['date'],
            y=self.get_portfolio_history()['value'],
            name="Portfolio Value",
            line=dict(color="#00FF9D", width=2),
            fill='tozeroy',
            fillcolor="rgba(0, 255, 157, 0.1)"
        ))
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1E1E1E",
            paper_bgcolor="#1E1E1E",
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)",
                showline=True,
                linecolor="rgba(255, 255, 255, 0.1)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def quantum_momentum_strategy(self, data):
        """
        Advanced momentum strategy using quantum-inspired algorithms
        """
        try:
            # Calculate technical indicators
            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            data['MACD'] = ta.trend.MACD(data['Close']).macd()
            data['Signal'] = ta.trend.MACD(data['Close']).macd_signal()
            
            # Calculate Quantum Oscillator (proprietary indicator)
            data['QO'] = self.calculate_quantum_oscillator(data)
            
            # Generate signals
            data['Position'] = 0
            
            # Buy conditions
            buy_condition = (
                (data['RSI'] < 40) & 
                (data['MACD'] > data['Signal']) & 
                (data['QO'] > 0.6)
            )
            
            # Sell conditions
            sell_condition = (
                (data['RSI'] > 60) & 
                (data['MACD'] < data['Signal']) & 
                (data['QO'] < -0.6)
            )
            
            data.loc[buy_condition, 'Position'] = 1
            data.loc[sell_condition, 'Position'] = -1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in quantum momentum strategy: {str(e)}")
            return None

    def calculate_quantum_oscillator(self, data):
        """
        Proprietary quantum-inspired oscillator
        """
        try:
            # Calculate price momentum
            returns = data['Close'].pct_change()
            
            # Calculate quantum probability wave
            wave = np.exp(-returns**2 / 0.02)  # Quantum wave function
            
            # Calculate quantum interference pattern
            interference = wave.rolling(window=20).mean() * np.cos(returns.rolling(window=10).std() * np.pi)
            
            # Normalize to [-1, 1]
            normalized = 2 * (interference - interference.min()) / (interference.max() - interference.min()) - 1
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error calculating quantum oscillator: {str(e)}")
            return None

    def risk_management(self, position_size, current_price, strategy='Quantum Momentum'):
        """
        Advanced risk management system
        """
        try:
            # Calculate position sizing based on Kelly Criterion
            win_rate = self.calculate_win_rate(strategy)
            odds_ratio = self.calculate_odds_ratio(strategy)
            kelly_fraction = (win_rate * odds_ratio - (1 - win_rate)) / odds_ratio
            
            # Apply position size limits
            max_position = st.session_state.cash * 0.1  # Max 10% per trade
            suggested_position = position_size * kelly_fraction
            
            # Calculate stop loss and take profit
            volatility = self.calculate_volatility(current_price)
            stop_loss = current_price * (1 - volatility * 2)
            take_profit = current_price * (1 + volatility * 3)
            
            return {
                'position_size': min(suggested_position, max_position),
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk management: {str(e)}")
            return None

    def connect_broker(self):
        """Connect to the broker"""
        try:
            api_key = os.getenv('ANGEL_API_KEY')
            client_id = os.getenv('ANGEL_CLIENT_ID')
            password = os.getenv('ANGEL_PASSWORD')
            totp_key = os.getenv('ANGEL_TOTP_KEY')
            
            # Initialize SmartConnect
            self.smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_key)
            
            # Generate session
            data = self.smart_api.generateSession(client_id, password, totp.now())
            
            if data.get('status'):
                self.feed_token = data['data']['feedToken']
                self.refresh_token = data['data']['refreshToken']
                self.smart_api.getProfile(self.refresh_token)
                
                # Initialize websocket
                self.initialize_websocket()
                
                self.logger.info("Successfully connected to broker")
                return True
            else:
                self.logger.error("Failed to connect to broker")
                return False
            
        except Exception as e:
            self.logger.error(f"Error connecting to broker: {str(e)}")
            return False

    def initialize_websocket(self):
        """Initialize websocket connection"""
        try:
            self.websocket = SmartWebSocket(
                feed_token=self.feed_token,
                client_code=os.getenv('ANGEL_CLIENT_ID'),
                order_placement_callback=self.on_order_update,
                socket_error_callback=self.on_socket_error,
                socket_close_callback=self.on_socket_close,
                market_status_messages_callback=self.on_market_status
            )
            
            # Connect websocket
            self.websocket.connect()
            
        except Exception as e:
            self.logger.error(f"Error initializing websocket: {str(e)}")

    def on_order_update(self, message):
        """Handle order updates"""
        self.logger.info(f"Order Update: {message}")

    def on_socket_error(self, error):
        """Handle socket errors"""
        self.logger.error(f"Socket Error: {error}")

    def on_socket_close(self):
        """Handle socket close"""
        self.logger.info("Socket Connection Closed")

    def on_market_status(self, message):
        """Handle market status updates"""
        self.logger.info(f"Market Status: {message}")

    def neural_network_strategy(self, data):
        """
        Neural network based trading strategy
        """
        try:
            # Feature engineering
            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            data['MACD'] = ta.trend.MACD(data['Close']).macd()
            data['BB_upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
            data['BB_lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
            
            # Generate signals using neural network predictions
            data['Signal'] = 0
            
            # Neural network prediction logic
            features = ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR']
            X = data[features].fillna(method='ffill')
            
            # Simple neural network decision rules (placeholder for actual model)
            data.loc[(data['RSI'] < 30) & (data['Close'] > data['BB_lower']), 'Signal'] = 1
            data.loc[(data['RSI'] > 70) & (data['Close'] < data['BB_upper']), 'Signal'] = -1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in neural network strategy: {str(e)}")
            return None

    def options_flow_strategy(self, data):
        """
        Options flow based trading strategy
        """
        try:
            # Calculate options flow indicators
            data['Call_Volume'] = self.get_call_volume(data)
            data['Put_Volume'] = self.get_put_volume(data)
            data['PC_Ratio'] = data['Put_Volume'] / data['Call_Volume']
            
            # Generate signals based on options flow
            data['Signal'] = 0
            data.loc[data['PC_Ratio'] > 2, 'Signal'] = 1  # Bullish when put/call ratio is high
            data.loc[data['PC_Ratio'] < 0.5, 'Signal'] = -1  # Bearish when put/call ratio is low
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in options flow strategy: {str(e)}")
            return None

    def smart_money_strategy(self, data):
        """
        Smart money flow tracking strategy
        """
        try:
            # Calculate smart money indicators
            data['Large_Trades'] = self.get_large_trades(data)
            data['Institution_Flow'] = self.get_institutional_flow(data)
            data['Dark_Pool_Activity'] = self.get_dark_pool_activity(data)
            
            # Generate signals based on smart money flow
            data['Signal'] = 0
            data.loc[(data['Large_Trades'] > 0) & (data['Institution_Flow'] > 0), 'Signal'] = 1
            data.loc[(data['Large_Trades'] < 0) & (data['Institution_Flow'] < 0), 'Signal'] = -1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in smart money strategy: {str(e)}")
            return None

    def volatility_edge_strategy(self, data):
        """
        Volatility based edge trading strategy
        """
        try:
            # Calculate volatility indicators
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
            data['Volatility'] = data['Close'].rolling(window=20).std()
            data['IV_Rank'] = self.calculate_iv_rank(data)
            
            # Generate signals based on volatility patterns
            data['Signal'] = 0
            
            # Long volatility when IV is low
            data.loc[data['IV_Rank'] < 20, 'Signal'] = 1
            
            # Short volatility when IV is high
            data.loc[data['IV_Rank'] > 80, 'Signal'] = -1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in volatility edge strategy: {str(e)}")
            return None

    # Helper methods for strategies
    def get_call_volume(self, data):
        """Get call options volume data"""
        # Placeholder - implement actual call volume data fetching
        return np.random.random(len(data)) * 1000

    def get_put_volume(self, data):
        """Get put options volume data"""
        # Placeholder - implement actual put volume data fetching
        return np.random.random(len(data)) * 1000

    def get_large_trades(self, data):
        """Get large trade flow data"""
        # Placeholder - implement actual large trades detection
        return np.random.random(len(data)) * 2 - 1

    def get_institutional_flow(self, data):
        """Get institutional order flow data"""
        # Placeholder - implement actual institutional flow detection
        return np.random.random(len(data)) * 2 - 1

    def get_dark_pool_activity(self, data):
        """Get dark pool activity data"""
        # Placeholder - implement actual dark pool data fetching
        return np.random.random(len(data)) * 2 - 1

    def calculate_iv_rank(self, data):
        """Calculate implied volatility rank"""
        # Placeholder - implement actual IV rank calculation
        return np.random.random(len(data)) * 100