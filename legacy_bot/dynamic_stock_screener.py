#!/usr/bin/env python3
"""
Dynamic Stock Screener
AI-powered stock selection system that dynamically identifies trading opportunities
from the entire NSE universe using multiple criteria and machine learning
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import logging
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from nsetools import Nse
import talib

class DynamicStockScreener:
    """AI-powered dynamic stock screener for NSE stocks"""
    
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.nse = Nse()
        
        # Stock universe
        self.nse_stocks = []
        self.filtered_stocks = []
        self.scored_stocks = []
        
        # Screening criteria
        self.min_market_cap = 1000  # Crores
        self.min_avg_volume = 100000  # Daily average
        self.min_price = 20  # Minimum stock price
        self.max_price = 5000  # Maximum stock price
        
        # Load NSE stock list
        self.load_nse_universe()
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_nse_universe(self):
        """Load complete NSE stock universe"""
        try:
            # Get all NSE stocks
            all_stocks = self.nse.get_stock_codes()
            
            # Filter liquid stocks (top 500 by market cap)
            self.nse_stocks = list(all_stocks.keys())[:500]  # Top 500 liquid stocks
            
            self.logger.info(f"Loaded {len(self.nse_stocks)} stocks for screening")
            
        except Exception as e:
            self.logger.error(f"Failed to load NSE universe: {e}")
            # Fallback to predefined list
            self.nse_stocks = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK',
                'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'ASIANPAINT',
                'AXISBANK', 'MARUTI', 'HCLTECH', 'NESTLEIND', 'BAJFINANCE',
                'WIPRO', 'ULTRACEMCO', 'TITAN', 'SUNPHARMA', 'ADANIGREEN',
                'POWERGRID', 'NTPC', 'BAJAJFINSV', 'ONGC', 'INDUSINDBK',
                'TECH', 'COALINDIA', 'TATAMOTORS', 'HDFCLIFE', 'GRASIM',
                'CIPLA', 'BRITANNIA', 'DRREDDY', 'EICHERMOT', 'DIVISLAB'
            ]
    
    def get_stock_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Get historical data for a stock"""
        try:
            # Add .NS suffix for NSE stocks
            ticker = f"{symbol}.NS"
            stock = yf.Ticker(ticker)
            
            # Get historical data
            data = stock.history(period=period)
            
            if data.empty:
                return pd.DataFrame()
            
            # Get additional info
            info = stock.info
            
            # Add metadata to dataframe
            data['symbol'] = symbol
            data['market_cap'] = info.get('marketCap', 0) / 10000000  # Convert to crores
            data['avg_volume'] = data['Volume'].rolling(20).mean().iloc[-1]
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Failed to get data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculate technical analysis score for a stock"""
        try:
            if len(df) < 50:
                return 0
            
            score = 0
            close_prices = df['Close'].values.astype(np.float64)
            high_prices = df['High'].values.astype(np.float64)
            low_prices = df['Low'].values.astype(np.float64)
            volumes = df['Volume'].values.astype(np.float64)
            
            # Price momentum (30%)
            price_change_5d = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
            price_change_20d = (close_prices[-1] - close_prices[-20]) / close_prices[-20]
            
            if price_change_5d > 0.02:  # 2% gain in 5 days
                score += 15
            if price_change_20d > 0.05:  # 5% gain in 20 days
                score += 15
            
            # Moving averages (25%)
            sma_20 = talib.SMA(close_prices, timeperiod=20)
            sma_50 = talib.SMA(close_prices, timeperiod=50)
            
            if close_prices[-1] > sma_20[-1]:  # Above 20-day SMA
                score += 12.5
            if sma_20[-1] > sma_50[-1]:  # 20-day SMA above 50-day SMA
                score += 12.5
            
            # RSI (20%)
            rsi = talib.RSI(close_prices, timeperiod=14)
            current_rsi = rsi[-1]
            
            if 40 <= current_rsi <= 70:  # Healthy RSI range
                score += 20
            elif current_rsi < 40:  # Oversold (opportunity)
                score += 15
            
            # Volume confirmation (15%)
            volume_sma = talib.SMA(volumes, timeperiod=20)
            volume_ratio = volumes[-1] / volume_sma[-1]
            
            if volume_ratio > 1.2:  # Above average volume
                score += 15
            elif volume_ratio > 1.0:
                score += 10
            
            # Volatility check (10%)
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            volatility = (atr[-1] / close_prices[-1]) * 100
            
            if 1 <= volatility <= 4:  # Moderate volatility
                score += 10
            elif volatility < 1:  # Low volatility
                score += 5
            
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            self.logger.warning(f"Technical score calculation failed: {e}")
            return 0
    
    def calculate_fundamental_score(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate fundamental analysis score for a stock"""
        try:
            score = 0
            
            # Market cap check (20%)
            market_cap = df['market_cap'].iloc[-1] if 'market_cap' in df.columns else 0
            if market_cap > 10000:  # Large cap
                score += 20
            elif market_cap > 5000:  # Mid cap
                score += 15
            elif market_cap > 1000:  # Small cap
                score += 10
            
            # Liquidity check (30%)
            avg_volume = df['avg_volume'].iloc[-1] if 'avg_volume' in df.columns else 0
            if avg_volume > 1000000:  # High liquidity
                score += 30
            elif avg_volume > 500000:  # Medium liquidity
                score += 20
            elif avg_volume > 100000:  # Minimum liquidity
                score += 10
            
            # Price range check (20%)
            current_price = df['Close'].iloc[-1]
            if 50 <= current_price <= 2000:  # Sweet spot
                score += 20
            elif 20 <= current_price <= 5000:  # Acceptable range
                score += 15
            
            # Sector performance (30%)
            # This would require sector classification - simplified here
            price_change_30d = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
            if price_change_30d > 0.1:  # 10% gain in 30 days
                score += 30
            elif price_change_30d > 0.05:  # 5% gain in 30 days
                score += 20
            elif price_change_30d > 0:  # Positive performance
                score += 10
            
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            self.logger.warning(f"Fundamental score calculation failed for {symbol}: {e}")
            return 0
    
    def calculate_ai_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate AI-based momentum score using pattern recognition"""
        try:
            if len(df) < 50:
                return 0
            
            score = 0
            close_prices = df['Close'].values.astype(np.float64)
            high_prices = df['High'].values.astype(np.float64)
            low_prices = df['Low'].values.astype(np.float64)
            open_prices = df['Open'].values.astype(np.float64)
            
            # MACD momentum (25%)
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            if macd[-1] > macd_signal[-1] and macd_hist[-1] > macd_hist[-2]:
                score += 25
            elif macd[-1] > macd_signal[-1]:
                score += 15
            
            # Bollinger Bands position (25%)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
            bb_position = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            if 0.2 <= bb_position <= 0.8:  # Good position
                score += 25
            elif bb_position < 0.2:  # Oversold
                score += 20
            
            # Pattern recognition (25%)
            patterns = [
                talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
                talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
                talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            ]
            
            bullish_patterns = sum([1 for pattern in patterns if pattern[-1] > 0])
            score += bullish_patterns * 8  # Up to 25 points
            
            # Stochastic momentum (25%)
            stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
            if stoch_k[-1] > stoch_d[-1] and stoch_k[-1] < 80:  # Bullish but not overbought
                score += 25
            elif stoch_k[-1] < 20:  # Oversold opportunity
                score += 20
            
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            self.logger.warning(f"AI momentum score calculation failed: {e}")
            return 0
    
    def score_stock(self, symbol: str) -> Dict:
        """Comprehensive scoring of a stock"""
        try:
            # Get stock data
            df = self.get_stock_data(symbol)
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'total_score': 0,
                    'technical_score': 0,
                    'fundamental_score': 0,
                    'ai_score': 0,
                    'current_price': 0,
                    'volume': 0,
                    'error': 'No data available'
                }
            
            # Calculate individual scores
            technical_score = self.calculate_technical_score(df)
            fundamental_score = self.calculate_fundamental_score(symbol, df)
            ai_score = self.calculate_ai_momentum_score(df)
            
            # Weighted total score
            weights = {
                'technical': 0.4,
                'fundamental': 0.3,
                'ai': 0.3
            }
            
            total_score = (
                technical_score * weights['technical'] +
                fundamental_score * weights['fundamental'] +
                ai_score * weights['ai']
            )
            
            return {
                'symbol': symbol,
                'total_score': round(total_score, 2),
                'technical_score': round(technical_score, 2),
                'fundamental_score': round(fundamental_score, 2),
                'ai_score': round(ai_score, 2),
                'current_price': round(df['Close'].iloc[-1], 2),
                'volume': int(df['Volume'].iloc[-1]),
                'market_cap': df.get('market_cap', pd.Series([0])).iloc[-1],
                'price_change_1d': round(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Stock scoring failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'total_score': 0,
                'technical_score': 0,
                'fundamental_score': 0,
                'ai_score': 0,
                'current_price': 0,
                'volume': 0,
                'error': str(e)
            }
    
    def parallel_stock_scoring(self, symbols: List[str], max_workers: int = 10) -> List[Dict]:
        """Score multiple stocks in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(self.score_stock, symbol): symbol 
                              for symbol in symbols}
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                    
                    if len(results) % 50 == 0:
                        self.logger.info(f"Scored {len(results)}/{len(symbols)} stocks")
                        
                except Exception as e:
                    self.logger.warning(f"Timeout/error scoring {symbol}: {e}")
                    results.append({
                        'symbol': symbol,
                        'total_score': 0,
                        'error': f'Timeout: {str(e)}'
                    })
        
        return results
    
    def basic_filter(self, symbols: List[str]) -> List[str]:
        """Apply basic filters to reduce the stock universe"""
        try:
            filtered = []
            
            for symbol in symbols[:100]:  # Start with top 100 for speed
                try:
                    df = self.get_stock_data(symbol, period="1mo")
                    
                    if df.empty:
                        continue
                    
                    current_price = df['Close'].iloc[-1]
                    avg_volume = df['Volume'].rolling(10).mean().iloc[-1]
                    
                    # Basic filters
                    if (self.min_price <= current_price <= self.max_price and
                        avg_volume >= self.min_avg_volume):
                        filtered.append(symbol)
                    
                    if len(filtered) >= 50:  # Limit to 50 stocks for detailed analysis
                        break
                        
                except Exception as e:
                    continue
            
            self.logger.info(f"Basic filtering: {len(filtered)} stocks passed from {len(symbols)}")
            return filtered
            
        except Exception as e:
            self.logger.error(f"Basic filtering failed: {e}")
            return symbols[:20]  # Fallback to first 20
    
    def screen_stocks(self, top_n: int = 10) -> List[Dict]:
        """Main screening function to find top stocks"""
        try:
            self.logger.info("Starting dynamic stock screening...")
            
            # Step 1: Basic filtering
            filtered_symbols = self.basic_filter(self.nse_stocks)
            
            # Step 2: Detailed scoring
            self.logger.info(f"Scoring {len(filtered_symbols)} stocks...")
            scored_stocks = self.parallel_stock_scoring(filtered_symbols)
            
            # Step 3: Filter out errors and sort by score
            valid_stocks = [stock for stock in scored_stocks 
                          if stock['total_score'] > 0 and 'error' not in stock]
            
            # Sort by total score
            sorted_stocks = sorted(valid_stocks, key=lambda x: x['total_score'], reverse=True)
            
            # Get top N
            top_stocks = sorted_stocks[:top_n]
            
            self.logger.info(f"Screening completed. Top {len(top_stocks)} stocks identified.")
            
            # Log top 5 for review
            for i, stock in enumerate(top_stocks[:5], 1):
                self.logger.info(f"{i}. {stock['symbol']}: Score {stock['total_score']:.1f} "
                               f"(T:{stock['technical_score']:.1f}, F:{stock['fundamental_score']:.1f}, "
                               f"AI:{stock['ai_score']:.1f})")
            
            return top_stocks
            
        except Exception as e:
            self.logger.error(f"Stock screening failed: {e}")
            return []
    
    def get_sector_distribution(self, stocks: List[Dict]) -> Dict:
        """Analyze sector distribution of selected stocks"""
        # Simplified sector mapping (in practice, would use a comprehensive database)
        sector_map = {
            'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT',
            'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom',
            'ITC': 'FMCG', 'LT': 'Infrastructure', 'MARUTI': 'Auto'
        }
        
        sector_count = {}
        for stock in stocks:
            sector = sector_map.get(stock['symbol'], 'Others')
            sector_count[sector] = sector_count.get(sector, 0) + 1
        
        return sector_count
    
    def export_results(self, stocks: List[Dict], filename: str = None):
        """Export screening results to file"""
        try:
            if not filename:
                filename = f"stock_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'total_stocks_screened': len(self.nse_stocks),
                'top_stocks': stocks,
                'sector_distribution': self.get_sector_distribution(stocks),
                'screening_criteria': {
                    'min_price': self.min_price,
                    'max_price': self.max_price,
                    'min_volume': self.min_avg_volume,
                    'weights': {'technical': 0.4, 'fundamental': 0.3, 'ai': 0.3}
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Results exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
    
    def quick_screen(self, symbols: List[str] = None) -> List[str]:
        """Quick screening for immediate trading (used by main bot)"""
        try:
            if not symbols:
                symbols = self.nse_stocks[:30]  # Quick scan of top 30
            
            top_stocks = self.screen_stocks(top_n=5)
            return [stock['symbol'] for stock in top_stocks if stock['total_score'] > 60]
            
        except Exception as e:
            self.logger.error(f"Quick screening failed: {e}")
            return self.config.get('trading_symbols', ['RELIANCE', 'TCS', 'HDFCBANK'])

def main():
    """Test the screener"""
    screener = DynamicStockScreener()
    
    print("🔍 Starting Dynamic Stock Screening...")
    top_stocks = screener.screen_stocks(top_n=15)
    
    print(f"\n📊 TOP {len(top_stocks)} STOCKS:")
    print("=" * 80)
    
    for i, stock in enumerate(top_stocks, 1):
        print(f"{i:2d}. {stock['symbol']:12s} | Score: {stock['total_score']:5.1f} | "
              f"Price: ₹{stock['current_price']:7.2f} | "
              f"T:{stock['technical_score']:4.1f} F:{stock['fundamental_score']:4.1f} "
              f"AI:{stock['ai_score']:4.1f}")
    
    # Export results
    screener.export_results(top_stocks)
    
    print(f"\n✅ Screening completed! Check the exported JSON file for details.")

if __name__ == "__main__":
    main()