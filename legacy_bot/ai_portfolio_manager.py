#!/usr/bin/env python3
"""
AI Portfolio Manager
Intelligent portfolio management with AI-driven position sizing, risk management,
and portfolio optimization using machine learning techniques
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

class AIPortfolioManager:
    """AI-powered portfolio management system"""
    
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.current_positions = {}
        self.portfolio_value = 0
        self.available_cash = 0
        self.daily_pnl = 0
        
        # Risk parameters from config
        self.max_position_size = self.config['capital']['max_position_size']
        self.max_positions = self.config['capital']['max_positions']
        self.daily_loss_limit = self.config['capital']['daily_loss_limit']
        self.total_capital = self.config['capital']['total_capital']
        
        # AI-specific parameters
        self.confidence_threshold = 65  # Minimum AI confidence for trading
        self.correlation_limit = 0.7   # Maximum correlation between positions
        self.sector_limit = 0.4        # Maximum sector exposure
        self.volatility_target = 0.15  # Target portfolio volatility
        
        # Portfolio metrics
        self.portfolio_metrics = {}
        self.correlation_matrix = pd.DataFrame()
        self.sector_exposure = {}
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def update_portfolio_state(self, positions: Dict, available_cash: float):
        """Update current portfolio state"""
        try:
            self.current_positions = positions
            self.available_cash = available_cash
            
            # Calculate portfolio value
            self.portfolio_value = sum(
                pos.get('quantity', 0) * pos.get('last_price', 0) 
                for pos in positions.values()
            ) + available_cash
            
            # Calculate daily P&L
            self.daily_pnl = sum(pos.get('pnl', 0) for pos in positions.values())
            
            self.logger.info(f"Portfolio updated: Value ₹{self.portfolio_value:,.2f}, "
                           f"Cash ₹{available_cash:,.2f}, P&L ₹{self.daily_pnl:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Portfolio state update failed: {e}")
    
    def calculate_position_correlation(self, symbols: List[str], 
                                     price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix between potential positions"""
        try:
            if len(symbols) < 2:
                return pd.DataFrame()
            
            # Create returns dataframe
            returns_data = {}
            
            for symbol in symbols:
                if symbol in price_data and not price_data[symbol].empty:
                    df = price_data[symbol]
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 20:  # Minimum data requirement
                        returns_data[symbol] = returns.tail(50)  # Last 50 days
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # Align data
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:  # Minimum observations
                return pd.DataFrame()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_volatility(self, weights: np.array, 
                                     returns_df: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        try:
            # Calculate covariance matrix using Ledoit-Wolf shrinkage
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_df).covariance_
            
            # Calculate portfolio volatility
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
            
            return portfolio_volatility
            
        except Exception as e:
            self.logger.error(f"Portfolio volatility calculation failed: {e}")
            return 0.0
    
    def optimize_portfolio_weights(self, expected_returns: pd.Series, 
                                 returns_df: pd.DataFrame,
                                 target_volatility: float = None) -> np.array:
        """Optimize portfolio weights using mean-variance optimization"""
        try:
            n_assets = len(expected_returns)
            
            if n_assets == 1:
                return np.array([1.0])
            
            # Target volatility (default to class parameter)
            if target_volatility is None:
                target_volatility = self.volatility_target
            
            # Objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_vol = self.calculate_portfolio_volatility(weights, returns_df)
                
                if portfolio_vol == 0:
                    return -np.inf
                
                # Sharpe ratio (assuming risk-free rate of 6%)
                sharpe_ratio = (portfolio_return - 0.06) / portfolio_vol
                return -sharpe_ratio  # Minimize negative Sharpe
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Add volatility constraint if specified
            if target_volatility:
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x: target_volatility - self.calculate_portfolio_volatility(x, returns_df)
                })
            
            # Bounds for individual weights (max 30% per position)
            bounds = tuple((0, 0.3) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = sco.minimize(
                objective, 
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                self.logger.warning("Portfolio optimization failed, using equal weights")
                return np.array([1/n_assets] * n_assets)
                
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            # Return equal weights as fallback
            return np.array([1/len(expected_returns)] * len(expected_returns))
    
    def calculate_ai_position_size(self, symbol: str, ai_prediction: Dict, 
                                 current_price: float, price_data: Dict = None) -> int:
        """Calculate optimal position size using AI confidence and risk management"""
        try:
            # Base position size from traditional method
            base_allocation = self.total_capital * self.max_position_size
            
            # AI confidence adjustment
            confidence = ai_prediction.get('confidence', 0)
            
            if confidence < self.confidence_threshold:
                self.logger.info(f"AI confidence {confidence}% below threshold {self.confidence_threshold}% for {symbol}")
                return 0
            
            # Confidence-based scaling (65% confidence = 0.5x, 95% = 1.5x)
            confidence_factor = max(0, (confidence - 50) / 50)  # 0 to 0.9 range
            confidence_multiplier = 0.5 + confidence_factor  # 0.5x to 1.4x range
            
            # Model agreement adjustment
            agreement = ai_prediction.get('agreement', 1.0)
            agreement_multiplier = 0.7 + (agreement * 0.3)  # 0.7x to 1.0x
            
            # Risk-adjusted position size
            risk_adjusted_allocation = base_allocation * confidence_multiplier * agreement_multiplier
            
            # Portfolio-level constraints
            current_position_count = len([pos for pos in self.current_positions.values() 
                                        if pos.get('quantity', 0) != 0])
            
            if current_position_count >= self.max_positions:
                self.logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return 0
            
            # Check available cash
            max_available = self.available_cash * 0.9  # Keep 10% buffer
            final_allocation = min(risk_adjusted_allocation, max_available)
            
            # Calculate quantity
            quantity = int(final_allocation / current_price)
            
            # Minimum trade size check
            min_trade_amount = self.config['capital']['min_trade_amount']
            if quantity * current_price < min_trade_amount:
                return 0
            
            self.logger.info(f"AI position sizing for {symbol}: Confidence {confidence}%, "
                           f"Allocation ₹{final_allocation:,.0f}, Quantity {quantity}")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"AI position sizing failed for {symbol}: {e}")
            return 0
    
    def check_correlation_risk(self, new_symbol: str, price_data: Dict) -> bool:
        """Check if adding new position would violate correlation limits"""
        try:
            current_symbols = [symbol for symbol, pos in self.current_positions.items() 
                             if pos.get('quantity', 0) != 0]
            
            if not current_symbols:
                return True  # No existing positions
            
            all_symbols = current_symbols + [new_symbol]
            
            # Calculate correlation matrix
            correlation_matrix = self.calculate_position_correlation(all_symbols, price_data)
            
            if correlation_matrix.empty:
                return True  # Cannot calculate, allow trade
            
            # Check correlation with existing positions
            if new_symbol in correlation_matrix.columns:
                correlations = correlation_matrix[new_symbol].drop(new_symbol)
                max_correlation = correlations.abs().max()
                
                if max_correlation > self.correlation_limit:
                    self.logger.warning(f"High correlation risk for {new_symbol}: "
                                      f"Max correlation {max_correlation:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Correlation check failed for {new_symbol}: {e}")
            return True  # Allow trade if check fails
    
    def check_sector_exposure(self, new_symbol: str) -> bool:
        """Check if adding new position would violate sector exposure limits"""
        try:
            # Simplified sector mapping (would be more comprehensive in production)
            sector_map = {
                'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT',
                'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'KOTAKBANK': 'Banking',
                'BHARTIARTL': 'Telecom', 'ITC': 'FMCG', 'LT': 'Infrastructure',
                'MARUTI': 'Auto', 'HINDUNILVR': 'FMCG', 'ASIANPAINT': 'Chemical'
            }
            
            new_sector = sector_map.get(new_symbol, 'Others')
            
            # Calculate current sector exposure
            sector_exposure = {}
            total_value = 0
            
            for symbol, pos in self.current_positions.items():
                if pos.get('quantity', 0) != 0:
                    sector = sector_map.get(symbol, 'Others')
                    position_value = pos.get('quantity', 0) * pos.get('last_price', 0)
                    sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
                    total_value += position_value
            
            if total_value > 0:
                current_sector_pct = sector_exposure.get(new_sector, 0) / total_value
                
                if current_sector_pct > self.sector_limit:
                    self.logger.warning(f"Sector exposure limit exceeded for {new_sector}: "
                                      f"{current_sector_pct:.1%}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sector exposure check failed for {new_symbol}: {e}")
            return True  # Allow trade if check fails
    
    def calculate_stop_loss_level(self, symbol: str, entry_price: float, 
                                 ai_confidence: float, volatility: float = None) -> float:
        """Calculate dynamic stop-loss level based on AI confidence and volatility"""
        try:
            # Base stop-loss from config
            base_stop_loss = self.config['capital']['stop_loss_pct']
            
            # Adjust based on AI confidence
            # High confidence = tighter stop loss, Low confidence = wider stop loss
            confidence_factor = ai_confidence / 100
            confidence_adjustment = 0.8 + (confidence_factor * 0.4)  # 0.8 to 1.2
            
            # Adjust based on volatility if available
            volatility_adjustment = 1.0
            if volatility:
                # Higher volatility = wider stop loss
                volatility_adjustment = max(0.8, min(1.5, 1 + (volatility - 0.02) * 10))
            
            # Calculate final stop loss
            adjusted_stop_loss = base_stop_loss * confidence_adjustment * volatility_adjustment
            
            # Cap at reasonable limits
            adjusted_stop_loss = max(0.02, min(0.1, adjusted_stop_loss))  # 2% to 10%
            
            stop_loss_price = entry_price * (1 - adjusted_stop_loss)
            
            self.logger.info(f"Dynamic stop-loss for {symbol}: {adjusted_stop_loss:.1%} "
                           f"(₹{stop_loss_price:.2f})")
            
            return stop_loss_price
            
        except Exception as e:
            self.logger.error(f"Stop-loss calculation failed for {symbol}: {e}")
            return entry_price * (1 - base_stop_loss)
    
    def calculate_take_profit_level(self, symbol: str, entry_price: float, 
                                   ai_confidence: float) -> float:
        """Calculate dynamic take-profit level based on AI confidence"""
        try:
            # Base take-profit from config
            base_take_profit = self.config['capital']['take_profit_pct']
            
            # Adjust based on AI confidence
            # High confidence = higher take profit target
            confidence_factor = ai_confidence / 100
            confidence_multiplier = 0.8 + (confidence_factor * 0.7)  # 0.8 to 1.5
            
            # Calculate final take profit
            adjusted_take_profit = base_take_profit * confidence_multiplier
            
            # Cap at reasonable limits
            adjusted_take_profit = max(0.05, min(0.25, adjusted_take_profit))  # 5% to 25%
            
            take_profit_price = entry_price * (1 + adjusted_take_profit)
            
            self.logger.info(f"Dynamic take-profit for {symbol}: {adjusted_take_profit:.1%} "
                           f"(₹{take_profit_price:.2f})")
            
            return take_profit_price
            
        except Exception as e:
            self.logger.error(f"Take-profit calculation failed for {symbol}: {e}")
            return entry_price * (1 + base_take_profit)
    
    def should_exit_position(self, symbol: str, current_price: float, 
                           ai_prediction: Dict = None) -> Dict:
        """Determine if a position should be exited based on AI and risk factors"""
        try:
            if symbol not in self.current_positions:
                return {'should_exit': False, 'reason': 'No position found'}
            
            position = self.current_positions[symbol]
            quantity = position.get('quantity', 0)
            
            if quantity == 0:
                return {'should_exit': False, 'reason': 'No quantity'}
            
            entry_price = position.get('average_price', current_price)
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Check traditional stop-loss and take-profit
            stop_loss_pct = self.config['capital']['stop_loss_pct']
            take_profit_pct = self.config['capital']['take_profit_pct']
            
            if quantity > 0:  # Long position
                if pnl_pct <= -stop_loss_pct:
                    return {'should_exit': True, 'reason': 'Stop-loss triggered', 'urgency': 'high'}
                if pnl_pct >= take_profit_pct:
                    return {'should_exit': True, 'reason': 'Take-profit triggered', 'urgency': 'medium'}
            else:  # Short position
                if pnl_pct >= stop_loss_pct:
                    return {'should_exit': True, 'reason': 'Stop-loss triggered', 'urgency': 'high'}
                if pnl_pct <= -take_profit_pct:
                    return {'should_exit': True, 'reason': 'Take-profit triggered', 'urgency': 'medium'}
            
            # AI-based exit signals
            if ai_prediction:
                prediction = ai_prediction.get('prediction', 0)
                confidence = ai_prediction.get('confidence', 0)
                
                # Exit if AI confidence is very low
                if confidence < 40:
                    return {'should_exit': True, 'reason': 'Low AI confidence', 'urgency': 'medium'}
                
                # Exit if AI prediction changed direction with high confidence
                if quantity > 0 and prediction == 0 and confidence > 70:
                    return {'should_exit': True, 'reason': 'AI signal reversal', 'urgency': 'medium'}
                elif quantity < 0 and prediction == 1 and confidence > 70:
                    return {'should_exit': True, 'reason': 'AI signal reversal', 'urgency': 'medium'}
            
            # Time-based exit (position timeout)
            position_age_days = 5  # Would calculate from entry date in practice
            max_hold_days = self.config.get('risk_management', {}).get('position_timeout_days', 5)
            
            if position_age_days >= max_hold_days:
                return {'should_exit': True, 'reason': 'Position timeout', 'urgency': 'low'}
            
            return {'should_exit': False, 'reason': 'All checks passed'}
            
        except Exception as e:
            self.logger.error(f"Exit check failed for {symbol}: {e}")
            return {'should_exit': False, 'reason': f'Error: {str(e)}'}
    
    def generate_portfolio_report(self) -> Dict:
        """Generate comprehensive portfolio performance report"""
        try:
            # Basic metrics
            total_positions = len([pos for pos in self.current_positions.values() 
                                 if pos.get('quantity', 0) != 0])
            
            total_invested = sum(
                abs(pos.get('quantity', 0)) * pos.get('average_price', 0)
                for pos in self.current_positions.values()
                if pos.get('quantity', 0) != 0
            )
            
            # Calculate returns
            total_pnl = sum(pos.get('pnl', 0) for pos in self.current_positions.values())
            portfolio_return = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            
            # Risk metrics
            cash_utilization = ((total_invested / self.total_capital) * 100) if self.total_capital > 0 else 0
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'available_cash': self.available_cash,
                'total_invested': total_invested,
                'cash_utilization': round(cash_utilization, 2),
                'total_positions': total_positions,
                'daily_pnl': self.daily_pnl,
                'portfolio_return_pct': round(portfolio_return, 2),
                'positions': []
            }
            
            # Position details
            for symbol, pos in self.current_positions.items():
                if pos.get('quantity', 0) != 0:
                    position_value = abs(pos.get('quantity', 0)) * pos.get('last_price', 0)
                    weight = (position_value / total_invested * 100) if total_invested > 0 else 0
                    
                    report['positions'].append({
                        'symbol': symbol,
                        'quantity': pos.get('quantity', 0),
                        'average_price': pos.get('average_price', 0),
                        'current_price': pos.get('last_price', 0),
                        'pnl': pos.get('pnl', 0),
                        'weight_pct': round(weight, 2)
                    })
            
            # Risk alerts
            risk_alerts = []
            
            if cash_utilization > 90:
                risk_alerts.append("High cash utilization")
            
            if total_positions >= self.max_positions:
                risk_alerts.append("Maximum positions reached")
            
            if abs(self.daily_pnl) > (self.total_capital * self.daily_loss_limit):
                risk_alerts.append("Daily loss limit approached")
            
            report['risk_alerts'] = risk_alerts
            
            return report
            
        except Exception as e:
            self.logger.error(f"Portfolio report generation failed: {e}")
            return {}
    
    def rebalance_portfolio(self, ai_predictions: Dict, price_data: Dict) -> List[Dict]:
        """Suggest portfolio rebalancing based on AI predictions and risk metrics"""
        try:
            rebalance_actions = []
            
            # Check each position for rebalancing
            for symbol in self.current_positions:
                position = self.current_positions[symbol]
                quantity = position.get('quantity', 0)
                
                if quantity == 0:
                    continue
                
                current_price = position.get('last_price', 0)
                
                # Check AI prediction for this symbol
                ai_pred = ai_predictions.get(symbol, {})
                
                # Get exit recommendation
                exit_check = self.should_exit_position(symbol, current_price, ai_pred)
                
                if exit_check['should_exit']:
                    rebalance_actions.append({
                        'action': 'EXIT',
                        'symbol': symbol,
                        'quantity': abs(quantity),
                        'reason': exit_check['reason'],
                        'urgency': exit_check.get('urgency', 'medium'),
                        'current_price': current_price
                    })
            
            return rebalance_actions
            
        except Exception as e:
            self.logger.error(f"Portfolio rebalancing failed: {e}")
            return []