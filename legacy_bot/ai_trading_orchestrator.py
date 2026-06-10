#!/usr/bin/env python3
"""
AI Trading Orchestrator
Main coordination system that integrates all AI components and manages
the complete AI-driven trading workflow
"""

import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import threading
import time

# Import AI components
from ai_trading_engine import AITradingEngine
from dynamic_stock_screener import DynamicStockScreener
from ai_portfolio_manager import AIPortfolioManager

class AITradingOrchestrator:
    """Main orchestrator for AI trading system"""
    
    def __init__(self, config_path='config.json', kite_connection=None):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.kite = kite_connection
        
        # Initialize AI components
        self.ai_engine = AITradingEngine(config_path)
        self.stock_screener = DynamicStockScreener(config_path)
        self.portfolio_manager = AIPortfolioManager(config_path)
        
        # Trading state
        self.ai_enabled = True
        self.last_screening_time = None
        self.last_model_update = None
        self.trading_universe = []
        self.ai_predictions = {}
        self.current_market_data = {}
        
        # Performance tracking
        self.ai_trade_history = []
        self.model_performance = {}
        
        # Configuration
        self.screening_interval_hours = 6  # Re-screen stocks every 6 hours
        self.model_retrain_days = 7        # Retrain models weekly
        self.min_data_points = 100         # Minimum data for AI predictions
        
        self.logger.info("🤖 AI Trading Orchestrator initialized")
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def set_kite_connection(self, kite_connection):
        """Set the Kite connection for trading operations"""
        self.kite = kite_connection
    
    def update_trading_universe(self) -> List[str]:
        """Update the trading universe using AI stock screener"""
        try:
            self.logger.info("🔍 Updating trading universe with AI screening...")
            
            # Get top stocks from AI screener
            top_stocks = self.stock_screener.screen_stocks(top_n=20)
            
            # Extract symbols with minimum score threshold
            min_score = 60  # Minimum AI score threshold
            selected_symbols = [
                stock['symbol'] for stock in top_stocks 
                if stock.get('total_score', 0) >= min_score
            ]
            
            # Ensure we have at least some stocks (fallback to config)
            if len(selected_symbols) < 3:
                fallback_symbols = self.config.get('trading_symbols', [])
                selected_symbols.extend(fallback_symbols[:5])
                selected_symbols = list(set(selected_symbols))  # Remove duplicates
            
            self.trading_universe = selected_symbols[:15]  # Limit to 15 stocks
            self.last_screening_time = datetime.now()
            
            self.logger.info(f"✅ Trading universe updated: {len(self.trading_universe)} stocks")
            self.logger.info(f"Selected stocks: {', '.join(self.trading_universe)}")
            
            return self.trading_universe
            
        except Exception as e:
            self.logger.error(f"❌ Trading universe update failed: {e}")
            # Fallback to config symbols
            self.trading_universe = self.config.get('trading_symbols', [])[:10]
            return self.trading_universe
    
    def collect_market_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Collect market data for analysis"""
        try:
            if not symbols:
                symbols = self.trading_universe
            
            market_data = {}
            
            for symbol in symbols:
                try:
                    # Get market data (this would use your existing method)
                    if hasattr(self, 'get_market_data'):
                        df = self.get_market_data(symbol, days=60)
                    else:
                        # Placeholder - would integrate with your existing data collection
                        df = pd.DataFrame()
                    
                    if not df.empty and len(df) >= self.min_data_points:
                        market_data[symbol] = df
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect data for {symbol}: {e}")
                    continue
            
            self.current_market_data = market_data
            self.logger.info(f"📊 Market data collected for {len(market_data)} symbols")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Market data collection failed: {e}")
            return {}
    
    def train_ai_models(self, symbols: List[str] = None, force_retrain: bool = False) -> Dict:
        """Train or update AI models for selected symbols"""
        try:
            if not symbols:
                symbols = self.trading_universe
            
            training_results = {}
            
            for symbol in symbols:
                try:
                    # Check if retraining is needed
                    if not force_retrain and symbol in self.ai_engine.models:
                        # Skip if model exists and not time for retraining
                        if self.last_model_update and \
                           (datetime.now() - self.last_model_update).days < self.model_retrain_days:
                            continue
                    
                    # Get market data
                    if symbol not in self.current_market_data:
                        continue
                    
                    df = self.current_market_data[symbol]
                    
                    if len(df) < self.min_data_points:
                        self.logger.warning(f"Insufficient data for {symbol}: {len(df)} points")
                        continue
                    
                    # Train models
                    self.logger.info(f"🧠 Training AI models for {symbol}...")
                    success = self.ai_engine.retrain_models(symbol, df)
                    
                    if success:
                        training_results[symbol] = {'status': 'success', 'data_points': len(df)}
                        self.logger.info(f"✅ Models trained for {symbol}")
                    else:
                        training_results[symbol] = {'status': 'failed', 'reason': 'Training failed'}
                        
                except Exception as e:
                    self.logger.error(f"Model training failed for {symbol}: {e}")
                    training_results[symbol] = {'status': 'error', 'reason': str(e)}
            
            self.last_model_update = datetime.now()
            self.logger.info(f"🎓 AI model training completed: {len(training_results)} symbols processed")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ AI model training failed: {e}")
            return {}
    
    def generate_ai_predictions(self, symbols: List[str] = None) -> Dict:
        """Generate AI predictions for symbols"""
        try:
            if not symbols:
                symbols = self.trading_universe
            
            predictions = {}
            
            for symbol in symbols:
                try:
                    if symbol not in self.current_market_data:
                        continue
                    
                    df = self.current_market_data[symbol]
                    
                    if len(df) < 50:  # Minimum data for prediction
                        continue
                    
                    # Generate AI prediction
                    prediction = self.ai_engine.predict(symbol, df)
                    
                    if prediction.get('confidence', 0) > 0:
                        predictions[symbol] = prediction
                        
                        self.logger.debug(f"🔮 AI prediction for {symbol}: "
                                        f"Signal={prediction.get('prediction', 0)}, "
                                        f"Confidence={prediction.get('confidence', 0):.1f}%")
                
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {symbol}: {e}")
                    continue
            
            self.ai_predictions = predictions
            self.logger.info(f"🎯 AI predictions generated for {len(predictions)} symbols")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ AI prediction generation failed: {e}")
            return {}
    
    def analyze_trading_opportunities(self) -> List[Dict]:
        """Analyze and rank trading opportunities using AI"""
        try:
            opportunities = []
            
            for symbol, prediction in self.ai_predictions.items():
                try:
                    if symbol not in self.current_market_data:
                        continue
                    
                    df = self.current_market_data[symbol]
                    current_price = df['close'].iloc[-1]
                    
                    # Only consider high-confidence predictions
                    confidence = prediction.get('confidence', 0)
                    if confidence < self.portfolio_manager.confidence_threshold:
                        continue
                    
                    # Get AI prediction signal
                    ai_signal = prediction.get('prediction', 0)
                    
                    if ai_signal == 1:  # Buy signal
                        # Check portfolio constraints
                        if not self.portfolio_manager.check_correlation_risk(symbol, self.current_market_data):
                            self.logger.info(f"⚠️ Correlation risk too high for {symbol}")
                            continue
                        
                        if not self.portfolio_manager.check_sector_exposure(symbol):
                            self.logger.info(f"⚠️ Sector exposure limit for {symbol}")
                            continue
                        
                        # Calculate position size
                        position_size = self.portfolio_manager.calculate_ai_position_size(
                            symbol, prediction, current_price, self.current_market_data
                        )
                        
                        if position_size > 0:
                            opportunity = {
                                'symbol': symbol,
                                'action': 'BUY',
                                'quantity': position_size,
                                'current_price': current_price,
                                'ai_confidence': confidence,
                                'ai_prediction': prediction,
                                'expected_investment': position_size * current_price,
                                'stop_loss': self.portfolio_manager.calculate_stop_loss_level(
                                    symbol, current_price, confidence
                                ),
                                'take_profit': self.portfolio_manager.calculate_take_profit_level(
                                    symbol, current_price, confidence
                                ),
                                'priority_score': confidence * prediction.get('agreement', 1.0)
                            }
                            
                            opportunities.append(opportunity)
                
                except Exception as e:
                    self.logger.warning(f"Opportunity analysis failed for {symbol}: {e}")
                    continue
            
            # Sort by priority score (confidence * agreement)
            opportunities.sort(key=lambda x: x['priority_score'], reverse=True)
            
            self.logger.info(f"📈 Found {len(opportunities)} AI trading opportunities")
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Trading opportunity analysis failed: {e}")
            return []
    
    def check_exit_signals(self) -> List[Dict]:
        """Check for exit signals on current positions"""
        try:
            exit_signals = []
            
            # Get current positions from portfolio manager
            for symbol in self.portfolio_manager.current_positions:
                position = self.portfolio_manager.current_positions[symbol]
                
                if position.get('quantity', 0) == 0:
                    continue
                
                # Get current price
                current_price = 0
                if symbol in self.current_market_data and not self.current_market_data[symbol].empty:
                    current_price = self.current_market_data[symbol]['close'].iloc[-1]
                
                if current_price == 0:
                    continue
                
                # Get AI prediction for exit decision
                ai_prediction = self.ai_predictions.get(symbol, {})
                
                # Check exit conditions
                exit_decision = self.portfolio_manager.should_exit_position(
                    symbol, current_price, ai_prediction
                )
                
                if exit_decision['should_exit']:
                    exit_signal = {
                        'symbol': symbol,
                        'action': 'SELL' if position['quantity'] > 0 else 'BUY',
                        'quantity': abs(position['quantity']),
                        'current_price': current_price,
                        'reason': exit_decision['reason'],
                        'urgency': exit_decision.get('urgency', 'medium'),
                        'ai_confidence': ai_prediction.get('confidence', 0)
                    }
                    
                    exit_signals.append(exit_signal)
            
            if exit_signals:
                self.logger.info(f"🚪 Found {len(exit_signals)} exit signals")
            
            return exit_signals
            
        except Exception as e:
            self.logger.error(f"❌ Exit signal check failed: {e}")
            return []
    
    def execute_ai_strategy(self) -> Dict:
        """Execute complete AI trading strategy"""
        try:
            execution_summary = {
                'timestamp': datetime.now().isoformat(),
                'screening_performed': False,
                'models_trained': False,
                'predictions_generated': False,
                'opportunities_found': 0,
                'exit_signals_found': 0,
                'actions_recommended': []
            }
            
            # Step 1: Update trading universe (if needed)
            if (not self.last_screening_time or 
                (datetime.now() - self.last_screening_time).total_seconds() > 
                self.screening_interval_hours * 3600):
                
                self.update_trading_universe()
                execution_summary['screening_performed'] = True
            
            # Step 2: Collect market data
            self.collect_market_data()
            
            # Step 3: Train/update AI models (if needed)
            if (not self.last_model_update or 
                (datetime.now() - self.last_model_update).days >= self.model_retrain_days):
                
                training_results = self.train_ai_models()
                execution_summary['models_trained'] = len(training_results) > 0
            
            # Step 4: Generate AI predictions
            predictions = self.generate_ai_predictions()
            execution_summary['predictions_generated'] = len(predictions) > 0
            
            # Step 5: Check exit signals first
            exit_signals = self.check_exit_signals()
            execution_summary['exit_signals_found'] = len(exit_signals)
            execution_summary['actions_recommended'].extend(exit_signals)
            
            # Step 6: Find new opportunities
            opportunities = self.analyze_trading_opportunities()
            execution_summary['opportunities_found'] = len(opportunities)
            execution_summary['actions_recommended'].extend(opportunities)
            
            # Step 7: Generate summary report
            self.logger.info("🎯 AI Strategy Execution Summary:")
            self.logger.info(f"   Screening: {'✅' if execution_summary['screening_performed'] else '⏭️'}")
            self.logger.info(f"   Models: {'✅' if execution_summary['models_trained'] else '⏭️'}")
            self.logger.info(f"   Predictions: {len(predictions)} generated")
            self.logger.info(f"   Exit Signals: {len(exit_signals)}")
            self.logger.info(f"   New Opportunities: {len(opportunities)}")
            
            return execution_summary
            
        except Exception as e:
            self.logger.error(f"❌ AI strategy execution failed: {e}")
            return {'error': str(e)}
    
    def get_ai_performance_report(self) -> Dict:
        """Generate AI system performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'ai_system_status': {
                    'models_loaded': len(self.ai_engine.models),
                    'trading_universe_size': len(self.trading_universe),
                    'active_predictions': len(self.ai_predictions),
                    'last_screening': self.last_screening_time.isoformat() if self.last_screening_time else None,
                    'last_model_update': self.last_model_update.isoformat() if self.last_model_update else None
                },
                'model_performance': {},
                'recent_ai_trades': self.ai_trade_history[-10:],  # Last 10 trades
                'current_universe': self.trading_universe,
                'active_predictions': {}
            }
            
            # Model performance metrics
            for symbol in self.ai_engine.models:
                performance = self.ai_engine.get_model_performance(symbol)
                if performance:
                    report['model_performance'][symbol] = performance
            
            # Current predictions summary
            for symbol, prediction in self.ai_predictions.items():
                report['active_predictions'][symbol] = {
                    'signal': prediction.get('prediction', 0),
                    'confidence': prediction.get('confidence', 0),
                    'agreement': prediction.get('agreement', 0)
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ AI performance report generation failed: {e}")
            return {}
    
    def record_ai_trade(self, trade_info: Dict):
        """Record AI trade for performance tracking"""
        try:
            ai_trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_info.get('symbol'),
                'action': trade_info.get('action'),
                'quantity': trade_info.get('quantity'),
                'price': trade_info.get('price'),
                'ai_confidence': trade_info.get('ai_confidence'),
                'ai_prediction': trade_info.get('ai_prediction'),
                'trade_id': trade_info.get('trade_id')
            }
            
            self.ai_trade_history.append(ai_trade_record)
            
            # Keep only last 100 trades
            if len(self.ai_trade_history) > 100:
                self.ai_trade_history = self.ai_trade_history[-100:]
            
            self.logger.info(f"📝 AI trade recorded: {trade_info.get('action')} "
                           f"{trade_info.get('quantity')} {trade_info.get('symbol')}")
            
        except Exception as e:
            self.logger.error(f"❌ AI trade recording failed: {e}")
    
    def get_trading_signals(self) -> Dict:
        """Get current trading signals for integration with main bot"""
        try:
            # Execute AI strategy to get latest signals
            execution_result = self.execute_ai_strategy()
            
            # Format signals for main trading bot
            signals = {
                'buy_signals': [],
                'sell_signals': [],
                'ai_enabled': self.ai_enabled,
                'execution_summary': execution_result
            }
            
            for action in execution_result.get('actions_recommended', []):
                if action['action'] == 'BUY':
                    signals['buy_signals'].append({
                        'symbol': action['symbol'],
                        'quantity': action['quantity'],
                        'confidence': action.get('ai_confidence', 0),
                        'reason': 'AI prediction',
                        'stop_loss': action.get('stop_loss'),
                        'take_profit': action.get('take_profit')
                    })
                elif action['action'] == 'SELL':
                    signals['sell_signals'].append({
                        'symbol': action['symbol'],
                        'quantity': action['quantity'],
                        'reason': action.get('reason', 'AI exit signal'),
                        'urgency': action.get('urgency', 'medium')
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Trading signals generation failed: {e}")
            return {'buy_signals': [], 'sell_signals': [], 'ai_enabled': False}
    
    def toggle_ai_system(self, enabled: bool):
        """Enable or disable AI system"""
        self.ai_enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"🤖 AI trading system {status}")
    
    def emergency_stop(self):
        """Emergency stop of AI system"""
        self.ai_enabled = False
        self.logger.warning("🛑 AI system emergency stop activated")
        
        # Clear current signals
        self.ai_predictions = {}
        self.trading_universe = []