import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1PredictionModel:
    """Comprehensive F1 prediction system with ML, Monte Carlo simulations and track analysis."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.qualifying_model = None
        self.race_model = None
        self.feature_importance = {}
        self.feature_analysis = {}
        self.sector_data = {}
        self.dnf_model = None
        self.feature_names = []
        self.track_stats = None
        
    def collect_data(self, year: int, races: int = 5) -> pd.DataFrame:
        """Collect comprehensive F1 data using fastf1"""
        logger.info(f"Collecting data for {year} season, last {races} races")
        
        fastf1.Cache.enable_cache('cache')
        all_data = []
        
        schedule = fastf1.get_event_schedule(year)
        completed_events = schedule[schedule['EventDate'] < datetime.now()]
        
        # Get the most recent completed races for training
        recent_events = completed_events.tail(races)
        
        # Store the schedule for predicting next race
        self.schedule = schedule
        self.completed_events = completed_events
        
        for _, event in recent_events.iterrows():
            try:
                session = fastf1.get_session(year, event['EventName'], 'R')
                session.load()
                
                # Collect session data
                session_data = self._process_session_data(session, event['EventName'])
                all_data.extend(session_data)
                
            except Exception as e:
                logger.warning(f"Error processing {event['EventName']}: {e}")
                continue
                
        return pd.DataFrame(all_data)
    
    def _process_session_data(self, session, event_name: str) -> List[Dict]:
        """Process individual session data"""
        data = []
        results = session.results
        
        for _, driver_result in results.iterrows():
            driver = driver_result['Abbreviation']
            
            try:
                driver_laps = session.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                # Calculate various metrics
                metrics = {
                    'Event': event_name,
                    'Driver': driver,
                    'Team': driver_result['TeamName'],
                    'GridPosition': driver_result['GridPosition'],
                    'Position': driver_result['Position'],
                    'Points': driver_result['Points'],
                    'Status': driver_result['Status'],
                    'DNF': 1 if driver_result['Status'] != 'Finished' else 0
                }
                
                # Add lap time statistics
                valid_laps = driver_laps[driver_laps['LapTime'].notna()]
                if not valid_laps.empty:
                    lap_times = valid_laps['LapTime']
                    # Convert to seconds if timedelta
                    if hasattr(lap_times.iloc[0], 'total_seconds'):
                        avg_time = lap_times.mean().total_seconds()
                        best_time = lap_times.min().total_seconds()
                        std_time = lap_times.std().total_seconds()
                    else:
                        avg_time = float(lap_times.mean())
                        best_time = float(lap_times.min())
                        std_time = float(lap_times.std())
                    
                    metrics.update({
                        'AvgLapTime': avg_time,
                        'BestLapTime': best_time,
                        'LapTimeStd': std_time,
                        'TotalLaps': len(valid_laps),
                        'ConsistencyScore': 1 / (1 + std_time)
                    })
                
                # Sector analysis
                sector_times = self._analyze_sectors(driver_laps)
                metrics.update(sector_times)
                
                data.append(metrics)
                
            except Exception as e:
                logger.debug(f"Error processing driver {driver}: {e}")
                continue
                
        return data
    
    def _analyze_sectors(self, laps: pd.DataFrame) -> Dict:
        """Analyze sector times and calculate theoretical best lap"""
        sector_data = {}
        
        try:
            # Get sector times
            for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                if sector in laps.columns:
                    valid_sectors = laps[laps[sector].notna()][sector]
                    if not valid_sectors.empty:
                        # Convert to seconds if timedelta
                        if hasattr(valid_sectors.iloc[0], 'total_seconds'):
                            sector_data[f'Best{sector}'] = valid_sectors.min().total_seconds()
                            sector_data[f'Avg{sector}'] = valid_sectors.mean().total_seconds()
                        else:
                            sector_data[f'Best{sector}'] = float(valid_sectors.min())
                            sector_data[f'Avg{sector}'] = float(valid_sectors.mean())
                        
            # Calculate theoretical best lap
            if all(f'Best{s}Time' in sector_data for s in ['Sector1', 'Sector2', 'Sector3']):
                sector_data['TheoreticalBest'] = sum(
                    sector_data[f'Best{s}Time'] for s in ['Sector1', 'Sector2', 'Sector3']
                )
                
                # Achievability factor
                if 'LapTime' in laps.columns:
                    valid_laps = laps[laps['LapTime'].notna()]
                    if not valid_laps.empty:
                        best_lap = valid_laps['LapTime'].min()
                        if hasattr(best_lap, 'total_seconds'):
                            best_actual = best_lap.total_seconds()
                        else:
                            best_actual = float(best_lap)
                        
                        if sector_data['TheoreticalBest'] > 0:
                            sector_data['AchievabilityFactor'] = best_actual / sector_data['TheoreticalBest']
                    
        except Exception as e:
            logger.debug(f"Sector analysis error: {e}")
            
        return sector_data
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML models"""
        logger.info("Engineering features...")
        
        # Grid to position difference
        df['GridPositionDelta'] = df['GridPosition'] - df['Position']
        
        # Performance volatility
        df['PerformanceVolatility'] = df.groupby('Driver')['Position'].transform('std')
        
        # Recent form (weighted average of last 3 races)
        df = df.sort_values(['Driver', 'Event'])
        df['RecentForm'] = df.groupby('Driver')['Points'].transform(
            lambda x: x.rolling(3, min_periods=1).apply(
                lambda y: np.average(y, weights=np.linspace(0.5, 1, len(y)))
            )
        )
        
        # DNF risk
        df['DNFRate'] = df.groupby('Driver')['DNF'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Team performance
        df['TeamAvgPosition'] = df.groupby(['Team', 'Event'])['Position'].transform('mean')
        
        # Consistency metrics
        if 'LapTimeStd' in df.columns:
            df['ConsistencyRank'] = df.groupby('Event')['LapTimeStd'].rank(ascending=True)
        
        # Sector performance
        for sector in ['Sector1', 'Sector2', 'Sector3']:
            if f'Best{sector}Time' in df.columns:
                df[f'{sector}Rank'] = df.groupby('Event')[f'Best{sector}Time'].rank(ascending=True)
        
        # Momentum tracking
        df['PointsMomentum'] = df.groupby('Driver')['Points'].transform(
            lambda x: x.diff().rolling(3, min_periods=1).mean()
        )
        
        # Track characteristics impact
        df['OvertakingDifficulty'] = self._calculate_overtaking_difficulty(df)
        
        return df
    
    def _calculate_overtaking_difficulty(self, df: pd.DataFrame) -> pd.Series:
        """Calculate track-specific overtaking difficulty"""
        # Simple heuristic based on position changes
        track_overtaking = df.groupby('Event')['GridPositionDelta'].agg(['mean', 'std'])
        track_overtaking['difficulty'] = 1 / (1 + track_overtaking['std'])
        
        return df['Event'].map(track_overtaking['difficulty'])
    
    def build_ensemble_models(self, df: pd.DataFrame):
        """Build ensemble ML models for predictions.

        The ensemble combines RandomForest, GradientBoosting, XGBoost and Ridge
<<<<<<< HEAD
        regression models to capture diverse patterns in the data.
=======
        regression models to capture diverse patterns in the data. During
        training, feature correlations and permutation importance are computed
        to better understand which inputs drive the predictions.
>>>>>>> rz4tq4-codex/improve-machine-learning-model
        """
        logger.info("Building ensemble models...")
        
        # Prepare features
        feature_cols = [
            'GridPosition', 'ConsistencyScore', 'RecentForm', 'DNFRate',
            'TeamAvgPosition', 'PerformanceVolatility', 'PointsMomentum',
            'OvertakingDifficulty'
        ]
        
        # Add sector features if available
        sector_features = [col for col in df.columns if 'Sector' in col and 'Rank' in col]
        feature_cols.extend(sector_features)
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features  # Store for later use
        
        # Prepare data - only use rows with valid positions
        valid_data = df[df['Position'].notna() & (df['Position'] > 0) & (df['Position'] <= 20)].copy()
        
        if len(valid_data) < 20:
            logger.warning(f"Limited valid data for training: {len(valid_data)} samples")
        
        X = valid_data[available_features].fillna(valid_data[available_features].mean())
        y_race = valid_data['Position'].astype(int)
        y_dnf = valid_data['DNF'].astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build race position model with an additional XGBoost regressor
        self.race_model = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05,
                                 max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, objective='reg:squarederror')),
            ('ridge', Ridge(alpha=1.0))
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(self.race_model, X_scaled, y_race, 
                                   cv=KFold(n_splits=min(5, len(valid_data)//10), 
                                           shuffle=True, random_state=42),
                                   scoring='neg_mean_absolute_error')
        
        logger.info(f"Cross-validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Fit models
        self.race_model.fit(X_scaled, y_race)
        
        # DNF prediction model
        from sklearn.ensemble import RandomForestClassifier
        self.dnf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.dnf_model.fit(X_scaled, y_dnf)
        
        # Analyze feature correlations
        correlations = X.corrwith(y_race).abs().sort_values(ascending=False)
        self.feature_analysis['correlation'] = correlations.to_dict()
        logger.info("Top correlated features: %s", correlations.head(5).to_dict())

        # Feature importance using permutation importance
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(self.race_model, X_scaled, y_race,
                                      n_repeats=10, random_state=42)
        self.feature_importance = dict(zip(
            available_features,
            perm.importances_mean
        ))
    
    def predict_qualifying(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Predict qualifying results with sector analysis"""
        predictions = []
        
        # First, collect all theoretical best times to calculate relative performance
        all_theoretical_bests = {}
        for driver in current_data['Driver'].unique():
            driver_data = current_data[current_data['Driver'] == driver]
            
            # Calculate theoretical best lap
            theoretical_best = 0
            for i in [1, 2, 3]:
                if f'BestSector{i}Time' in driver_data.columns:
                    sector_time = driver_data[f'BestSector{i}Time'].min()
                    if pd.notna(sector_time):
                        theoretical_best += sector_time
            
            if theoretical_best > 0:
                all_theoretical_bests[driver] = theoretical_best
        
        # Calculate performance-based probabilities
        if all_theoretical_bests:
            # Get the fastest theoretical time as reference
            fastest_time = min(all_theoretical_bests.values())
            
        for driver in current_data['Driver'].unique():
            driver_data = current_data[current_data['Driver'] == driver]
            
            # Get theoretical best for this driver
            theoretical_best = all_theoretical_bests.get(driver, 0)
            
            # Purple sector probability
            purple_sectors = []
            for sector in [1, 2, 3]:
                if f'BestSector{sector}Time' in driver_data.columns:
                    all_sector_times = current_data.groupby('Driver')[f'BestSector{sector}Time'].min()
                    if driver in all_sector_times.index and pd.notna(all_sector_times[driver]):
                        sector_rank = all_sector_times.rank(ascending=True)[driver]  # 1 = fastest
                        purple_prob = max(0.05, 1.0 - (sector_rank - 1) / len(all_sector_times))
                        purple_sectors.append(purple_prob)
            
            # Calculate pole and Q3 probabilities based on theoretical best performance
            if theoretical_best > 0 and fastest_time > 0:
                # Calculate time gap to fastest
                time_gap = theoretical_best - fastest_time
                
                # Pole probability: exponential decay based on time gap
                # Faster drivers get higher probability
                pole_prob = max(0.001, np.exp(-time_gap * 10))  # Scale factor of 10 for realistic gaps
                
                # Q3 probability: more lenient, top ~10 drivers should have good chance
                # Convert to position estimate and calculate Q3 probability
                estimated_position = 1 + (time_gap / 0.1)  # Rough estimate: 0.1s = 1 position
                q3_prob = max(0.05, 1.0 / (1 + np.exp((estimated_position - 10) / 2)))
                
                # Apply recent form modifier (but don't let it dominate)
                recent_form = driver_data['RecentForm'].mean() if 'RecentForm' in driver_data.columns else 10
                form_multiplier = 0.5 + (recent_form / 20)  # Range: 0.5 to 1.0
                
                pole_prob *= form_multiplier
                q3_prob *= form_multiplier
                
            else:
                # Fallback for drivers without sector data
                recent_form = driver_data['RecentForm'].mean() if 'RecentForm' in driver_data.columns else 10
                q3_prob = max(0.05, 1 / (1 + np.exp((recent_form - 10) / 2)))
                pole_prob = q3_prob * 0.1  # Lower pole chance without sector data
            
            # Ensure probabilities are reasonable
            pole_prob = min(pole_prob, 1)  # Cap at 30%
            q3_prob = min(q3_prob, 0.99)   # Cap at 99%
            
            # Position prediction based on theoretical best
            position_range = self._predict_quali_position_from_time(theoretical_best, all_theoretical_bests)
            
            predictions.append({
                'Driver': driver,
                'TheoreticalBest': theoretical_best if theoretical_best > 0 else np.nan,
                'PurpleSectorProbs': purple_sectors,
                'Q3Probability': q3_prob,
                'PoleProbability': pole_prob,
                'PredictedPositionMin': position_range[0],
                'PredictedPositionMax': position_range[1]
            })
        
        return pd.DataFrame(predictions)
    
    def _predict_quali_position_from_time(self, theoretical_best: float, all_times: Dict[str, float]) -> Tuple[float, float]:
        """Predict qualifying position based on theoretical best time"""
        if theoretical_best <= 0 or not all_times:
            return (10, 15)  # Default range
        
        # Sort times to get position estimate
        sorted_times = sorted(all_times.values())
        
        try:
            # Find where this time would rank
            position = sorted_times.index(theoretical_best) + 1
        except ValueError:
            # If exact time not found, estimate position
            position = sum(1 for t in sorted_times if t < theoretical_best) + 1
        
        # Add uncertainty based on how close the times are
        if len(sorted_times) > 1:
            time_spread = max(sorted_times) - min(sorted_times)
            uncertainty = max(1, time_spread * 20)  # Scale uncertainty with time spread
        else:
            uncertainty = 2
        
        min_pos = max(1, position - uncertainty)
        max_pos = min(20, position + uncertainty)
        
        return (min_pos, max_pos)
    
    def _predict_quali_position(self, driver_data: pd.DataFrame) -> Tuple[float, float]:
        """Predict qualifying position with confidence interval"""
        # Simplified prediction based on recent form and consistency
        if 'RecentForm' in driver_data.columns:
            base_position = 20 - (driver_data['RecentForm'].mean() * 0.8)
        else:
            base_position = 10  # Default middle position
            
        if 'PerformanceVolatility' in driver_data.columns:
            uncertainty = driver_data['PerformanceVolatility'].mean() * 2
        else:
            uncertainty = 3  # Default uncertainty
        
        return (max(1, base_position - uncertainty), min(20, base_position + uncertainty))
    
    def monte_carlo_simulation(self, current_data: pd.DataFrame, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulations for race predictions"""
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        results = {driver: [] for driver in current_data['Driver'].unique()}
        
        # Prepare features for prediction
        feature_cols = [col for col in self.scaler.feature_names_in_ if col in current_data.columns]
        X = current_data[feature_cols].fillna(current_data[feature_cols].mean())
        X_scaled = self.scaler.transform(X)
        
        # Calculate adaptive uncertainty based on data quality
        data_completeness = 1 - (current_data[feature_cols].isna().sum().sum() / 
                                (len(current_data) * len(feature_cols)))
        base_uncertainty = 2.0 * (1 - data_completeness)
        
        for sim in range(n_simulations):
            # Add noise based on various factors
            # Model uncertainty - affects all features
            model_noise = np.random.normal(0, base_uncertainty, X_scaled.shape)
            
            # Track variance - broadcast to all features
            track_difficulty = current_data['OvertakingDifficulty'].fillna(0.5).mean()
            track_noise = np.random.normal(0, track_difficulty, (X_scaled.shape[0], 1))
            track_noise = np.broadcast_to(track_noise, X_scaled.shape)
            
            # Consistency noise - based on driver volatility
            volatility = current_data['PerformanceVolatility'].fillna(1.0).values
            consistency_noise = np.random.normal(0, volatility.reshape(-1, 1) * 0.5, X_scaled.shape)
            
            # Combine all noise factors
            total_noise = model_noise + track_noise * 0.3 + consistency_noise * 0.3
            X_noisy = X_scaled + total_noise
            
            # Predict positions
            predicted_positions = self.race_model.predict(X_noisy)
            
            # DNF simulation
            dnf_probs = self.dnf_model.predict_proba(X_noisy)[:, 1]
            dnfs = np.random.random(len(dnf_probs)) < dnf_probs
            
            # Assign results
            for i, driver in enumerate(current_data['Driver']):
                if dnfs[i]:
                    results[driver].append(21)  # DNF position
                else:
                    results[driver].append(max(1, min(20, int(predicted_positions[i]))))
        
        # Calculate statistics
        simulation_stats = {}
        for driver, positions in results.items():
            positions = np.array(positions)
            valid_positions = positions[positions <= 20]
            
            if len(valid_positions) > 0:                simulation_stats[driver] = {
                    'mean_position': np.mean(valid_positions),
                    'median_position': np.median(valid_positions),
                    'position_std': np.std(valid_positions),
                    'dnf_probability': np.mean(positions > 20),
                    'points_probability': np.mean(positions <= 10),
                    'confidence_interval': (np.percentile(valid_positions, 5),
                                          np.percentile(valid_positions, 95))
                }
            else:
                # All DNFs case
                simulation_stats[driver] = {
                    'mean_position': 21,
                    'median_position': 21,
                    'position_std': 0,
                    'dnf_probability': 1.0,
                    'points_probability': 0.0,
                    'confidence_interval': (21, 21)
                }
        
        return simulation_stats
    
    def analyze_practice_sessions(self, practice_data: pd.DataFrame) -> Dict:
        """Analyze practice session data for insights"""
        analysis = {}
        
        # Check if FuelLoad column exists, otherwise estimate based on lap times
        if 'FuelLoad' not in practice_data.columns:
            # Estimate fuel load based on lap time patterns
            practice_data['FuelLoad'] = 50  # Default assumption
        
        # Separate qualifying and race runs
        qual_runs = practice_data[practice_data['FuelLoad'] < 20]  # Assuming low fuel for quali
        race_runs = practice_data[practice_data['FuelLoad'] >= 20]
        
        for driver in practice_data['Driver'].unique():
            driver_qual = qual_runs[qual_runs['Driver'] == driver]
            driver_race = race_runs[race_runs['Driver'] == driver]
            
            # Stint analysis
            stints = self._identify_stints(driver_race)
            
            # Degradation modeling
            degradation = self._calculate_degradation(stints)
            
            # Improvement rate
            improvement = 0
            if not driver_qual.empty and 'LapTime' in driver_qual.columns:
                lap_times = driver_qual['LapTime'].dropna()
                if len(lap_times) > 1:
                    # Convert to numeric values for pct_change
                    if hasattr(lap_times.iloc[0], 'total_seconds'):
                        numeric_times = lap_times.apply(lambda x: x.total_seconds())
                    else:
                        numeric_times = lap_times.astype(float)
                    improvement = numeric_times.pct_change().mean()
            
            # Get lap time statistics
            qual_pace = None
            race_pace = None
            consistency = None
            
            if not driver_qual.empty and 'LapTime' in driver_qual.columns:
                qual_laps = driver_qual['LapTime'].dropna()
                if not qual_laps.empty:
                    if hasattr(qual_laps.iloc[0], 'total_seconds'):
                        qual_pace = qual_laps.min().total_seconds()
                    else:
                        qual_pace = float(qual_laps.min())
            
            if not driver_race.empty and 'LapTime' in driver_race.columns:
                race_laps = driver_race['LapTime'].dropna()
                if not race_laps.empty:
                    if hasattr(race_laps.iloc[0], 'total_seconds'):
                        race_pace = race_laps.mean().total_seconds()
                        consistency = race_laps.std().total_seconds()
                    else:
                        race_pace = float(race_laps.mean())
                        consistency = float(race_laps.std())
            
            analysis[driver] = {
                'qualifying_pace': qual_pace,
                'race_pace': race_pace,
                'degradation_rate': degradation,
                'improvement_rate': improvement,
                'consistency': consistency,
                'stint_count': len(stints)
            }
        
        return analysis
    
    def _identify_stints(self, laps: pd.DataFrame) -> List[pd.DataFrame]:
        """Identify continuous stints in lap data"""
        if laps.empty:
            return []
        
        stints = []
        current_stint = []
        
        for i, lap in laps.iterrows():
            if not current_stint or (lap['LapNumber'] - current_stint[-1]['LapNumber'] == 1):
                current_stint.append(lap)
            else:
                if len(current_stint) > 3:
                    stints.append(pd.DataFrame(current_stint))
                current_stint = [lap]
        
        if len(current_stint) > 3:
            stints.append(pd.DataFrame(current_stint))
        
        return stints
    
    def _calculate_degradation(self, stints: List[pd.DataFrame]) -> float:
        """Calculate tire degradation rate using linear regression"""
        if not stints:
            return 0
        
        degradations = []
        for stint in stints:
            if len(stint) > 5:
                # Fit linear regression to lap times
                X = stint['LapNumber'].values.reshape(-1, 1)
                
                # Convert lap times to seconds
                lap_times = stint['LapTime']
                if hasattr(lap_times.iloc[0], 'total_seconds'):
                    y = lap_times.apply(lambda x: x.total_seconds()).values
                else:
                    y = lap_times.astype(float).values
                
                model = LinearRegression()
                model.fit(X, y)
                degradations.append(model.coef_[0])
        
        return np.mean(degradations) if degradations else 0
    
    def visualize_predictions(self, predictions: Dict, simulation_stats: Dict):
        """Create visualizations for predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Check if simulation_stats is valid
        if not simulation_stats:
            logger.warning("No simulation stats to visualize")
            plt.text(0.5, 0.5, 'No simulation data available', 
                    ha='center', va='center', transform=fig.transFigure)
            return fig
        
        # Filter out drivers with no valid finishes
        valid_drivers = {d: s for d, s in simulation_stats.items() 
                        if s['mean_position'] < 21}
        
        if not valid_drivers:
            logger.warning("No drivers with valid finishes to visualize")
            plt.text(0.5, 0.5, 'No valid finish data available', 
                    ha='center', va='center', transform=fig.transFigure)
            return fig
        
        # 1. Position predictions with confidence intervals
        ax1 = axes[0, 0]
        drivers = list(valid_drivers.keys())
        mean_positions = [valid_drivers[d]['mean_position'] for d in drivers]
        confidence_intervals = [valid_drivers[d]['confidence_interval'] for d in drivers]
        
        y_pos = np.arange(len(drivers))
        errors = [[m - ci[0] for m, ci in zip(mean_positions, confidence_intervals)],
                 [ci[1] - m for m, ci in zip(mean_positions, confidence_intervals)]]
        
        ax1.barh(y_pos, mean_positions, xerr=errors, capsize=5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(drivers)
        ax1.set_xlabel('Predicted Position')
        ax1.set_title('Race Position Predictions with 90% Confidence Intervals')
        ax1.invert_xaxis()
        ax1.set_xlim(20, 0)
        
        # 2. DNF Risk Assessment
        ax2 = axes[0, 1]
        all_drivers = list(simulation_stats.keys())
        dnf_probs = [simulation_stats[d]['dnf_probability'] * 100 for d in all_drivers]
        colors = ['red' if p > 20 else 'orange' if p > 10 else 'green' for p in dnf_probs]
        
        y_pos_all = np.arange(len(all_drivers))
        ax2.barh(y_pos_all, dnf_probs, color=colors)
        ax2.set_yticks(y_pos_all)
        ax2.set_yticklabels(all_drivers)
        ax2.set_xlabel('DNF Probability (%)')
        ax2.set_title('DNF Risk Assessment')
        
        # 3. Points Scoring Probability
        ax3 = axes[1, 0]
        points_probs = [simulation_stats[d]['points_probability'] * 100 for d in all_drivers]
        
        ax3.barh(y_pos_all, points_probs, color='lightblue')
        ax3.set_yticks(y_pos_all)
        ax3.set_yticklabels(all_drivers)
        ax3.set_xlabel('Points Scoring Probability (%)')
        ax3.set_title('Probability of Scoring Points (Top 10 Finish)')
        
        # 4. Feature Importance
        ax4 = axes[1, 1]
        if self.feature_importance:
            features = list(self.feature_importance.keys())
            importances = list(self.feature_importance.values())
            
            ax4.bar(range(len(features)), importances)
            ax4.set_xticks(range(len(features)))
            ax4.set_xticklabels(features, rotation=45, ha='right')
            ax4.set_ylabel('Importance')
            ax4.set_title('Feature Importance in Predictions')
        else:
            ax4.text(0.5, 0.5, 'Feature importance not available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, current_data: pd.DataFrame,
                       qualifying_predictions: pd.DataFrame,
                       simulation_stats: Dict,
                       next_race_info: Optional[Dict] = None) -> str:
        """Generate comprehensive prediction report"""
        report = []
        report.append("# F1 PREDICTION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add next race information if available
        if next_race_info:
            report.append(f"## NEXT RACE: {next_race_info['EventName'].upper()}")
            report.append(f"Date: {next_race_info['EventDate'].strftime('%Y-%m-%d')}")
            report.append(f"Location: {next_race_info['Location']}, {next_race_info['Country']}")
            report.append("-" * 50)
        
        # Check if simulation_stats is valid
        if not simulation_stats:
            report.append("Warning: No simulation results available")
            return "\n".join(report)
        
        # Qualifying Predictions
        report.append("## QUALIFYING PREDICTIONS")
        report.append("-" * 50)
        
        qual_sorted = qualifying_predictions.sort_values('PoleProbability', ascending=False)
        for i, (_, driver) in enumerate(qual_sorted.head(10).iterrows()):
            report.append(f"{i+1}. {driver['Driver']}")
            report.append(f"   - Pole Probability: {driver['PoleProbability']:.1%}")
            report.append(f"   - Q3 Probability: {driver['Q3Probability']:.1%}")
            if pd.notna(driver['TheoreticalBest']) and driver['TheoreticalBest'] > 0:
                report.append(f"   - Theoretical Best: {driver['TheoreticalBest']:.3f}s")
        
        # Race Predictions
        report.append("\n## RACE PREDICTIONS")
        report.append("-" * 50)
        
        # Filter out drivers with invalid mean positions
        valid_stats = {d: s for d, s in simulation_stats.items() 
                      if s['mean_position'] < 21}
        
        race_sorted = sorted(valid_stats.items(), 
                           key=lambda x: x[1]['mean_position'])
        
        for i, (driver, stats) in enumerate(race_sorted[:10]):
            report.append(f"{i+1}. {driver}")
            report.append(f"   - Predicted Position: {stats['mean_position']:.1f} "
                         f"({stats['confidence_interval'][0]:.0f}-"
                         f"{stats['confidence_interval'][1]:.0f})")
            report.append(f"   - Points Probability: {stats['points_probability']:.1%}")
            report.append(f"   - DNF Risk: {stats['dnf_probability']:.1%}")
        
        # Key Insights
        report.append("\n## KEY INSIGHTS")
        report.append("-" * 50)
        
        # Most improved
        if 'PointsMomentum' in current_data.columns:
            valid_momentum = current_data[current_data['PointsMomentum'].notna()]
            if not valid_momentum.empty:
                momentum_leader = valid_momentum.loc[valid_momentum['PointsMomentum'].idxmax(), 'Driver']
                report.append(f"• Strongest Momentum: {momentum_leader}")
        
        # High risk drivers
        high_risk = [d for d, s in simulation_stats.items() 
                    if s['dnf_probability'] > 0.2]
        if high_risk:
            report.append(f"• High DNF Risk: {', '.join(high_risk[:3])}")
        
        # Dark horses
        if len(race_sorted) > 10:
            dark_horses = [d for d, s in race_sorted[5:10]
                          if s['points_probability'] > 0.3]
            if dark_horses:
                report.append(f"• Dark Horses for Points: {', '.join(dark_horses[:3])}")

        # Feature analysis insights
        if self.feature_analysis.get('correlation'):
            top_corr = max(self.feature_analysis['correlation'], key=self.feature_analysis['correlation'].get)
            report.append(f"• Feature most correlated with results: {top_corr}")
        if self.feature_importance:
            top_imp = max(self.feature_importance, key=self.feature_importance.get)
            report.append(f"• Model emphasizes: {top_imp}")
        
        # Data quality note
        missing_features = sum(1 for col in self.feature_names 
                             if col not in current_data.columns)
        if missing_features > 0:
            report.append(f"\n• Note: {missing_features} features unavailable in current data")
        
        return "\n".join(report)
    
    def get_next_race(self) -> Optional[Dict]:
        """Get information about the next upcoming race"""
        if not hasattr(self, 'schedule') or not hasattr(self, 'completed_events'):
            logger.warning("Schedule data not available. Run collect_data first.")
            return None
        
        # Find the next race after completed events
        upcoming_events = self.schedule[self.schedule['EventDate'] >= datetime.now()]
        
        if upcoming_events.empty:
            logger.warning("No upcoming races found in the schedule")
            return None
        
        next_race = upcoming_events.iloc[0]
        
        return {
            'EventName': next_race['EventName'],
            'EventDate': next_race['EventDate'],
            'Country': next_race.get('Country', 'Unknown'),
            'Location': next_race.get('Location', 'Unknown'),
            'OfficialEventName': next_race.get('OfficialEventName', next_race['EventName'])
        }
    
    def prepare_prediction_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for predicting the next race using the most recent driver performance"""
        logger.info("Preparing prediction data from most recent performances...")
        
        # Get the most recent race data for each driver
        latest_race = historical_data['Event'].max()
        current_data = historical_data[historical_data['Event'] == latest_race].copy()
        
        # If we don't have enough drivers from the latest race, expand to include more recent races
        if len(current_data) < 15:  # Minimum threshold for reasonable predictions
            logger.warning(f"Only {len(current_data)} drivers found in latest race. Expanding search...")
            recent_races = historical_data['Event'].unique()[-3:]  # Last 3 races
            current_data = historical_data[historical_data['Event'].isin(recent_races)]
            
            # Take the most recent entry for each driver
            current_data = current_data.sort_values('Event').groupby('Driver').tail(1).reset_index(drop=True)
        
        logger.info(f"Prepared prediction data for {len(current_data)} drivers")
        return current_data

    def analyze_track(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze track-specific effects such as grid importance and DNF rate."""
        logger.info("Analyzing track characteristics...")

        results = []
        for event in df['Event'].unique():
            event_data = df[df['Event'] == event]

            if len(event_data) < 5:
                continue

            # Importance of starting grid position using linear regression
            grid_importance = np.nan
            try:
                if event_data['GridPosition'].nunique() > 1:
                    lr = LinearRegression()
                    lr.fit(event_data['GridPosition'].values.reshape(-1, 1),
                           event_data['Position'].values)
                    grid_importance = lr.coef_[0]
            except Exception as e:
                logger.debug(f"Grid importance calculation failed for {event}: {e}")

            overtakes = (event_data['GridPosition'] - event_data['Position']).abs().mean()
            dnf_rate = event_data['DNF'].mean()

            results.append({
                'Event': event,
                'GridImportance': grid_importance,
                'AverageOvertakes': overtakes,
                'DNFRate': dnf_rate
            })

        track_stats = pd.DataFrame(results)
        self.track_stats = track_stats
        return track_stats


# Example usage
def main():
    # Initialize model
    model = F1PredictionModel()
    
    # Collect historical data (use all available races from current year for better training)
    print("Collecting F1 data...")
    historical_data = model.collect_data(year=2025, races=9)
    
    # Get information about the next race
    next_race_info = model.get_next_race()
    if next_race_info:
        print(f"\nPredicting for: {next_race_info['EventName']}")
        print(f"Date: {next_race_info['EventDate'].strftime('%Y-%m-%d')}")
        print(f"Location: {next_race_info['Location']}, {next_race_info['Country']}")
    else:
        print("\nNo upcoming race found, using most recent data for demonstration")
    
    # Engineer features
    print("\nEngineering features...")
    featured_data = model.engineer_features(historical_data)
    
    # Build models with all available historical data
    print("Training ML models...")
    model.build_ensemble_models(featured_data)
    
    # Prepare prediction data using the most recent driver performances
    current_data = model.prepare_prediction_data(featured_data)
    
    # Qualifying predictions
    print("Generating qualifying predictions...")
    qual_predictions = model.predict_qualifying(current_data)
    
    # Monte Carlo simulations
    print("Running Monte Carlo simulations...")
    simulation_results = model.monte_carlo_simulation(current_data, n_simulations=1000)
    
    # Debug: Check simulation results
    if not simulation_results:
        print("Warning: Simulation results are empty!")
    else:
        print(f"Simulation completed for {len(simulation_results)} drivers")
    
    # Generate report
    print("\nGenerating final report...")
    report = model.generate_report(current_data, qual_predictions, simulation_results, next_race_info)
    print(report)
    
    # Visualize results
    print("\nCreating visualizations...")
    fig = model.visualize_predictions(qual_predictions, simulation_results)
    plt.show()

if __name__ == "__main__":
    main()