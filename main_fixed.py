import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)

class F1RacePredictor:
    def __init__(self, year=2025):
        self.year = year
        self.historical_data = {}
        self.current_weekend_data = {}
        
        # Define race calendars
        self.race_calendar = {
            2024: [
                'Bahrain', 'Saudi Arabian', 'Australian', 'Japanese', 'Chinese', 
                'Miami', 'Emilia Romagna', 'Monaco', 'Canadian', 'Spanish', 
                'Austrian', 'British', 'Hungarian', 'Belgian', 'Dutch', 
                'Italian', 'Azerbaijan', 'Singapore', 'United States', 
                'Mexico City', 'São Paulo', 'Las Vegas', 'Qatar', 'Abu Dhabi'
            ],
            2025: [
                'Australian', 'Chinese', 'Japanese', 'Bahrain', 'Saudi Arabian',
                'Miami', 'Emilia Romagna', 'Monaco', 'Canadian', 'Spanish',
                'Austrian', 'British', 'Belgian', 'Hungarian', 'Dutch',
                'Italian', 'Azerbaijan', 'Singapore', 'United States',
                'Mexico City', 'São Paulo', 'Las Vegas', 'Qatar', 'Abu Dhabi'
            ]
        }
        
        # Add track overtaking data (average overtakes per race)
        self.track_overtakes = {
            # Very Low Overtaking (0-15 overtakes)
            'Monaco': 12,
            'Hungarian': 15,
            'Singapore': 18,
            'Abu Dhabi': 20,
            
            # Low Overtaking (20-30 overtakes)
            'Spanish': 25,
            'Australian': 28,
            'Emilia Romagna': 26,
            'Dutch': 24,
            
            # Medium Overtaking (30-45 overtakes)
            'Miami': 35,
            'Canadian': 38,
            'British': 40,
            'Japanese': 32,
            'United States': 42,
            'Qatar': 36,
            
            # High Overtaking (45-60 overtakes)
            'Austrian': 48,
            'Italian': 52,
            'Belgian': 55,
            'Mexico City': 50,
            'Las Vegas': 58,
            
            # Very High Overtaking (60+ overtakes)
            'Bahrain': 65,
            'Saudi Arabian': 68,
            'Chinese': 62,
            'Azerbaijan': 70,
            'São Paulo': 72
        }
        
        # Categorize tracks by overtaking difficulty
        self.track_categories = {
            'very_low': ['Monaco', 'Hungarian', 'Singapore'],
            'low': ['Spanish', 'Australian', 'Emilia Romagna', 'Dutch', 'Abu Dhabi'],
            'medium': ['Miami', 'Canadian', 'British', 'Japanese', 'United States', 'Qatar'],
            'high': ['Austrian', 'Italian', 'Belgian', 'Mexico City', 'Las Vegas'],
            'very_high': ['Bahrain', 'Saudi Arabian', 'Chinese', 'Azerbaijan', 'São Paulo']
        }
    
    def collect_practice_data(self, year, gp_name, session_type):
        """Collect data from practice sessions using FastF1"""
        try:
            # For 2025, use 2024 data if we're still in 2024
            if year == 2025 and datetime.now().year == 2024:
                print(f"⚠️  2025 data not available yet, using 2024 {gp_name} data for testing")
                year = 2024
                
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()
            
            # Extract lap data
            laps = session.laps
            
            # Remove invalid laps
            laps = laps[laps['LapTime'].notna()]
            laps = laps[laps['PitOutTime'].isna()]  # Remove pit out laps
            
            # Get all drivers in the session
            all_drivers = laps['Driver'].unique()
            
            # Separate qualifying runs and race runs
            qualifying_runs = []
            race_runs = []
            
            for driver in all_drivers:
                driver_laps = laps[laps['Driver'] == driver]
                
                # Find the best lap time for this driver (regardless of stint length)
                if len(driver_laps) > 0:
                    best_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
                    
                    # Always add a qualifying run entry for each driver
                    qualifying_runs.append({
                        'Driver': driver,
                        'LapTime': best_lap['LapTime'].total_seconds(),
                        'Compound': best_lap['Compound'],
                        'Session': session_type,
                        'TyreLife': best_lap['TyreLife'],
                        'IsQualifyingRun': len(driver_laps) <= 3  # Flag if it's a proper quali run
                    })
                
                # Also look for race runs (long stints)
                for stint_num in driver_laps['Stint'].unique():
                    stint = driver_laps[driver_laps['Stint'] == stint_num]
                    
                    if len(stint) >= 5:  # Race simulation
                        stint_laps = stint['LapTime'].dt.total_seconds()
                        race_runs.append({
                            'Driver': driver,
                            'AvgLapTime': stint_laps.mean(),
                            'Compound': stint['Compound'].iloc[0],
                            'StintLength': len(stint),
                            'Degradation': stint_laps.iloc[-1] - stint_laps.iloc[0],
                            'Session': session_type
                        })
                        
        except Exception as e:
            print(f"Error loading {session_type}: {e}")
            return pd.DataFrame(), pd.DataFrame()
            
        return pd.DataFrame(qualifying_runs), pd.DataFrame(race_runs)
    
    def extract_weekend_features(self, gp_name):
        """Extract features from all practice sessions"""
        all_quali_runs = []
        all_race_runs = []
        
        # Use appropriate year
        year = self.year
        if self.year == 2025 and datetime.now().year == 2024:
            year = 2024
        
        for session in ['FP1', 'FP2', 'FP3']:
            quali_runs, race_runs = self.collect_practice_data(year, gp_name, session)
            if not quali_runs.empty:
                all_quali_runs.append(quali_runs)
            if not race_runs.empty:
                all_race_runs.append(race_runs)
        
        # Combine all sessions
        quali_df = pd.concat(all_quali_runs, ignore_index=True) if all_quali_runs else pd.DataFrame()
        race_df = pd.concat(all_race_runs, ignore_index=True) if all_race_runs else pd.DataFrame()
        
        return quali_df, race_df
    
    def get_recent_races_before_gp(self, target_gp, target_year=None, num_races=5):
        """
        Get the most recent races before the target GP
        Returns list of tuples: (year, race_name, round_number)
        """
        if target_year is None:
            target_year = self.year
            
        recent_races = []
        
        # Find position of target GP in the calendar
        if target_year in self.race_calendar and target_gp in self.race_calendar[target_year]:
            target_position = self.race_calendar[target_year].index(target_gp)
            
            # Strategy 1: Get previous races from the same season
            for i in range(target_position - 1, -1, -1):
                if len(recent_races) < num_races:
                    race_name = self.race_calendar[target_year][i]
                    recent_races.append((target_year, race_name, i + 1))
            
            # Strategy 2: If we need more races, get from end of previous season
            if len(recent_races) < num_races and (target_year - 1) in self.race_calendar:
                prev_year = target_year - 1
                prev_calendar = self.race_calendar[prev_year]
                
                # Start from the last race of previous season
                for i in range(len(prev_calendar) - 1, -1, -1):
                    if len(recent_races) < num_races:
                        race_name = prev_calendar[i]
                        recent_races.append((prev_year, race_name, i + 1))
        
        return recent_races
    
    def load_historical_data(self, current_gp, num_recent_races=5):
        """Load recent historical performance with smart weighting"""
        historical_features = {}
        
        # Determine which year we're actually in
        current_date = datetime.now()
        actual_year = current_date.year
        
        # Get recent races
        recent_races = self.get_recent_races_before_gp(current_gp, self.year, num_recent_races)
        
        print(f"\n📊 Loading historical data for {current_gp} {self.year}")
        print(f"📅 Current date: {current_date.strftime('%Y-%m-%d')}")
        print(f"🏁 Loading {len(recent_races)} most recent races:")
        print("-" * 70)
        
        successfully_loaded = 0
        
        for idx, (year, race_name, round_num) in enumerate(recent_races):
            try:
                # Skip future races (2025 races if we're still in 2024)
                if year > actual_year:
                    print(f"⏭️  Skipping {year} {race_name} (future race)")
                    continue
                
                print(f"📥 Loading: {year} {race_name} (Round {round_num})", end='')
                
                # Load race data
                race = fastf1.get_session(year, round_num, 'R')
                race.load()
                results = race.results
                
                # Calculate weight - exponential decay based on recency
                # Most recent race (idx=0) gets highest weight
                time_decay = np.exp(-0.15 * idx)  # Decay factor
                
                # Additional weight adjustments
                season_penalty = 0.9 if year < self.year else 1.0  # Slight penalty for previous season
                
                final_weight = time_decay * season_penalty
                
                print(f" → Weight: {final_weight:.3f}")
                
                # Process driver data
                for _, row in results.iterrows():
                    driver = row['Abbreviation']
                    if pd.isna(driver):
                        continue
                    
                    if driver not in historical_features:
                        historical_features[driver] = {
                            'race_positions': [],
                            'quali_positions': [],
                            'points': [],
                            'dnf_count': 0,
                            'total_races': 0,
                            'weights': [],
                            'recent_momentum': [],  # Track if improving or declining
                            'team_changes': []
                        }
                    
                    # Store race data
                    race_pos = row['Position'] if pd.notna(row['Position']) else 20
                    historical_features[driver]['race_positions'].append(race_pos)
                    historical_features[driver]['points'].append(row['Points'] if pd.notna(row['Points']) else 0)
                    
                    # Track DNFs
                    if row['Status'] != 'Finished' and '+' not in str(row['Status']):
                        historical_features[driver]['dnf_count'] += final_weight
                    
                    # Store weight for this race
                    historical_features[driver]['weights'].append(final_weight)
                    historical_features[driver]['total_races'] += 1
                    
                    # Calculate momentum (position change from previous race)
                    if len(historical_features[driver]['race_positions']) > 1:
                        prev_pos = historical_features[driver]['race_positions'][-2]
                        momentum = prev_pos - race_pos  # Positive = improvement
                        historical_features[driver]['recent_momentum'].append(momentum * final_weight)
                
                successfully_loaded += 1
                
            except Exception as e:
                print(f" ❌ Error: {str(e)[:50]}...")
                continue
        
        print(f"\n✅ Successfully loaded {successfully_loaded}/{len(recent_races)} races")
        
        return self._calculate_enhanced_stats(historical_features)
    
    def _calculate_enhanced_stats(self, historical_features):
        """Calculate statistics with enhanced weighting and momentum"""
        stats = {}
        
        for driver, data in historical_features.items():
            if not data['race_positions']:
                continue
            
            weights = np.array(data['weights'])
            race_pos = np.array(data['race_positions'])
            points = np.array(data['points'])
            
            # Weighted statistics
            avg_race_pos = np.average(race_pos, weights=weights)
            avg_points = np.average(points, weights=weights)
            
            # Consistency (lower std dev = more consistent)
            if len(race_pos) > 1:
                consistency = 1 / (1 + np.std(race_pos))
            else:
                consistency = 0.5
            
            # Recent form - heavily weight recent races
            if len(race_pos) >= 3:
                recent_weights = weights[-3:]  # Last 3 races
                recent_positions = race_pos[-3:]
                recent_form = np.average(recent_positions, weights=recent_weights)
            else:
                recent_form = avg_race_pos
            
            # Momentum score
            if data['recent_momentum']:
                momentum_score = sum(data['recent_momentum']) / sum(weights)
            else:
                momentum_score = 0
            
            # Performance trend (improving/declining)
            if len(race_pos) >= 3:
                first_half = race_pos[:len(race_pos)//2]
                second_half = race_pos[len(race_pos)//2:]
                trend = np.mean(first_half) - np.mean(second_half)  # Positive = improving
            else:
                trend = 0
            
            stats[driver] = {
                'avg_race_pos': avg_race_pos,
                'avg_points': avg_points,
                'consistency': consistency,
                'dnf_probability': data['dnf_count'] / sum(weights) if sum(weights) > 0 else 0.05,
                'recent_form': recent_form,  # Recent race positions
                'momentum': momentum_score,  # Recent improvement/decline
                'trend': trend,  # Long-term trend
                'races_analyzed': data['total_races'],
                'total_weight': sum(weights),
                'last_position': race_pos[-1] if len(race_pos) > 0 else 15
            }
        
        return stats
    
    def prepare_features_for_ml(self, quali_runs, race_runs, historical_stats):
        """Prepare ML features with historical context"""
        if quali_runs.empty:
            return pd.DataFrame()
        
        # Get list of all drivers from practice session
        session = fastf1.get_session(self.year if self.year <= datetime.now().year else 2024, 
                                    gp_name, 'FP1')
        session.load()
        all_drivers_in_session = session.laps['Driver'].unique()
        
        # Current weekend performance - ensure we have all drivers
        quali_features = quali_runs.groupby('Driver').agg({
            'LapTime': ['min', 'mean', 'std', 'count']
        }).reset_index()
        quali_features.columns = ['Driver', 'best_lap', 'avg_lap', 'lap_consistency', 'lap_count']
        
        # Add any missing drivers with default values
        missing_drivers = set(all_drivers_in_session) - set(quali_features['Driver'])
        if missing_drivers:
            print(f"⚠️  Adding {len(missing_drivers)} drivers without qualifying runs: {missing_drivers}")
            
            # Get average lap time to estimate for missing drivers
            avg_time = quali_features['best_lap'].mean()
            
            missing_data = []
            for driver in missing_drivers:
                missing_data.append({
                    'Driver': driver,
                    'best_lap': avg_time * 1.02,  # 2% slower than average
                    'avg_lap': avg_time * 1.03,
                    'lap_consistency': 0.5,
                    'lap_count': 0
                })
            
            missing_df = pd.DataFrame(missing_data)
            quali_features = pd.concat([quali_features, missing_df], ignore_index=True)
        
        # Calculate gap to fastest
        fastest_lap = quali_features['best_lap'].min()
        quali_features['gap_to_fastest'] = quali_features['best_lap'] - fastest_lap
        quali_features['gap_percentage'] = (quali_features['gap_to_fastest'] / fastest_lap) * 100
        
        # Initialize features dataframe
        features = quali_features.copy()
        
        # Add race run features if available
        if not race_runs.empty:
            race_features = race_runs.groupby('Driver').agg({
                'AvgLapTime': 'mean',
                'Degradation': 'mean',
                'StintLength': 'mean'
            }).reset_index()
            race_features.columns = ['Driver', 'race_pace', 'avg_degradation', 'avg_stint_length']
            
            # Merge with main features
            features = features.merge(race_features, on='Driver', how='left')
        else:
            # Estimate race pace from qualifying
            features['race_pace'] = features['avg_lap'] * 1.05
            features['avg_degradation'] = 0.1
            features['avg_stint_length'] = 20
        
        # Add historical features with proper weighting
        for driver in features['Driver']:
            if driver in historical_stats:
                stats = historical_stats[driver]
                
                # Core historical metrics
                features.loc[features['Driver'] == driver, 'hist_avg_finish'] = stats['avg_race_pos']
                features.loc[features['Driver'] == driver, 'hist_consistency'] = stats['consistency']
                features.loc[features['Driver'] == driver, 'hist_dnf_rate'] = stats['dnf_probability']
                features.loc[features['Driver'] == driver, 'hist_avg_points'] = stats['avg_points']
                
                # Recent form indicators
                features.loc[features['Driver'] == driver, 'recent_form'] = stats['recent_form']
                features.loc[features['Driver'] == driver, 'momentum'] = stats['momentum']
                features.loc[features['Driver'] == driver, 'trend'] = stats['trend']
                features.loc[features['Driver'] == driver, 'last_race_pos'] = stats['last_position']
                
                # Calculate form-adjusted expected position
                form_weight = 0.3  # How much recent form affects prediction
                base_prediction = features.loc[features['Driver'] == driver, 'gap_to_fastest'].iloc[0]
                form_adjustment = (stats['recent_form'] - stats['avg_race_pos']) * form_weight
                features.loc[features['Driver'] == driver, 'form_adjusted_prediction'] = base_prediction - form_adjustment
            else:
                # Default values for drivers without history
                features.loc[features['Driver'] == driver, 'hist_avg_finish'] = 15
                features.loc[features['Driver'] == driver, 'hist_consistency'] = 0.5
                features.loc[features['Driver'] == driver, 'hist_dnf_rate'] = 0.05
                features.loc[features['Driver'] == driver, 'hist_avg_points'] = 2
                features.loc[features['Driver'] == driver, 'recent_form'] = 15
                features.loc[features['Driver'] == driver, 'momentum'] = 0
                features.loc[features['Driver'] == driver, 'trend'] = 0
                features.loc[features['Driver'] == driver, 'last_race_pos'] = 15
                features.loc[features['Driver'] == driver, 'form_adjusted_prediction'] = features.loc[features['Driver'] == driver, 'gap_to_fastest']
        
        # Fill missing values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(features[numeric_columns].mean())
        
        return features
    
    def predict_qualifying(self, features):
        """Enhanced qualifying predictions using historical momentum"""
        if features.empty:
            return pd.DataFrame()
        
        # Base prediction on practice pace
        practice_weight = 0.6
        historical_weight = 0.3
        momentum_weight = 0.1
        
        # Calculate position scores
        features['practice_score'] = features['gap_to_fastest'].rank()
        features['historical_score'] = features['hist_avg_finish']
        features['momentum_score'] = 10 - features['momentum'] * 2  # Convert momentum to position impact
        
        # Combined prediction
        features['predicted_position'] = (
            practice_weight * features['practice_score'] +
            historical_weight * features['historical_score'] +
            momentum_weight * features['momentum_score']
        ).rank().astype(int)
        
        # Confidence calculation based on consistency and recent form stability
        features['confidence'] = (
            50 +  # Base confidence
            20 * features['hist_consistency'] +  # Consistency bonus
            15 * (1 - features['lap_consistency'] / features['lap_consistency'].max()) +  # Practice consistency
            15 * np.exp(-abs(features['trend']) / 5)  # Stability bonus
        )
        features['confidence'] = features['confidence'].clip(0, 100)
        
        # Position range based on historical volatility
        position_variance = 2 * (1 - features['hist_consistency'])
        features['position_lower'] = np.maximum(1, features['predicted_position'] - position_variance)
        features['position_upper'] = np.minimum(20, features['predicted_position'] + position_variance)
        
        # Probability calculations with momentum adjustment
        momentum_boost = features['momentum'].clip(-2, 2) * 5
        
        features['q3_probability'] = np.where(
            features['predicted_position'] <= 10,
            np.clip(100 - (features['predicted_position'] - 1) * 8 + momentum_boost, 0, 100),
            np.clip(30 - features['predicted_position'] + momentum_boost, 0, 100)
        )
        
        features['pole_probability'] = np.where(
            features['predicted_position'] <= 3,
            np.clip(60 - (features['predicted_position'] - 1) * 20 + momentum_boost * 2, 0, 100),
            np.clip(5 - features['predicted_position'] / 2 + momentum_boost, 0, 20)
        )
        
        # Create results dataframe
        results = pd.DataFrame({
            'Driver': features['Driver'],
            'Predicted_Position': features['predicted_position'],
            'Confidence': features['confidence'].round(1),
            'Position_Range': [f"{int(l)}-{int(u)}" for l, u in zip(features['position_lower'], features['position_upper'])],
            'Q3_Probability': features['q3_probability'].round(1),
            'Pole_Probability': features['pole_probability'].round(1),
            'Recent_Form': features['recent_form'].round(1),
            'Momentum': features['momentum'].round(2)
        })
        
        return results.sort_values('Predicted_Position')
    
    def get_track_overtaking_factor(self, gp_name):
        """Calculate overtaking difficulty factor for a track"""
        # Get average overtakes for this track
        avg_overtakes = self.track_overtakes.get(gp_name, 40)  # Default to medium
        
        # Calculate factor (0 = impossible to overtake, 1 = very easy)
        # Monaco (12) -> ~0.17, Baku (70) -> ~0.97
        overtaking_factor = avg_overtakes / 72  # Normalize by max overtakes
        
        # Get category
        for category, tracks in self.track_categories.items():
            if gp_name in tracks:
                return overtaking_factor, category
        
        return 0.5, 'medium'  # Default
    
    def predict_race_with_overtaking(self, features, quali_predictions, gp_name, race_length=58):
        """Enhanced race prediction considering track overtaking characteristics"""
        if features.empty or quali_predictions.empty:
            return pd.DataFrame(), 0.5, 'medium'
        
        # Get track overtaking factor
        overtaking_factor, track_category = self.get_track_overtaking_factor(gp_name)
        
        print(f"\n🛣️  Track Analysis: {gp_name}")
        print(f"   Overtaking Factor: {overtaking_factor:.2f} ({track_category.replace('_', ' ').title()})")
        print(f"   Average Overtakes: {self.track_overtakes.get(gp_name, 'N/A')}")
        
        # Merge qualifying predictions
        features = features.merge(
            quali_predictions[['Driver', 'Predicted_Position']], 
            on='Driver'
        )
        
        features['grid_position'] = features['Predicted_Position']
        
        # ADJUSTED WEIGHTS based on track overtaking difficulty
        # At Monaco, grid position is crucial; at Baku, race pace matters more
        grid_weight = 0.45 - (overtaking_factor * 0.25)  # 0.45 at Monaco, 0.20 at Baku
        pace_weight = 0.15 + (overtaking_factor * 0.15)   # 0.15 at Monaco, 0.30 at Baku
        historical_weight = 0.20
        tire_weight = 0.10 + (overtaking_factor * 0.05)  # More important on high-deg tracks
        consistency_weight = 0.05
        momentum_weight = 0.05
        
        # Calculate race score with track-adjusted weights
        features['race_score'] = (
            grid_weight * features['grid_position'] +
            pace_weight * features['race_pace'].rank() +
            historical_weight * features['hist_avg_finish'] +
            tire_weight * (20 - features['avg_degradation'].rank()) +
            consistency_weight * features['hist_consistency'] * 10 +
            momentum_weight * (10 - features['momentum'] * 2)
        )
        
        # Track-specific adjustments
        if track_category == 'very_low':
            # Monaco-style: Grid position is almost everything
            features['position_lock_factor'] = 0.9  # Very hard to change positions
            features['max_positions_gain'] = 3
            features['max_positions_loss'] = 2
        elif track_category == 'low':
            features['position_lock_factor'] = 0.7
            features['max_positions_gain'] = 5
            features['max_positions_loss'] = 4
        elif track_category == 'medium':
            features['position_lock_factor'] = 0.5
            features['max_positions_gain'] = 8
            features['max_positions_loss'] = 6
        elif track_category == 'high':
            features['position_lock_factor'] = 0.3
            features['max_positions_gain'] = 12
            features['max_positions_loss'] = 10
        else:  # very_high
            features['position_lock_factor'] = 0.1
            features['max_positions_gain'] = 15
            features['max_positions_loss'] = 12
        
        # Apply position lock factor to keep cars closer to grid position on difficult tracks
        features['locked_position'] = (
            features['grid_position'] * features['position_lock_factor'] +
            features['race_score'].rank() * (1 - features['position_lock_factor'])
        )
        
        # Predict base finishing position
        features['predicted_finish_raw'] = features['locked_position'].rank().astype(int)
        
        # Calculate realistic position changes based on track
        features['pace_advantage'] = features['race_pace'].rank() - features['grid_position']
        features['potential_change'] = -features['pace_advantage'] * overtaking_factor
        
        # Limit position changes based on track characteristics
        features['positions_gained'] = np.clip(
            features['potential_change'],
            -features['max_positions_loss'],
            features['max_positions_gain']
        ).round()
        
        # Final predicted position
        features['predicted_finish'] = np.clip(
            features['grid_position'] - features['positions_gained'],
            1, 20
        ).astype(int)
        
        # Recalculate to ensure unique positions
        features['predicted_finish'] = features['predicted_finish'].rank(method='first').astype(int)
        
        # DNF probability (slightly higher on street circuits)
        street_circuit_factor = 1.2 if gp_name in ['Monaco', 'Singapore', 'Azerbaijan', 'Las Vegas'] else 1.0
        features['dnf_probability'] = np.clip(
            features['hist_dnf_rate'] * 100 * street_circuit_factor + 
            (features['lap_consistency'] / features['lap_consistency'].max() * 5),
            0, 40
        )
        
        # Strategy importance based on track
        features['strategy_importance'] = overtaking_factor  # More important when overtaking is possible
        
        # Points probability
        features['points_probability'] = np.where(
            features['predicted_finish'] <= 10,
            100 - features['dnf_probability'],
            np.clip(50 - (features['predicted_finish'] - 10) * 5 - features['dnf_probability'], 0, 100)
        )
        
        # Overtaking opportunities
        features['expected_overtakes'] = abs(features['positions_gained']) * 0.7  # Not every position change is an on-track pass
        
        # Create detailed race results
        race_results = pd.DataFrame({
            'Driver': features['Driver'],
            'Grid_Position': features['grid_position'].astype(int),
            'Predicted_Finish': features['predicted_finish'],
            'Positions_Change': (features['grid_position'] - features['predicted_finish']).astype(int),
            'DNF_Risk': features['dnf_probability'].round(1),
            'Points_Probability': features['points_probability'].round(1),
            'Race_Pace_Rank': features['race_pace'].rank().astype(int),
            'Expected_Overtakes': features['expected_overtakes'].round(1),
            'Strategy_Impact': (features['strategy_importance'] * 100).round(0)
        })
        
        return race_results.sort_values('Predicted_Finish'), overtaking_factor, track_category

    def simulate_race(self, race_predictions, num_simulations=1000):
        """Simple Monte Carlo simulation based on predicted results."""
        if race_predictions.empty:
            return pd.DataFrame()

        drivers = race_predictions['Driver'].tolist()
        results = {d: [] for d in drivers}

        for _ in range(num_simulations):
            sim_positions = {}
            for _, row in race_predictions.iterrows():
                pos = row['Predicted_Finish'] + np.random.normal(0, 2.0)
                pos = np.clip(pos, 1, 20)
                if np.random.rand() < row['DNF_Risk'] / 100:
                    pos = 99
                sim_positions[row['Driver']] = pos

            finished = {d: p for d, p in sim_positions.items() if p != 99}
            order = sorted(finished.items(), key=lambda x: x[1])
            rank = 1
            for drv, _ in order:
                sim_positions[drv] = rank
                rank += 1

            for drv in drivers:
                results[drv].append(sim_positions[drv])

        summary = []
        for drv, pos_list in results.items():
            arr = np.array(pos_list)
            dnfs = (arr == 99).sum()
            finished = arr[arr != 99]
            avg = finished.mean() if len(finished) > 0 else 20
            win = (arr == 1).sum() / len(arr) * 100
            dnf_rate = dnfs / len(arr) * 100
            summary.append({'Driver': drv, 'Avg_Position': avg,
                            'Win_Prob': win, 'DNF_Rate': dnf_rate})

        return pd.DataFrame(summary).sort_values('Avg_Position')

    def generate_full_report(self, gp_name, race_length=58):
        """Generate comprehensive prediction report"""
        print(f"\n{'='*80}")
        print(f"🏁 F1 RACE PREDICTIONS - {gp_name.upper()} GRAND PRIX {self.year}")
        print(f"{'='*80}\n")
        
        # Collect practice data
        print("📊 Analyzing practice session data...")
        quali_runs, race_runs = self.extract_weekend_features(gp_name)
        
        if quali_runs.empty:
            print("❌ No qualifying runs found in practice data!")
            print("Make sure practice sessions have been completed.")
            return None
        
        print(f"✅ Found {len(quali_runs)} qualifying runs and {len(race_runs)} race runs")
        
        # Load historical data with proper recent race weighting
        print("\n📈 Loading historical performance data...")
        historical_stats = self.load_historical_data(gp_name, num_recent_races=5)
        
        # Show historical stats summary
        if historical_stats:
            print("\n📊 Historical Performance Summary (Weighted):")
            print("-" * 70)
            print(f"{'Driver':<12} {'Avg Pos':<10} {'Recent Form':<12} {'Momentum':<10} {'Trend':<10}")
            print("-" * 70)
            
            # Sort by recent form
            sorted_drivers = sorted(historical_stats.items(), 
                                  key=lambda x: x[1]['recent_form'])[:10]
            
            for driver, stats in sorted_drivers:
                trend_symbol = "↗️" if stats['trend'] > 0.5 else "↘️" if stats['trend'] < -0.5 else "→"
                print(f"{driver:<12} {stats['avg_race_pos']:<10.1f} "
                      f"{stats['recent_form']:<12.1f} {stats['momentum']:<10.2f} "
                      f"{stats['trend']:<8.2f} {trend_symbol}")
        
        # Prepare features
        print("\n🔧 Preparing features for ML models...")
        features = self.prepare_features_for_ml(quali_runs, race_runs, historical_stats)
        
        if features.empty:
            print("❌ Could not prepare features for predictions!")
            return None
        
        # Generate qualifying predictions
        print("\n🏁 QUALIFYING PREDICTIONS")
        print("-" * 80)
        quali_predictions = self.predict_qualifying(features)
        
        if not quali_predictions.empty:
            print(f"{'Pos':<4} {'Driver':<12} {'Confidence':<12} {'Range':<10} "
                  f"{'Q3 Prob':<10} {'Pole Prob':<10} {'Form':<8} {'Momentum':<10}")
            print("-" * 80)
            
            for idx, row in quali_predictions.iterrows():
                momentum_symbol = "↗️" if row['Momentum'] > 0.5 else "↘️" if row['Momentum'] < -0.5 else "→"
                print(f"{row['Predicted_Position']:<4} {row['Driver']:<12} "
                      f"{row['Confidence']:.1f}%{'':<6} {row['Position_Range']:<10} "
                      f"{row['Q3_Probability']:.1f}%{'':<4} {row['Pole_Probability']:.1f}%{'':<4} "
                      f"{row['Recent_Form']:.1f}{'':<4} {row['Momentum']:>6.2f} {momentum_symbol}")
        
        # Enhanced race predictions with overtaking factors
        print("\n🏁 RACE PREDICTIONS")
        print("-" * 80)
        
        race_predictions, overtaking_factor, track_category = self.predict_race_with_overtaking(
            features, quali_predictions, gp_name, race_length
        )
        
        if not race_predictions.empty:
            # Show basic track characteristics
            print(f"\n📊 Track Characteristics: {track_category.replace('_', ' ').title()} Overtaking")
            print(f"   Expected Total Overtakes: {self.track_overtakes.get(gp_name, 'N/A')}")

            print(f"\n{'Pos':<4} {'Driver':<12} {'Grid':<6} {'Change':<8} "
                  f"{'DNF Risk':<10} {'Points %':<10} {'Pace':<6} {'Overtakes':<10}")
            print("-" * 80)

            for _, row in race_predictions.iterrows():
                change_symbol = "↗️" if row['Positions_Change'] > 0 else "↘️" if row['Positions_Change'] < 0 else "→"
                print(f"{row['Predicted_Finish']:<4} {row['Driver']:<12} "
                      f"P{row['Grid_Position']:<5} {row['Positions_Change']:+3d} {change_symbol:<4} "
                      f"{row['DNF_Risk']:<8.1f}% {row['Points_Probability']:<8.1f}% "
                      f"P{row['Race_Pace_Rank']:<5} {row['Expected_Overtakes']:<10.1f}")
        
        print("\n" + "="*80)

        # Run Monte Carlo simulation
        mc_results = self.simulate_race(race_predictions)

        return {
            'qualifying': quali_predictions,
            'race': race_predictions,
            'monte_carlo': mc_results,
            'historical_stats': historical_stats,
            'features': features,
            'track_category': track_category,
            'overtaking_factor': overtaking_factor
        }
    

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = F1RacePredictor(year=2025)
    
    # Choose Grand Prix
    gp_name = "Spain"  # Change this to any GP
    
    print(f"🏎️  F1 Race Predictor - {gp_name} GP {predictor.year}")
    print("=" * 80)
    
    # Show track info
    overtaking_factor, category = predictor.get_track_overtaking_factor(gp_name)
    print(f"\n🏁 Track Profile: {gp_name}")
    print(f"   Category: {category.replace('_', ' ').title()} Overtaking")
    print(f"   Average Overtakes: {predictor.track_overtakes.get(gp_name, 'N/A')}")
    print(f"   Overtaking Factor: {overtaking_factor:.2f}")
    
    # Show which races will be loaded for historical context
    recent_races = predictor.get_recent_races_before_gp(gp_name, predictor.year, 5)
    print(f"\n📅 Historical races to be analyzed:")
    print("-" * 70)
    for i, (year, race, round_num) in enumerate(recent_races):
        weight = np.exp(-0.15 * i) * (0.9 if year < predictor.year else 1.0)
        print(f"   {i+1}. {year} {race} (Round {round_num}) - Weight: {weight:.3f}")
    
    predictions = predictor.generate_full_report(gp_name)

    if predictions and not predictions['monte_carlo'].empty:
        print("\n🎲 Monte Carlo Summary:")
        for _, row in predictions['monte_carlo'].head(5).iterrows():
            print(f"   {row['Driver']}: Avg P{row['Avg_Position']:.1f}, "
                  f"Win {row['Win_Prob']:.1f}%, DNF {row['DNF_Rate']:.1f}%")
