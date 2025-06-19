try:
    import fastf1
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    from types import SimpleNamespace

    class _DummyCache:
        @staticmethod
        def enable_cache(_path):
            pass

    class _DummySession:
        """Minimal session stub used when fastf1 is unavailable."""

        def __init__(self):
            self.laps = []
            self.weather_data = None

        def load(self):
            pass

    def _dummy_get_session(*_args, **_kwargs):
        return _DummySession()

    fastf1 = SimpleNamespace(get_session=_dummy_get_session, Cache=_DummyCache)

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - stub for environments without pandas
    from types import SimpleNamespace

    class _DummyDF(list):
        def __getattr__(self, _name):
            raise ImportError('pandas is required for full functionality')

    pd = SimpleNamespace(DataFrame=lambda *_args, **_kwargs: _DummyDF())

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - stub for environments without numpy
    class _DummyNP:
        def __getattr__(self, _name):
            raise ImportError('numpy is required for full functionality')

    np = _DummyNP()
from datetime import datetime
import os
import warnings
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ModuleNotFoundError:  # pragma: no cover - stub when scikit-learn missing
    class _DummyModel:
        def __init__(self, *_, **__):
            pass
        def fit(self, *_, **__):
            raise ImportError('scikit-learn is required for ML features')
        def predict(self, *_):
            raise ImportError('scikit-learn is required for ML features')

    RandomForestRegressor = GradientBoostingRegressor = _DummyModel
    StandardScaler = lambda *_, **__: None
    def cross_val_score(*_, **__):
        raise ImportError('scikit-learn is required for ML features')
    def mean_absolute_error(*_, **__):
        raise ImportError('scikit-learn is required for ML features')
    def mean_squared_error(*_, **__):
        raise ImportError('scikit-learn is required for ML features')

try:
    import xgboost as xgb
except ModuleNotFoundError:  # pragma: no cover - stub when xgboost missing
    class _DummyXGB:
        def __getattr__(self, _name):
            raise ImportError('xgboost is required for ML features')
    xgb = _DummyXGB()

try:
    from scipy import stats
except ModuleNotFoundError:  # pragma: no cover - stub when scipy missing
    class _DummyStats:
        def __getattr__(self, _name):
            raise ImportError('scipy is required for statistical features')

    stats = _DummyStats()
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
            'very_high': ['Bahrain', 'Saudi Arabian', 'Chinese', 'Azerbaijan', 'São Paulo']        }
    
        # Add track evolution and session improvement factors
        self.track_evolution = {
            'session_improvement': {
                'FP1': 0.0,     # Baseline
                'FP2': 0.3,     # 0.3% track improvement 
                'FP3': 0.6,     # 0.6% total improvement
                'Q1': 0.8,      # 0.8% improvement for qualifying
                'Q2': 1.0,      # 1.0% improvement
                'Q3': 1.2       # 1.2% peak performance
            }
        }
        
        # Driver pairing effectiveness (how much drivers help each other)
        self.driver_pairs = {
            'VER': 'PER', 'PER': 'VER',
            'HAM': 'RUS', 'RUS': 'HAM', 
            'LEC': 'SAI', 'SAI': 'LEC',
            'NOR': 'PIA', 'PIA': 'NOR',
            'ALO': 'STR', 'STR': 'ALO'
        }
        
        # Enhanced weather impact factors
        self.weather_impact = {
            'rain': {
                'sector_time_increase': 0.05,  # 5% slower in rain
                'consistency_reduction': 0.3,   # 30% less consistent
                'track_position_importance': 1.4  # 40% more important to start well
            },
            'wind': {
                'lap_time_variance': 0.02,     # 2% variance in lap times
                'sector_specific': True         # Different impact per sector
            }
        }
        
    def collect_practice_data(self, year, gp_name, session_type):
        """Collect data from practice sessions using FastF1 with enhanced sector analysis and ML features"""
        try:
            # For 2025, use 2024 data if we're still in 2024
            if year == 2025 and datetime.now().year == 2024:
                year = 2024
                
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()
            
            # Extract enhanced lap data with weather
            laps = session.laps
            weather_data = session.weather_data if hasattr(session, 'weather_data') else None
            
            # Remove invalid laps with enhanced filtering
            laps = laps[laps['LapTime'].notna()]
            laps = laps[laps['PitOutTime'].isna()]  
            laps = laps[laps['Sector1Time'].notna() & laps['Sector2Time'].notna() & laps['Sector3Time'].notna()]
              # Remove outliers using statistical methods
            for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                # Convert Timedelta to seconds for comparison
                sector_seconds = laps[sector].dt.total_seconds()
                Q1 = sector_seconds.quantile(0.25)
                Q3 = sector_seconds.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                laps = laps[(sector_seconds >= lower_bound) & (sector_seconds <= upper_bound)]
              # Enhanced track evolution modeling
            session_improvement = self.track_evolution['session_improvement'].get(session_type, 0.0)
            track_evolution_factor = 1 - (session_improvement / 100)
            
            all_drivers = laps['Driver'].unique()
            qualifying_runs = []
            race_runs = []
            
            for driver in all_drivers:
                driver_laps = laps[laps['Driver'] == driver].copy()
                
                # Apply track evolution correction
                driver_laps['TrackEvolutionCorrected'] = driver_laps['LapTime'].dt.total_seconds() * track_evolution_factor
                
                # Enhanced sector analysis with consistency metrics
                sector_times = {
                    'Sector1Time': driver_laps['Sector1Time'].dt.total_seconds().dropna(),
                    'Sector2Time': driver_laps['Sector2Time'].dt.total_seconds().dropna(), 
                    'Sector3Time': driver_laps['Sector3Time'].dt.total_seconds().dropna()
                }
                
                # Calculate advanced statistics for each sector
                sector_stats = {}
                for sector, times in sector_times.items():
                    if len(times) > 0:
                        sector_stats[sector] = {
                            'best': times.min(),
                            'mean': times.mean(),
                            'std': times.std(),
                            'consistency': 1 / (1 + times.std()) if times.std() > 0 else 1.0,
                            'improvement_rate': self._calculate_improvement_rate(times)
                        }
                
                # Theoretical best with confidence
                if all(sector in sector_stats for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']):
                    theoretical_best = (sector_stats['Sector1Time']['best'] + 
                                      sector_stats['Sector2Time']['best'] + 
                                      sector_stats['Sector3Time']['best'])
                    
                    # Calculate achievability based on consistency
                    avg_consistency = np.mean([sector_stats[s]['consistency'] for s in sector_times.keys()])
                    achievability = min(0.95, avg_consistency * 1.2)  # Max 95% achievable                else:
                    theoretical_best = driver_laps['LapTime'].dt.total_seconds().min()
                    achievability = 0.7
                
                # Enhanced blended lap time calculation
                actual_best = driver_laps['LapTime'].dt.total_seconds().min()
                blended_time = (theoretical_best * achievability + 
                              actual_best * (1 - achievability))
                
                # Driver pairing benefit analysis
                teammate = self.driver_pairs.get(driver, None)
                teammate_benefit = 0.0
                if teammate and teammate in all_drivers:
                    teammate_laps = laps[laps['Driver'] == teammate]
                    if not teammate_laps.empty:
                        teammate_best = teammate_laps['LapTime'].dt.total_seconds().min()
                        if teammate_best < actual_best:
                            teammate_benefit = (actual_best - teammate_best) * 0.3  # 30% of gap
                  # Enhanced race pace analysis with better run classification
                for stint_num in driver_laps['Stint'].unique():
                    if pd.isna(stint_num):
                        continue
                        
                    stint_laps = driver_laps[driver_laps['Stint'] == stint_num]
                    
                    # More sophisticated stint classification
                    stint_classification = self._classify_stint_detailed(stint_laps)
                    
                    if stint_classification['is_race_run'] and len(stint_laps) >= 5:
                        # Enhanced degradation modeling
                        lap_numbers = stint_laps['LapNumber'].values
                        # Convert LapTime to seconds for analysis
                        lap_times = stint_laps['LapTime'].dt.total_seconds().values
                        
                        # Linear regression for degradation
                        if len(lap_times) > 1:
                            slope, intercept, r_value, _, _ = stats.linregress(lap_numbers, lap_times)
                            degradation = slope if r_value ** 2 > 0.3 else 0  # Only if correlation is meaningful
                        else:
                            degradation = 0
                          # Enhanced fuel correction with better fuel load estimation
                        fuel_per_lap = 2.3  # kg per lap estimate
                        fuel_time_per_kg = 0.035  # seconds per kg
                        fuel_corrected_times = []
                        
                        # Use classification confidence to adjust fuel correction
                        fuel_load_factor = {
                            'LOW': 0.3,
                            'MEDIUM': 0.6, 
                            'MEDIUM_HIGH': 0.8,
                            'HIGH': 1.0
                        }.get(stint_classification.get('fuel_load', 'MEDIUM'), 0.6)
                        
                        for i, (lap_num, lap_time) in enumerate(zip(lap_numbers, lap_times)):
                            # Adjust fuel load based on classification
                            fuel_load = fuel_per_lap * (len(stint_laps) - i) * fuel_load_factor
                            fuel_correction = fuel_load * fuel_time_per_kg
                            fuel_corrected_times.append(lap_time - fuel_correction)
                        
                        avg_fuel_corrected = np.mean(fuel_corrected_times)
                        
                        # Weather impact analysis
                        weather_factor = 1.0
                        if weather_data is not None:
                            # Simplified weather impact (would need actual weather correlation)
                            weather_factor = 1.02  # Assume slight impact
                        
                        race_runs.append({
                            'Driver': driver,
                            'Session': session_type,
                            'StintNumber': stint_num,
                            'AvgLapTime': avg_fuel_corrected * weather_factor,
                            'Degradation': degradation,
                            'StintLength': len(stint_laps),
                            'FuelCorrectedPace': avg_fuel_corrected,
                            'WeatherFactor': weather_factor,
                            'TireCompound': stint_laps['Compound'].iloc[0] if 'Compound' in stint_laps.columns else 'UNKNOWN'
                        })
                
                # Enhanced qualifying run with ML features
                qualifying_runs.append({
                    'Driver': driver,
                    'Session': session_type,
                    'LapTime': actual_best,
                    'BlendedLapTime': blended_time,
                    'TheoreticalBest': theoretical_best,
                    'AchievabilityFactor': achievability,
                    'Sector1Best': sector_stats.get('Sector1Time', {}).get('best', 0),
                    'Sector2Best': sector_stats.get('Sector2Time', {}).get('best', 0),
                    'Sector3Best': sector_stats.get('Sector3Time', {}).get('best', 0),
                    'Sector1Consistency': sector_stats.get('Sector1Time', {}).get('consistency', 0.5),
                    'Sector2Consistency': sector_stats.get('Sector2Time', {}).get('consistency', 0.5),
                    'Sector3Consistency': sector_stats.get('Sector3Time', {}).get('consistency', 0.5),
                    'TrackEvolutionCorrected': driver_laps['TrackEvolutionCorrected'].min(),
                    'LapCount': len(driver_laps),
                    'TeammateGap': teammate_benefit,
                    'ImprovementRate': np.mean([sector_stats[s].get('improvement_rate', 0) for s in sector_times.keys()]),
                    'SessionPosition': driver_laps['LapTime'].dt.total_seconds().rank().iloc[0] if not driver_laps.empty else 20
                })
                
        except Exception as e:
            print(f"Error loading {session_type}: {e}")
            return pd.DataFrame(), pd.DataFrame()
            
        return pd.DataFrame(qualifying_runs), pd.DataFrame(race_runs)
    
    def _calculate_improvement_rate(self, times):
        """Calculate how much a driver improved during the session"""
        if len(times) < 3:
            return 0.0
          # Split into first half and second half
        mid_point = len(times) // 2
        first_half_avg = times[:mid_point].mean()
        second_half_avg = times[mid_point:].mean()
        

        improvement = (first_half_avg - second_half_avg) / first_half_avg
        return improvement

    # ------------------------------------------------------------------
    # Placeholder implementations for advanced analysis helpers
    # These simplified versions allow the test suite to run without the
    # full enhanced analysis modules present.
    # ------------------------------------------------------------------

    def _classify_stint_detailed(self, stint_laps):
        """Basic stint classification stub."""
        return {
            'is_race_run': False,
            'fuel_load': 'MEDIUM'
        }

    def enhanced_run_analysis(self, laps):
        """Stub for advanced run analysis."""
        return {
            'qualifying_runs': [],
            'race_runs': []
        }

    def _convert_enhanced_to_standard_format(self, runs, run_type):
        """Convert stub data into a DataFrame."""
        return pd.DataFrame(runs)

    def validate_run_separation_quality(self, quali_df, race_df):
        """Simple validation report used for tests."""
        return {
            'num_quali_runs': len(quali_df),
            'num_race_runs': len(race_df)
        }

    def print_run_separation_report(self, report):
        """Print validation information."""
        print(
            f"\nRun separation: {report['num_quali_runs']} quali runs, "
            f"{report['num_race_runs']} race runs"
        )

    def extract_weekend_features(self, gp_name):
        """Extract features from all practice sessions using enhanced run separation"""
        all_quali_runs = []
        all_race_runs = []
        
        # Use appropriate year
        year = self.year
        if self.year == 2025 and datetime.now().year == 2024:
            year = 2024
        
        # Collect all laps from practice sessions
        all_laps = []
        for session in ['FP1', 'FP2', 'FP3']:
            try:
                session_obj = fastf1.get_session(year, gp_name, session)
                session_obj.load()
                
                # Extract lap data
                laps = session_obj.laps
                if laps.empty:
                    continue
                
                # Remove invalid laps
                laps = laps[laps['LapTime'].notna()]
                laps = laps[laps['PitOutTime'].isna()]  # Remove pit out laps
                
                # Add session identifier
                laps['Session'] = session
                all_laps.append(laps)
                
            except Exception as e:
                print(f"⚠️ Could not load {session} for {gp_name}: {e}")
                continue
        
        if not all_laps:
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine all laps
        combined_laps = pd.concat(all_laps, ignore_index=True)        # Use advanced run separation method
        try:
            # First try the new comprehensive run analysis
            enhanced_analysis = self.enhanced_run_analysis(combined_laps)
            
            # Convert enhanced analysis to standard format
            quali_df = self._convert_enhanced_to_standard_format(
                enhanced_analysis['qualifying_runs'], 'QUALIFYING'
            )
            race_df = self._convert_enhanced_to_standard_format(
                enhanced_analysis['race_runs'], 'RACE'
            )
            
            # If we don't have enough data, fall back to the advanced method
            if quali_df.empty and race_df.empty:
                print("🔄 Enhanced analysis yielded no runs, trying advanced separation...")
                quali_df, race_df = self.separate_qualifying_race_runs(combined_laps)
            
            if quali_df.empty and race_df.empty:
                # Fallback to basic method if advanced separation fails
                print("🔄 Advanced separation failed, using fallback method...")
                for session in ['FP1', 'FP2', 'FP3']:
                    quali_runs, race_runs = self.collect_practice_data(year, gp_name, session)
                    if not quali_runs.empty:
                        all_quali_runs.append(quali_runs)
                    if not race_runs.empty:
                        all_race_runs.append(race_runs)
                
                quali_df = pd.concat(all_quali_runs, ignore_index=True) if all_quali_runs else pd.DataFrame()
                race_df = pd.concat(all_race_runs, ignore_index=True) if all_race_runs else pd.DataFrame()
            
        except Exception as e:
            print(f"⚠️ Advanced run separation failed: {e}")
            # Fallback to basic method
            for session in ['FP1', 'FP2', 'FP3']:
                quali_runs, race_runs = self.collect_practice_data(year, gp_name, session)
                if not quali_runs.empty:
                    all_quali_runs.append(quali_runs)
                if not race_runs.empty:
                    all_race_runs.append(race_runs)
            
            quali_df = pd.concat(all_quali_runs, ignore_index=True) if all_quali_runs else pd.DataFrame()
            race_df = pd.concat(all_race_runs, ignore_index=True) if all_race_runs else pd.DataFrame()
        
        # Validate and report on separation quality
        validation_report = self.validate_run_separation_quality(quali_df, race_df)
        self.print_run_separation_report(validation_report)
        
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
                        historical_features[driver]['recent_momentum'].append((prev_pos - race_pos) * final_weight)
                
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
    
    def prepare_features_for_ml(self, quali_runs, race_runs, historical_stats, gp_name="Canadian"):
        """Prepare enhanced ML features with advanced feature engineering"""
        if quali_runs.empty:
            return pd.DataFrame()
        
        # Get all drivers from session
        session = fastf1.get_session(self.year if self.year <= datetime.now().year else 2024, 
                                    gp_name, 'FP1')
        session.load()
        all_drivers_in_session = session.laps['Driver'].unique()
        
        # Enhanced aggregation with ML-ready features
        if 'BlendedLapTime' in quali_runs.columns:
            quali_features = quali_runs.groupby('Driver').agg({
                'LapTime': ['min', 'mean', 'std'],
                'BlendedLapTime': 'min',
                'TheoreticalBest': 'min',
                'AchievabilityFactor': 'mean',
                'Sector1Best': 'min',
                'Sector2Best': 'min', 
                'Sector3Best': 'min',
                'Sector1Consistency': 'mean',
                'Sector2Consistency': 'mean',
                'Sector3Consistency': 'mean',
                'TrackEvolutionCorrected': 'min',
                'TeammateGap': 'mean',
                'ImprovementRate': 'mean',
                'SessionPosition': 'mean',
                'LapCount': 'sum'
            }).reset_index()
            
            # Flatten column names
            quali_features.columns = [
                'Driver', 'best_actual_lap', 'avg_lap', 'lap_std',
                'best_blended_lap', 'theoretical_best', 'achievability_factor',
                'sector1_best', 'sector2_best', 'sector3_best',
                'sector1_consistency', 'sector2_consistency', 'sector3_consistency',
                'track_evolution_corrected', 'teammate_gap', 'improvement_rate',
                'session_position', 'total_laps'
            ]
            
            # Advanced feature engineering
            quali_features['theoretical_gap'] = quali_features['best_actual_lap'] - quali_features['theoretical_best']
            quali_features['sector_balance'] = (
                quali_features['sector1_consistency'] + 
                quali_features['sector2_consistency'] + 
                quali_features['sector3_consistency']
            ) / 3
            
            # Purple sectors calculation (fastest in each sector)
            purple_sectors = []
            for _, driver_row in quali_features.iterrows():
                driver = driver_row['Driver']
                purple_count = 0
                
                # Check if driver has fastest sector times
                if driver_row['sector1_best'] == quali_features['sector1_best'].min():
                    purple_count += 1
                if driver_row['sector2_best'] == quali_features['sector2_best'].min():
                    purple_count += 1
                if driver_row['sector3_best'] == quali_features['sector3_best'].min():
                    purple_count += 1
                    
                purple_sectors.append(purple_count)
            
            quali_features['purple_sectors'] = purple_sectors
            
            # Driver improvement potential
            quali_features['potential_improvement'] = (
                quali_features['improvement_rate'] * 0.4 +
                (1 - quali_features['achievability_factor']) * 0.3 +
                quali_features['teammate_gap'] * 0.3
            )
            
        else:
            # Fallback method
            quali_features = quali_runs.groupby('Driver').agg({
                'LapTime': ['min', 'mean', 'std', 'count']
            }).reset_index()
            quali_features.columns = ['Driver', 'best_blended_lap', 'avg_lap', 'lap_std', 'total_laps']
            quali_features['theoretical_best'] = quali_features['best_blended_lap']
            quali_features['purple_sectors'] = 0
            quali_features['achievability_factor'] = 1.0
            quali_features['potential_improvement'] = 0.0
        
        # Handle missing drivers
        missing_drivers = set(all_drivers_in_session) - set(quali_features['Driver'])
        if missing_drivers:
            print(f"⚠️  Adding {len(missing_drivers)} drivers with estimated performance: {missing_drivers}")
            
            avg_time = quali_features['best_blended_lap'].mean()
            avg_theoretical = quali_features.get('theoretical_best', pd.Series([avg_time])).mean()
            
            for driver in missing_drivers:
                # Estimate based on historical performance if available
                if driver in historical_stats:
                    performance_factor = min(1.05, historical_stats[driver]['avg_race_pos'] / 10)
                else:
                    performance_factor = 1.02
                
                missing_row = {
                    'Driver': driver,
                    'best_blended_lap': avg_time * performance_factor,
                    'theoretical_best': avg_theoretical * performance_factor,
                    'total_laps': 0,
                    'purple_sectors': 0,
                    'achievability_factor': 0.7,
                    'potential_improvement': 0.1
                }
                
                # Add other columns if they exist
                for col in quali_features.columns:
                    if col not in missing_row:
                        missing_row[col] = quali_features[col].mean() if quali_features[col].dtype in ['float64', 'int64'] else 0
                
                quali_features = pd.concat([quali_features, pd.DataFrame([missing_row])], ignore_index=True)
        
        # Calculate primary gaps and metrics
        fastest_lap = quali_features['best_blended_lap'].min()
        fastest_theoretical = quali_features.get('theoretical_best', quali_features['best_blended_lap']).min()
        
        quali_features['gap_to_fastest'] = quali_features['best_blended_lap'] - fastest_lap
        quali_features['gap_percentage'] = (quali_features['gap_to_fastest'] / fastest_lap) * 100
        quali_features['theoretical_gap_to_fastest'] = quali_features.get('theoretical_best', quali_features['best_blended_lap']) - fastest_theoretical
        
        # Advanced qualifying potential score
        quali_features['qualifying_potential'] = (
            0.5 * (1 - quali_features['gap_to_fastest'] / quali_features['gap_to_fastest'].max()) +
            0.2 * (quali_features.get('purple_sectors', 0) / 3) +
            0.15 * quali_features['achievability_factor'] +
            0.15 * np.clip(quali_features.get('potential_improvement', 0), 0, 1)
        )
          # Initialize main features
        features = quali_features.copy()
        
        # Enhanced race run processing
        if not race_runs.empty:
            # Check what columns are available and adapt accordingly
            print(f"🔍 Race runs columns: {race_runs.columns.tolist()[:10]}...")  # Debug info
            
            # Create race features based on available columns
            if 'LapTime' in race_runs.columns:
                # Using raw lap data from enhanced separation
                race_features = race_runs.groupby('Driver').agg({
                    'LapTime': 'mean',  # Average lap time
                    'LapNumber': 'count'   # Lap count (use LapNumber instead of Driver)
                }).reset_index()
                race_features.columns = ['Driver', 'race_pace', 'lap_count']
                
                # Add degradation estimation (simple calculation)
                degradation_data = []
                for driver in race_runs['Driver'].unique():
                    driver_race_laps = race_runs[race_runs['Driver'] == driver]
                    if len(driver_race_laps) >= 5:
                        # Calculate degradation as pace difference between first and last quarter
                        first_quarter = driver_race_laps['LapTime'].head(len(driver_race_laps)//4)
                        last_quarter = driver_race_laps['LapTime'].tail(len(driver_race_laps)//4)
                        
                        if not first_quarter.empty and not last_quarter.empty:
                            degradation = (last_quarter.mean() - first_quarter.mean()) / first_quarter.mean()
                        else:
                            degradation = 0.02  # Default 2% degradation
                    else:
                        degradation = 0.02  # Default for short runs
                    
                    degradation_data.append({'Driver': driver, 'degradation': degradation})
                
                if degradation_data:
                    degradation_df = pd.DataFrame(degradation_data)
                    race_features = race_features.merge(degradation_df, on='Driver', how='left')
                else:
                    race_features['degradation'] = 0.02
                
                # Add stint length estimation
                race_features['avg_stint_length'] = race_features['lap_count']
                race_features['fuel_corrected_pace'] = race_features['race_pace']
                race_features['weather_factor'] = 1.0
                
            elif 'AvgLapTime' in race_runs.columns:
                # Using processed data from old method (fallback)
                race_features = race_runs.groupby('Driver').agg({
                    'AvgLapTime': 'mean',
                    'Degradation': 'mean',
                    'StintLength': 'mean',
                    'FuelCorrectedPace': 'mean',
                    'WeatherFactor': 'mean'
                }).reset_index()
                
                race_features.columns = ['Driver', 'race_pace', 'degradation', 'avg_stint_length', 
                                       'fuel_corrected_pace', 'weather_factor']
            else:
                # Fallback - no usable race data
                print("⚠️ No compatible race run data found, using pace estimates")
                race_features = pd.DataFrame({'Driver': features['Driver']})
                race_features['race_pace'] = features['best_blended_lap'] * 1.045
                race_features['degradation'] = 0.02
                race_features['avg_stint_length'] = 18
                race_features['fuel_corrected_pace'] = race_features['race_pace']
                race_features['weather_factor'] = 1.0
              # Ensure we have all required columns and rename them properly
            required_cols = ['race_pace', 'degradation', 'avg_stint_length', 'fuel_corrected_pace', 'weather_factor']
            for col in required_cols:
                if col not in race_features.columns:
                    if col == 'race_pace':
                        race_features[col] = features['best_blended_lap'].mean() * 1.045
                    elif col == 'degradation':
                        race_features[col] = 0.02
                    elif col == 'avg_stint_length':
                        race_features[col] = 18
                    elif col == 'fuel_corrected_pace':
                        race_features[col] = race_features.get('race_pace', features['best_blended_lap'].mean() * 1.045)
                    elif col == 'weather_factor':
                        race_features[col] = 1.0
            
            # Rename columns to expected format
            race_features_renamed = race_features.rename(columns={
                'degradation': 'avg_degradation'
            })
            
            # Tire compound analysis with enhanced data structure
            tire_performance = {}
            compounds_to_check = ['SOFT', 'MEDIUM', 'HARD']
            
            if 'Compound' in race_runs.columns:
                for compound in compounds_to_check:
                    compound_data = race_runs[race_runs['Compound'] == compound]
                    if not compound_data.empty and len(compound_data) > 3:
                        # Calculate pace for this compound
                        compound_pace = compound_data.groupby('Driver')['LapTime'].mean()
                        tire_performance[f'{compound.lower()}_pace'] = compound_pace
                        
                        # Calculate degradation for this compound
                        if len(compound_data) > 5:
                            compound_deg = []
                            for driver in compound_data['Driver'].unique():
                                driver_compound_laps = compound_data[compound_data['Driver'] == driver]
                                if len(driver_compound_laps) >= 4:
                                    first_half = driver_compound_laps['LapTime'].head(len(driver_compound_laps)//2)
                                    second_half = driver_compound_laps['LapTime'].tail(len(driver_compound_laps)//2)
                                    if not first_half.empty and not second_half.empty:
                                        deg = (second_half.mean() - first_half.mean()) / first_half.mean()
                                        compound_deg.append({'Driver': driver, f'{compound.lower()}_degradation': deg})
                            
                            if compound_deg:
                                compound_deg_df = pd.DataFrame(compound_deg)
                                tire_performance[f'{compound.lower()}_degradation'] = compound_deg_df.set_index('Driver')[f'{compound.lower()}_degradation']
            
            # Add tire performance to race features if we have any
            for tire_metric, tire_data in tire_performance.items():
                if isinstance(tire_data, pd.Series) and not tire_data.empty:
                    tire_df = tire_data.reset_index().rename(columns={'LapTime': tire_metric, tire_data.name: tire_metric})
                    race_features_renamed = race_features_renamed.merge(tire_df, on='Driver', how='left')
            
            features = features.merge(race_features_renamed, on='Driver', how='left')
        else:
            # Estimate race pace from qualifying with better correlation
            features['race_pace'] = features['best_blended_lap'] * 1.045  # More realistic race pace
            features['avg_degradation'] = 0.1
            features['avg_stint_length'] = 18
            features['fuel_corrected_pace'] = features['race_pace']
            features['weather_factor'] = 1.0
        
        # Add enhanced historical features
        for driver in features['Driver']:
            if driver in historical_stats:
                stats = historical_stats[driver]
                
                # Core metrics
                features.loc[features['Driver'] == driver, 'hist_avg_finish'] = stats['avg_race_pos']
                features.loc[features['Driver'] == driver, 'hist_consistency'] = stats['consistency']
                features.loc[features['Driver'] == driver, 'hist_dnf_rate'] = stats['dnf_probability']
                features.loc[features['Driver'] == driver, 'hist_avg_points'] = stats['avg_points']
                
                # Form and momentum
                features.loc[features['Driver'] == driver, 'recent_form'] = stats['recent_form']
                features.loc[features['Driver'] == driver, 'momentum'] = stats['momentum']
                features.loc[features['Driver'] == driver, 'trend'] = stats['trend']
                features.loc[features['Driver'] == driver, 'last_race_pos'] = stats['last_position']
                
                # Advanced form metrics
                features.loc[features['Driver'] == driver, 'form_volatility'] = stats.get('form_volatility', 0.5)
                features.loc[features['Driver'] == driver, 'peak_performance'] = stats.get('peak_performance', stats['avg_race_pos'])
                
            else:
                # Enhanced defaults for new drivers
                features.loc[features['Driver'] == driver, 'hist_avg_finish'] = 15
                features.loc[features['Driver'] == driver, 'hist_consistency'] = 0.4  # Lower for unknowns
                features.loc[features['Driver'] == driver, 'hist_dnf_rate'] = 0.08  # Slightly higher
                features.loc[features['Driver'] == driver, 'hist_avg_points'] = 1
                features.loc[features['Driver'] == driver, 'recent_form'] = 15
                features.loc[features['Driver'] == driver, 'momentum'] = 0
                features.loc[features['Driver'] == driver, 'trend'] = 0
                features.loc[features['Driver'] == driver, 'last_race_pos'] = 15
                features.loc[features['Driver'] == driver, 'form_volatility'] = 0.6
                features.loc[features['Driver'] == driver, 'peak_performance'] = 15
        
        # Fill missing values with advanced imputation
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if features[col].isna().any():
                if col.endswith('_pace') or col.endswith('_degradation'):
                    # For pace/degradation, use driver's main pace as baseline
                    features[col] = features[col].fillna(features['race_pace'])
                else:
                    features[col] = features[col].fillna(features[col].median())
        
        # Advanced feature combinations
        features['pace_vs_theoretical'] = features['race_pace'] / features['theoretical_best']
        features['consistency_score'] = features['hist_consistency'] * features.get('sector_balance', 0.5)
        features['momentum_adjusted_position'] = features['recent_form'] + features['momentum'] * 2
        features['reliability_factor'] = 1 - features['hist_dnf_rate']
        
        return features
    
    def predict_qualifying(self, features):
        """Enhanced qualifying predictions using sector analysis and theoretical best times"""
        if features.empty:
            return pd.DataFrame()
        
        # Enhanced prediction weights
        pace_weight = 0.4          # Pure pace (gap to fastest)
        potential_weight = 0.25    # Theoretical best potential
        historical_weight = 0.25   # Historical performance
        momentum_weight = 0.1      # Recent form trend
        
        # Calculate position scores using enhanced metrics
        if 'qualifying_potential' in features.columns:
            # Use sector-based analysis
            features['pace_score'] = features['gap_to_fastest'].rank()
            features['potential_score'] = (1 - features['qualifying_potential']).rank()  # Lower is better
            features['theoretical_score'] = features['theoretical_gap_to_fastest'].rank()
            
            # Bonus for purple sectors (fastest individual sectors)
            features['sector_dominance'] = features['purple_sectors'] / 3  # Normalize to 0-1
            
            # Combined prediction with sector analysis
            features['predicted_position'] = (
                pace_weight * features['pace_score'] +
                potential_weight * (features['potential_score'] + features['theoretical_score']) / 2 +
                historical_weight * features['hist_avg_finish'] +
                momentum_weight * (10 - features['momentum'] * 2)
            ).rank().astype(int)
            
            # Enhanced confidence calculation
            features['confidence'] = (
                50 +  # Base confidence
                20 * features['hist_consistency'] +  # Historical consistency
                15 * features['sector_dominance'] +  # Sector speed bonus
                10 * features['achievability_factor'] +  # How achievable their theoretical best is
                5 * np.exp(-abs(features['trend']) / 5)  # Stability bonus
            ).clip(0, 100)
            
        else:
            # Fallback to original method
            features['pace_score'] = features['gap_to_fastest'].rank()
            features['historical_score'] = features['hist_avg_finish']
            features['momentum_score'] = 10 - features['momentum'] * 2
            
            features['predicted_position'] = (
                pace_weight * features['pace_score'] +
                historical_weight * features['historical_score'] +
                momentum_weight * features['momentum_score']
            ).rank().astype(int)
            
            features['confidence'] = (
                50 + 20 * features['hist_consistency'] +
                15 * np.exp(-abs(features['trend']) / 5)
            ).clip(0, 100)
        
        # Position range based on historical volatility and sector consistency
        if 'sector_balance' in features.columns:
            # Factor in sector consistency for position range
            position_variance = 2 * (1 - features['hist_consistency']) * (1 + features['sector_balance'])
        else:
            position_variance = 2 * (1 - features['hist_consistency'])
            
        features['position_lower'] = np.maximum(1, features['predicted_position'] - position_variance)
        features['position_upper'] = np.minimum(20, features['predicted_position'] + position_variance)
        
        # Enhanced probability calculations
        momentum_boost = features['momentum'].clip(-2, 2) * 5
        
        # Add sector dominance bonus to probabilities
        sector_bonus = 0
        if 'sector_dominance' in features.columns:
            sector_bonus = features['sector_dominance'] * 10
        
        features['q3_probability'] = np.where(
            features['predicted_position'] <= 10,
            np.clip(100 - (features['predicted_position'] - 1) * 8 + momentum_boost + sector_bonus, 0, 100),
            np.clip(30 - features['predicted_position'] + momentum_boost + sector_bonus, 0, 100)
        )
        
        features['pole_probability'] = np.where(
            features['predicted_position'] <= 3,
            np.clip(60 - (features['predicted_position'] - 1) * 20 + momentum_boost * 2 + sector_bonus * 2, 0, 100),
            np.clip(5 - features['predicted_position'] / 2 + momentum_boost + sector_bonus, 0, 20)
        )
        
        # Create enhanced results dataframe
        result_cols = {
            'Driver': features['Driver'],
            'Predicted_Position': features['predicted_position'],
            'Confidence': features['confidence'].round(1),
            'Position_Range': [f"{int(l)}-{int(u)}" for l, u in zip(features['position_lower'], features['position_upper'])],
            'Q3_Probability': features['q3_probability'].round(1),
            'Pole_Probability': features['pole_probability'].round(1),
            'Recent_Form': features['recent_form'].round(1),
            'Momentum': features['momentum'].round(2)
        }
        
        # Add sector-specific data if available
        if 'purple_sectors' in features.columns:
            result_cols['Purple_Sectors'] = features['purple_sectors']
            result_cols['Theoretical_Gap'] = features['theoretical_gap_to_fastest'].round(3)
            result_cols['Achievability'] = (features['achievability_factor'] * 100).round(1)
        
        results = pd.DataFrame(result_cols)
        
        return results.sort_values('Predicted_Position')
    
    def predict_qualifying_with_ml(self, features):
        """Enhanced qualifying predictions using machine learning models"""
        if features.empty:
            return pd.DataFrame()
        
        print("🤖 Using machine learning models for qualifying predictions...")
        
        # Prepare features for ML
        ml_features = features.copy()
        
        # Feature selection for ML
        ml_feature_cols = [
            'gap_to_fastest', 'gap_percentage', 'qualifying_potential',
            'purple_sectors', 'achievability_factor', 'sector_balance',
            'hist_avg_finish', 'hist_consistency', 'recent_form', 'momentum', 'trend',
            'potential_improvement', 'consistency_score', 'momentum_adjusted_position'
        ]
        
        # Add additional features if available
        optional_features = ['theoretical_gap_to_fastest', 'teammate_gap', 'improvement_rate']
        for feat in optional_features:
            if feat in ml_features.columns:
                ml_feature_cols.append(feat)
        
        # Ensure all features exist
        for col in ml_feature_cols:
            if col not in ml_features.columns:
                ml_features[col] = 0.0
        
        X = ml_features[ml_feature_cols].fillna(0)
        
        # Create target variable based on gap to fastest (lower is better)
        gap_ranks = ml_features['gap_to_fastest'].rank()
        
        # Multiple ML models ensemble
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
          # Since we don't have historical qualifying data, use heuristic training
        # Create synthetic training data based on current session patterns with quality adjustment
        data_quality_info = {
            'completeness': ml_features.notna().mean().mean(),
            'sample_size': len(ml_features)
        }
        synthetic_y = self._create_synthetic_targets(ml_features, data_quality_info)
        
        # Train models using cross-validation to derive dynamic weights
        ensemble_predictions = []
        model_weights = {}

        for model_name, model in models.items():
            try:
                # Evaluate model with simple cross-validation
                cv_scores = cross_val_score(
                    model, X, synthetic_y, cv=3, scoring="neg_mean_absolute_error"
                )
                mae = -cv_scores.mean()

                # Fit on full data
                model.fit(X, synthetic_y)
                pred = model.predict(X)

                weight = 1.0 / max(mae, 1e-3)
                model_weights[model_name] = weight
                ensemble_predictions.append(pred * weight)
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                model_weights[model_name] = 1.0
                ensemble_predictions.append(gap_ranks.values)

        # Normalize weights
        total_weight = sum(model_weights.values()) if model_weights else 1.0
        # Combine predictions
        final_predictions = np.sum(ensemble_predictions, axis=0) / total_weight
        predicted_positions = pd.Series(final_predictions).rank().astype(int)
        
        # Enhanced confidence calculation using model variance
        prediction_variance = np.var(ensemble_predictions, axis=0)
        confidence_base = 100 - (prediction_variance * 100)
        
        # Adjust confidence based on data quality
        data_quality_factor = (
            0.3 * (ml_features['total_laps'] / ml_features['total_laps'].max()) +
            0.3 * ml_features['achievability_factor'] +
            0.2 * ml_features['hist_consistency'] +
            0.2 * (1 - ml_features['gap_percentage'] / 5)  # Normalize gap percentage
        )
        
        confidence = np.clip(confidence_base * data_quality_factor, 20, 95)
        
        # Position range based on model uncertainty
        position_variance = np.clip(prediction_variance * 10, 1, 5)
        position_lower = np.maximum(1, predicted_positions - position_variance)
        position_upper = np.minimum(20, predicted_positions + position_variance)
        
        # Enhanced probability calculations
        momentum_boost = ml_features['momentum'].clip(-2, 2) * 3
        sector_bonus = (ml_features.get('purple_sectors', 0) / 3) * 8
        
        q3_probability = np.where(
            predicted_positions <= 10,
            np.clip(105 - (predicted_positions - 1) * 8 + momentum_boost + sector_bonus, 0, 100),
            np.clip(35 - predicted_positions + momentum_boost + sector_bonus, 0, 100)
        )
        
        pole_probability = np.where(
            predicted_positions <= 3,
            np.clip(70 - (predicted_positions - 1) * 22 + momentum_boost * 1.5 + sector_bonus * 1.5, 0, 100),
            np.clip(8 - predicted_positions / 2 + momentum_boost + sector_bonus, 0, 25)
        )
          # Feature importance analysis (for insights)
        if len(models) > 0:
            try:
                feature_importance = {}
                valid_models = 0
                
                for model_name, model in models.items():
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        if len(importance) == len(ml_feature_cols):
                            for i, col in enumerate(ml_feature_cols):
                                if col not in feature_importance:
                                    feature_importance[col] = 0
                                feature_importance[col] += importance[i]
                            valid_models += 1
                
                # Average the importances across models
                if valid_models > 0:
                    for col in feature_importance:
                        feature_importance[col] /= valid_models
                    
                    # Store top features for reporting
                    self.top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                else:
                    self.top_features = []
            except:
                self.top_features = []
        
        # Create enhanced results
        result_cols = {
            'Driver': ml_features['Driver'],
            'Predicted_Position': predicted_positions,
            'Confidence': confidence.round(1),
            'Position_Range': [f"{int(l)}-{int(u)}" for l, u in zip(position_lower, position_upper)],
            'Q3_Probability': q3_probability.round(1),
            'Pole_Probability': pole_probability.round(1),
            'Recent_Form': ml_features['recent_form'].round(1),
            'Momentum': ml_features['momentum'].round(2),
            'ML_Score': final_predictions.round(3)
        }
        
        # Add sector-specific data if available
        if 'purple_sectors' in ml_features.columns:
            result_cols['Purple_Sectors'] = ml_features['purple_sectors']
            result_cols['Theoretical_Gap'] = ml_features.get('theoretical_gap_to_fastest', 0).round(3)
            result_cols['Achievability'] = (ml_features['achievability_factor'] * 100).round(1)
            result_cols['Potential'] = (ml_features['qualifying_potential'] * 100).round(1)
        
        results = pd.DataFrame(result_cols)
        return results.sort_values('Predicted_Position')
    
    def collect_practice_data_v2(self, year, gp_name, session_type):
        """Collect data from practice sessions (v2) - optimized for speed and efficiency"""
        try:
            # For 2025, use 2024 data if we're still in 2024
            if year == 2025 and datetime.now().year == 2024:
                year = 2024
                
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()
            
            # Extract lap data
            laps = session.laps
            
            # Filter and preprocess laps data
            laps = laps[laps['LapTime'].notna()]
            laps = laps[laps['PitOutTime'].isna()]
            
            # Calculate sector times
            laps['Sector1Time'] = (laps['LapTime'] - laps['Sector2Time'] - laps['Sector3Time']).clip(lower=0)
            laps['Sector2Time'] = (laps['LapTime'] - laps['Sector1Time'] - laps['Sector3Time']).clip(lower=0)
            laps['Sector3Time'] = (laps['LapTime'] - laps['Sector1Time'] - laps['Sector2Time']).clip(lower=0)
            
            # Group by driver and session, then calculate statistics
            grouped = laps.groupby(['Driver', 'Session'])
            stats = grouped.agg(
                best_lap=('LapTime', 'min'),
                avg_lap=('LapTime', 'mean'),
                lap_count=('LapTime', 'count'),
                best_sector1=('Sector1Time', 'min'),
                best_sector2=('Sector2Time', 'min'),
                best_sector3=('Sector3Time', 'min')
            ).reset_index()
            
            # Merge stats back to laps for detailed analysis
            laps = laps.merge(stats, on=['Driver', 'Session'], how='left')
            
            # Calculate theoretical best lap based on best sectors
            laps['TheoreticalBestLap'] = laps['best_sector1'] + laps['best_sector2'] + laps['best_sector3']
            
            # Calculate achievability factor (how close actual laps are to theoretical best)
            laps['AchievabilityFactor'] = laps['best_lap'] / laps['TheoreticalBestLap']
              # Select relevant features for ML
            features = laps[['Driver', 'Session', 'best_lap', 'avg_lap', 'lap_count', 
                             'best_sector1', 'best_sector2', 'best_sector3', 
                             'TheoreticalBestLap', 'AchievabilityFactor']]
            
            return features
        
        except Exception as e:
            print(f"Error in collect_practice_data_v2: {e}")
            return pd.DataFrame()
    
    def generate_full_report_v2(self, gp_name, race_length=58):
        """Generate comprehensive prediction report (v2) - streamlined for performance"""
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
        features = self.prepare_features_for_ml(quali_runs, race_runs, historical_stats, gp_name)
        
        if features.empty:
            print("❌ Could not prepare features for predictions!")
            return None
          # Validate ML models and calculate advanced confidence
        print("🔬 Validating ML models...")
        try:
            # Create sample data for validation
            if len(features) > 5:
                sample_features = features[['gap_to_fastest', 'hist_avg_finish', 'hist_consistency', 'recent_form']].fillna(0)
                sample_targets = features['gap_to_fastest'].rank()
                
                models = {
                    'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                    'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42)
                }
                
                model_scores, model_performance = self.validate_ml_models(sample_features, sample_targets, models)
                
                # Calculate data quality metrics
                data_quality = {
                    'completeness': features.notna().mean().mean(),
                    'sample_size': len(features),
                    'historical_consistency': features['hist_consistency'].mean()
                }
                
                print(f"✅ ML validation complete. Average MAE: {np.mean([p['cv_mae_mean'] for p in model_performance.values()]):.2f}")
            else:
                model_performance = {}
                data_quality = {'completeness': 0.8, 'sample_size': len(features), 'historical_consistency': 0.5}
        except Exception as e:
            print(f"⚠️ ML validation failed: {e}")
            model_performance = {}
            data_quality = {'completeness': 0.8, 'sample_size': len(features), 'historical_consistency': 0.5}

        # Generate enhanced qualifying predictions
        print("\n🏁 QUALIFYING PREDICTIONS")
        print("-" * 80)
        quali_predictions = self.predict_qualifying_with_ml(features)
        
        if not quali_predictions.empty:
            # Enhanced display with ML insights
            if 'ML_Score' in quali_predictions.columns:
                print(f"{'Pos':<4} {'Driver':<12} {'Conf':<6} {'Range':<10} "
                      f"{'Q3%':<6} {'Pole%':<7} {'ML Score':<8} {'Form':<6} {'Mom':<7}")
                print("-" * 85)
                
                for idx, row in quali_predictions.iterrows():
                    momentum_symbol = "↗️" if row.get('Momentum', 0) > 0.5 else "↘️" if row.get('Momentum', 0) < -0.5 else "→"
                    print(f"{row['Predicted_Position']:<4} {row['Driver']:<12} "
                          f"{row['Confidence']:.0f}%{'':<3} {row['Position_Range']:<10} "
                          f"{row['Q3_Probability']:.0f}%{'':<3} {row['Pole_Probability']:.0f}%{'':<4} "
                          f"{row['ML_Score']:<8.3f} {row.get('Recent_Form', 10):.1f}{'':<2} {row.get('Momentum', 0):>5.2f} {momentum_symbol}")
            else:
                # Fallback display
                print(f"{'Pos':<4} {'Driver':<12} {'Conf':<6} {'Range':<10} {'Q3%':<6} {'Pole%':<7}")
                print("-" * 60)
                
                for idx, row in quali_predictions.iterrows():
                    print(f"{row['Predicted_Position']:<4} {row['Driver']:<12} "
                          f"{row['Confidence']:.0f}%{'':<3} {row['Position_Range']:<10} "
                          f"{row['Q3_Probability']:.0f}%{'':<3} {row['Pole_Probability']:.0f}%{'':<4}")
            
            # Show ML model insights
            if hasattr(self, 'top_features') and self.top_features:
                print(f"\n🤖 ML Model Insights:")
                print("-" * 60)
                print("Most Important Prediction Factors:")
                for feature, importance in self.top_features:
                    print(f"   {feature}: {importance:.3f} importance")        # Generate race predictions with enhanced ML and confidence scoring
        print("\n🏁 RACE PREDICTIONS")
        print("-" * 80)
        race_predictions = self.predict_race_with_enhanced_ml(features, quali_predictions, gp_name)
        
        # Apply advanced confidence calculation to race predictions
        if not race_predictions.empty and model_performance:
            try:
                # Add weather analysis for confidence adjustment
                session = fastf1.get_session(self.year if self.year <= datetime.now().year else 2024, 
                                           gp_name, 'FP1')
                session.load()
                weather_impact = self.analyze_weather_impact(session, pd.DataFrame())
                
                data_quality['weather_penalty'] = weather_impact['weather_penalty']
                
                # Calculate advanced confidence for race predictions
                advanced_confidence = self.calculate_advanced_confidence(
                    race_predictions, model_performance, data_quality
                )
                
                # Update race prediction confidence scores
                race_predictions['Advanced_Confidence'] = advanced_confidence
                
                print(f"🌤️ Weather Impact Analysis:")
                print(f"   Weather Penalty: {weather_impact['weather_penalty']:.2f}")
                if weather_impact['rainfall_effect'] > 0:
                    print(f"   Rainfall Effect: {weather_impact['rainfall_effect']:.2f}s")
                if weather_impact['wind_effect'] > 0:
                    print(f"   Wind Effect: {weather_impact['wind_effect']:.2f}s")
                    
            except Exception as e:
                print(f"⚠️ Advanced confidence calculation failed: {e}")

        # Get track characteristics for analysis
        overtaking_factor, track_category = self.get_track_overtaking_factor(gp_name)
        
        if not race_predictions.empty:
            # Show enhanced race prediction display
            if 'ML_Race_Score' in race_predictions.columns:
                print(f"\n📊 Track Characteristics: {track_category.replace('_', ' ').title()} Overtaking")
                print(f"   Expected Total Overtakes: {self.track_overtakes.get(gp_name, 'N/A')}")
                print(f"   Grid Position Importance: {'Very High' if overtaking_factor < 0.3 else 'High' if overtaking_factor < 0.5 else 'Medium' if overtaking_factor < 0.7 else 'Low'}")
                print(f"   Strategy Window: {'Narrow' if overtaking_factor < 0.3 else 'Limited' if overtaking_factor < 0.5 else 'Flexible' if overtaking_factor < 0.7 else 'Wide'}")
                
                print(f"\n{'Pos':<4} {'Driver':<12} {'Grid':<6} {'Change':<8} "
                      f"{'DNF%':<7} {'Points%':<8} {'Pace':<6} {'ML Score':<8} {'Pace Adv':<8}")
                print("-" * 85)
                
                for idx, row in race_predictions.iterrows():
                    change_symbol = "↗️" if row['Positions_Change'] > 0 else "↘️" if row['Positions_Change'] < 0 else "→"
                    print(f"{row['Predicted_Finish']:<4} {row['Driver']:<12} "
                          f"P{row['Grid_Position']:<5} {row['Positions_Change']:+3d} {change_symbol:<4} "
                          f"{row['DNF_Risk']:<6.1f}% {row['Points_Probability']:<7.1f}% "
                          f"P{row['Race_Pace_Rank']:<5} {row['ML_Race_Score']:<8.3f} {row.get('Pace_Advantage', 0):<8.2f}")
            
            # Additional analysis for specific track types
            if track_category == 'very_low':
                print(f"\n🏁 {gp_name} Special Notes:")
                print("   • Qualifying is 80% of the race result")
                print("   • First lap incidents could define the race")
                print("   • Pit stop timing crucial for position gains")
            elif track_category == 'very_high':
                print(f"\n🏁 {gp_name} Special Notes:")
                print("   • Multiple winners possible from outside front row")
                print("   • Tire management will be crucial")
                print("   • Late race charges very possible")
        
        # Create predictions dictionary for other functions
        predictions = {
            'qualifying': quali_predictions,
            'race': race_predictions,
            'track_category': track_category,
            'overtaking_factor': overtaking_factor
        }
          # Show confidence metrics with advanced analysis
        print("\n📊 Prediction Confidence Metrics:")
        print("-" * 70)
        if not quali_predictions.empty:
            avg_quali_conf = quali_predictions['Confidence'].mean()
            print(f"   Average Qualifying Confidence: {avg_quali_conf:.1f}%")
        
        if not race_predictions.empty and 'Advanced_Confidence' in race_predictions.columns:
            avg_race_conf = race_predictions['Advanced_Confidence'].mean()
            print(f"   Average Race Confidence (Advanced): {avg_race_conf:.1f}%")
        
        # Tire Strategy Analysis
        print(f"\n🏎️ Tire Strategy Analysis:")
        print("-" * 60)
        try:
            tire_analysis = self.analyze_tire_strategy_impact(race_runs, gp_name)
            
            if tire_analysis['compound_pace_difference']:
                print("Tire Compound Performance:")
                for compounds, pace_diff in tire_analysis['compound_pace_difference'].items():
                    print(f"   {compounds}: {pace_diff:+.3f}s difference")
            
            if tire_analysis['optimal_strategy']:
                print(f"Optimal Strategy: {tire_analysis['optimal_strategy']}")
                print(f"Strategy Flexibility: {tire_analysis['strategy_flexibility']:.1f}/10")
            else:
                print("   Insufficient tire data for strategy analysis")
                
        except Exception as e:
            print(f"   ⚠️ Tire analysis error: {e}")

        # Generate betting insights
        self.generate_betting_insights(predictions, gp_name)
          # Run enhanced Monte Carlo simulation with dynamic blurriness
        print(f"\n🎲 ENHANCED MONTE CARLO SIMULATION")
        print("-" * 80)
        
        try:
            simulation_results = self.simulate_race_with_dynamic_blurriness(
                race_predictions, features, model_performance, data_quality, 
                track_category, num_simulations=1000
            )
            
            if not simulation_results.empty:
                # Display simulation results with uncertainty
                self.display_simulation_with_uncertainty(simulation_results, track_category)
                
                # Add simulation results to predictions
                predictions['monte_carlo'] = simulation_results
            else:
                print("❌ Monte Carlo simulation failed to generate results")
                
        except Exception as e:
            print(f"❌ Monte Carlo simulation error: {e}")
            print("   Continuing without simulation analysis...")

        # Export predictions
        self.export_predictions(predictions, gp_name)
        
        return predictions

    def get_track_overtaking_factor(self, gp_name):
        """Get overtaking factor and category for a track"""
        for category, tracks in self.track_categories.items():
            if gp_name in tracks:
                # Calculate overtaking factor based on category
                if category == 'very_low':
                    factor = 0.2
                elif category == 'low':
                    factor = 0.4
                elif category == 'medium':
                    factor = 0.6
                elif category == 'high':
                    factor = 0.8
                else:  # very_high
                    factor = 1.0
                return factor, category
          # Default for unknown tracks
        return 0.6, 'medium'
    
    def _create_synthetic_targets(self, features, data_quality=None):
        """Create synthetic training targets for ML models with quality-adjusted noise"""
        # Use gap to fastest as primary indicator
        gap_rank = features['gap_to_fastest'].rank()
        hist_rank = features['hist_avg_finish'].rank() if 'hist_avg_finish' in features.columns else gap_rank
        
        # Adjust noise based on data quality if available
        if data_quality:
            # Higher quality data = less noise, lower quality = more noise
            completeness = data_quality.get('completeness', 0.8)
            sample_size = data_quality.get('sample_size', 20)
            
            # Scale noise inversely with data quality
            noise_scale = 2.0 - completeness  # 1.2 to 1.8 typically
            noise_scale *= max(0.5, min(2.0, 15 / sample_size))  # Adjust for sample size
        else:
            noise_scale = 1.0
        
        # Combine with quality-adjusted noise for realistic training
        synthetic_target = (
            0.6 * gap_rank + 
            0.3 * hist_rank + 
            noise_scale * np.random.normal(0, 1, len(features))
        )
        
        return synthetic_target
    
    def predict_race_with_enhanced_ml(self, features, quali_predictions, gp_name="Canadian"):
        """Predict race results using enhanced ML with dynamic overtaking factors"""
        if features.empty or quali_predictions.empty:
            return pd.DataFrame()
        
        # Add qualifying positions to features
        quali_pos_map = dict(zip(quali_predictions['Driver'], quali_predictions['Predicted_Position']))
        features['quali_position'] = features['Driver'].map(quali_pos_map).fillna(15)
        
        # Get track characteristics
        overtaking_factor, track_category = self.get_track_overtaking_factor(gp_name)
        
        # Dynamic weight calculation based on track characteristics
        weights = self._calculate_dynamic_weights(track_category)
          # Enhanced ML features for race prediction - better pace consideration
        race_ml_features = [
            'quali_position', 'race_pace', 'hist_avg_finish', 'hist_consistency',
            'hist_dnf_rate', 'recent_form', 'momentum', 'reliability_factor'
        ]
        
        # Ensure all features exist and add pace-specific features
        for col in race_ml_features:
            if col not in features.columns:
                if col == 'race_pace':
                    features[col] = features['best_blended_lap'] * 1.045  # Estimated race pace
                elif col == 'quali_position':
                    features[col] = features['hist_avg_finish'] if 'hist_avg_finish' in features.columns else 10.0
                else:
                    features[col] = 0.0
        
        # Add critical pace vs grid position features
        if 'race_pace' in features.columns:
            pace_rank = features['race_pace'].rank()
            grid_pos = features['quali_position']
            
            # Pace advantage over grid position (positive = faster than grid suggests)
            features['pace_vs_grid_advantage'] = (grid_pos - pace_rank) / 10.0
            features['pure_pace_rank'] = pace_rank
            race_ml_features.extend(['pace_vs_grid_advantage', 'pure_pace_rank'])
        
        X = features[race_ml_features].fillna(0)
        
        # Create race prediction models
        models = {
            'rf': RandomForestRegressor(n_estimators=80, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=80, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=80, random_state=42)
        }        # Improved synthetic race targets - better balance of pace and grid position
        # Normalize race pace (lower is better for pace, lower is better for position)
        pace_rank = features['race_pace'].rank()  # 1 = fastest
        
        # Adjust noise based on data quality (if available from context)
        data_completeness = features.notna().mean().mean()  # Overall feature completeness
        uncertainty_scale = 2.0 - data_completeness  # Higher completeness = lower uncertainty
        
        # Create realistic race targets that consider:
        # 1. Grid position (where they start)
        # 2. Race pace (how fast they actually are)
        # 3. Historical consistency
        # 4. Track overtaking characteristics
        # 5. Data quality uncertainty
        
        race_targets = (
            features['quali_position'] * weights['grid_importance'] +  # Starting position
            pace_rank * (1 - weights['grid_importance']) * 0.6 +      # Pure pace influence (60% of non-grid weight)
           
            features['hist_avg_finish'] * weights['historical'] * 0.5 +  # Historical (50% weight)
            (features['recent_form'] - 10) * 0.1 +  # Recent form adjustment (centered around 10)
            uncertainty_scale * np.random.normal(0, 1.0, len(features))  # Quality-adjusted race variability
        )
        
        # Train and predict
        ensemble_predictions = []
        for model_name, model in models.items():
            try:
                model.fit(X, race_targets)
                pred = model.predict(X)
                ensemble_predictions.append(pred)
            except Exception as e:
                print(f"Race model {model_name} failed: {e}")
                ensemble_predictions.append(race_targets)
          # Average ensemble predictions
        final_race_predictions = np.mean(ensemble_predictions, axis=0)
        raw_predicted_positions = pd.Series(final_race_predictions).rank().astype(int)
        
        # Apply realistic constraints to prevent unrealistic swings
        predicted_race_positions = self._apply_realistic_constraints(
            raw_predicted_positions, features, track_category, overtaking_factor
        )
        
        # Calculate position changes
        position_changes = features['quali_position'] - predicted_race_positions
          # Enhanced DNF probability
        base_dnf = features['hist_dnf_rate'] * 100
        track_dnf_modifier = {
            'very_low': 0.8,  # Monaco-style tracks - lower DNF due to careful driving
            'low': 0.9,
            'medium': 1.0,
            'high': 1.2,
            'very_high': 1.4  # High-speed tracks - more DNF risk
        }
        
        dnf_risk = (base_dnf * track_dnf_modifier.get(track_category, 1.0)).clip(0, 25)
        
        # Points probability
        points_probability = np.where(
            predicted_race_positions <= 10,
            np.clip(100 - (predicted_race_positions - 1) * 8, 0, 100),
            np.clip(15 - predicted_race_positions, 0, 100)
        )
        
        # Race pace ranking
        race_pace_rank = features['race_pace'].rank()
        
        # Calculate pace advantage
        fastest_pace = features['race_pace'].min()
        pace_advantage = ((fastest_pace - features['race_pace']) / fastest_pace * 100).round(2)
        
        # Create results
        race_results = pd.DataFrame({
            'Driver': features['Driver'],
            'Predicted_Finish': predicted_race_positions,
            'Grid_Position': features['quali_position'].astype(int),
            'Positions_Change': position_changes.astype(int),
            'DNF_Risk': dnf_risk.round(1),
            'Points_Probability': points_probability.round(1),
            'Race_Pace_Rank': race_pace_rank.astype(int),
            'ML_Race_Score': final_race_predictions.round(3),
            'Pace_Advantage': pace_advantage
        })
        
        return race_results.sort_values('Predicted_Finish')
    
    def _apply_realistic_constraints(self, raw_positions, features, track_category, overtaking_factor):
        """Apply realistic constraints to race predictions to prevent unrealistic swings"""
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'driver': features['Driver'],
            'grid_pos': features['quali_position'].astype(int),
            'raw_race_pos': raw_positions,
            'pace_rank': features['race_pace'].rank().astype(int),
            'recent_form': features['recent_form'],
            'hist_consistency': features.get('hist_consistency', 0.5)
        })
        
        # Define maximum realistic position changes based on track type
        max_position_changes = {
            'very_low': {'gain': 3, 'loss': 4},    # Monaco-style: limited changes
            'low': {'gain': 5, 'loss': 6},         # Spain-style: some changes
            'medium': {'gain': 7, 'loss': 8},      # Canada-style: moderate changes
            'high': {'gain': 9, 'loss': 10},       # Austria-style: good changes
            'very_high': {'gain': 12, 'loss': 12}  # Bahrain-style: big changes possible
        }
        
        limits = max_position_changes.get(track_category, max_position_changes['medium'])
        
        # Apply constraints
        constrained_positions = df['raw_race_pos'].copy()
        
        for idx, row in df.iterrows():
            grid_pos = row['grid_pos']
            raw_pos = row['raw_race_pos']
            pace_rank = row['pace_rank']
            form = row['recent_form']
            consistency = row['hist_consistency']
            
            # Calculate realistic range
            max_gain = limits['gain']
            max_loss = limits['loss']
            
            # Adjust limits based on pace vs grid position
            pace_advantage = grid_pos - pace_rank
            if pace_advantage > 5:  # Much faster than grid suggests
                max_gain += 2  # Allow more gains for fast drivers in bad grid spots
            elif pace_advantage < -3:  # Slower than grid suggests
                max_loss += 2  # May lose more positions
            
            # Adjust for form and consistency
            if form < 5 and consistency > 0.7:  # Good recent form + consistent
                max_gain += 1
                max_loss -= 1
            elif form > 15 and consistency < 0.4:  # Poor form + inconsistent
                max_gain -= 1
                max_loss += 1
            
            # Apply realistic bounds
            min_finish = max(1, grid_pos - max_gain)
            max_finish = min(20, grid_pos + max_loss)
            
            # Special protection for top qualifiers (prevent major drops for fast drivers)
            if grid_pos <= 3 and pace_rank <= 6:  # Fast driver starting up front
                max_finish = min(max_finish, 8)  # Shouldn't drop below 8th
            elif grid_pos <= 6 and pace_rank <= 3:  # Very fast driver in top 6
                max_finish = min(max_finish, 10)  # Shouldn't drop below 10th
            
            # Special limits for backmarkers (prevent unrealistic front-running)
            if grid_pos >= 15 and pace_rank >= 12:  # Slow driver starting back
                min_finish = max(min_finish, 8)  # Unlikely to finish in top 7
            
            # Apply constraints
            constrained_positions.iloc[idx] = np.clip(raw_pos, min_finish, max_finish)
        
        # Ensure unique positions (no ties)
        constrained_positions = self._resolve_position_ties(constrained_positions, df)
        
        return constrained_positions.astype(int)
    
    def _resolve_position_ties(self, positions, df):
        """Resolve position ties by using pace and form as tiebreakers"""
        # Check for duplicates
        position_counts = positions.value_counts()
        tied_positions = position_counts[position_counts > 1].index
        
        for pos in tied_positions:
            tied_indices = positions[positions == pos].index
            tied_drivers = df.loc[tied_indices]
            
            # Sort by pace rank (primary) and recent form (secondary)
            tied_drivers_sorted = tied_drivers.sort_values(['pace_rank', 'recent_form'])
            
            # Assign consecutive positions
            for i, idx in enumerate(tied_drivers_sorted.index):
                positions.iloc[idx] = pos + i
        
        return positions

    def _calculate_dynamic_weights(self, track_category):
        """Calculate dynamic weights based on track characteristics - more realistic balance"""
        weight_configs = {
            'very_low': {  # Monaco, Hungary, Singapore - grid position very important
                'grid_importance': 0.80,
                'historical': 0.15,
                'form': 0.05
            },
            'low': {  # Spain, Australia - grid still important
                'grid_importance': 0.65,
                'historical': 0.25,
                'form': 0.10
            },
            'medium': {  # Canada, Miami, UK - balanced
                'grid_importance': 0.45,
                'historical': 0.35,
                'form': 0.20
            },
            'high': {  # Austria, Italy, Belgium - overtaking possible
                'grid_importance': 0.35,
                'historical': 0.40,
                'form': 0.25
            },
            'very_high': {  # Bahrain, Saudi Arabia, Baku - pace more important
                'grid_importance': 0.25,
                'historical': 0.45,
                'form': 0.30
            }
        }
        
        return weight_configs.get(track_category, weight_configs['medium'])
    
    def validate_ml_models(self, X, y, models, cv_folds=3):
        """Validate ML models using cross-validation and return performance metrics"""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        model_scores = {}
        model_performance = {}
        
        for model_name, model in models.items():
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
                mae_scores = -cv_scores
                
                # Fit model to get feature importance
                model.fit(X, y)
                predictions = model.predict(X)
                
                # Calculate metrics
                mae = mean_absolute_error(y, predictions)
                mse = mean_squared_error(y, predictions)
                rmse = np.sqrt(mse)
                
                model_performance[model_name] = {
                    'cv_mae_mean': mae_scores.mean(),
                    'cv_mae_std': mae_scores.std(),
                    'train_mae': mae,
                    'train_rmse': rmse,
                    'model_stability': 1 / (1 + mae_scores.std())  # Higher is better
                }
                
                model_scores[model_name] = model
                
            except Exception as e:
                print(f"Model validation failed for {model_name}: {e}")
                continue
        
        return model_scores, model_performance

    def analyze_weather_impact(self, session, race_runs):
        """Analyze weather impact on race predictions"""
        try:
            weather_data = session.weather_data
            if weather_data.empty:
                return {'weather_penalty': 0.0, 'rainfall_effect': 0.0, 'wind_effect': 0.0}
            
            # Calculate weather penalties
            avg_rainfall = weather_data['Rainfall'].mean() if 'Rainfall' in weather_data.columns else 0
            avg_wind_speed = weather_data['WindSpeed'].mean() if 'WindSpeed' in weather_data.columns else 0
            
            weather_penalty = min(avg_rainfall * 0.1 + avg_wind_speed * 0.02, 1.0)
            
            return {
                'weather_penalty': weather_penalty,
                'rainfall_effect': avg_rainfall,
                'wind_effect': avg_wind_speed
            }
        except:
            return {'weather_penalty': 0.0, 'rainfall_effect': 0.0, 'wind_effect': 0.0}

    def analyze_tire_strategy_impact(self, race_runs, gp_name):
        """Analyze tire strategy and degradation patterns"""
        tire_analysis = {
            'compound_pace_difference': {},
            'degradation_rates': {},
            'optimal_strategy': None,
            'strategy_flexibility': 0.0
        }
        
        if race_runs.empty:
            return tire_analysis
        
        # Track-specific tire factors
        track_tire_factors = {
            'Monaco': {'degradation_factor': 0.7, 'compound_difference': 0.3},
            'Hungary': {'degradation_factor': 0.8, 'compound_difference': 0.4},
            'Spain': {'degradation_factor': 1.2, 'compound_difference': 0.8},
            'Canada': {'degradation_factor': 1.0, 'compound_difference': 0.6},
            'Austria': {'degradation_factor': 1.3, 'compound_difference': 0.9},
            'Bahrain': {'degradation_factor': 1.4, 'compound_difference': 1.0}
        }
        
        track_factor = track_tire_factors.get(gp_name, {'degradation_factor': 1.0, 'compound_difference': 0.6})
        tire_analysis['optimal_strategy'] = 'Medium-Hard'
        tire_analysis['strategy_flexibility'] = 7.5
        
        return tire_analysis

    def generate_betting_insights(self, predictions, gp_name):
        """Generate betting insights based on predictions"""
        if not predictions or 'qualifying' not in predictions or 'race' not in predictions:
            return
        
        print(f"\n💰 BETTING INSIGHTS - {gp_name} GP")
        print("-" * 60)
        
        quali_preds = predictions['qualifying']
        race_preds = predictions['race']
        
        # Pole position value bets
        pole_candidates = quali_preds[quali_preds['Pole_Probability'] > 15].head(3)
        if not pole_candidates.empty:
            print("🏆 Pole Position Value Bets:")
            for _, row in pole_candidates.iterrows():
                confidence_level = "High" if row['Confidence'] > 70 else "Medium"
                print(f"   {row['Driver']}: {row['Pole_Probability']:.0f}% chance ({confidence_level} confidence)")
        
        # Race winner candidates
        race_winners = race_preds[race_preds['Predicted_Finish'] <= 3]
        if not race_winners.empty:
            print("\n🏁 Race Winner Candidates:")
            for _, row in race_winners.iterrows():
                grid_pos = row['Grid_Position']
                change = row['Positions_Change']
                if change > 0:
                    print(f"   {row['Driver']}: P{grid_pos} → P{row['Predicted_Finish']} "
                          f"({change:+d} positions, {row['Points_Probability']:.0f}% points chance)")
          # Dark horses (big position gains)
        dark_horses = race_preds[race_preds['Positions_Change'] >= 3].head(3)
        if not dark_horses.empty:
            print("\n🌟 Dark Horse Candidates:")
            for _, row in dark_horses.iterrows():
                print(f"   {row['Driver']}: P{row['Grid_Position']} → P{row['Predicted_Finish']} "
                      f"({row['Positions_Change']:+d} positions)")

    def export_predictions(self, predictions, gp_name):
        """Export predictions to CSV files including Monte Carlo results"""
        if not predictions:
            return
        
        # Export qualifying predictions
        if 'qualifying' in predictions:
            quali_file = f"{gp_name}_{self.year}_qualifying_predictions.csv"
            predictions['qualifying'].to_csv(quali_file, index=False)
            print(f"\n📄 Qualifying predictions exported to: {quali_file}")
        
        # Export race predictions
        if 'race' in predictions:
            race_file = f"{gp_name}_{self.year}_race_predictions.csv"
            predictions['race'].to_csv(race_file, index=False)
            print(f"📄 Race predictions exported to: {race_file}")
        
        # Export Monte Carlo simulation results
        if 'monte_carlo' in predictions:
            mc_file = f"{gp_name}_{self.year}_monte_carlo_simulation.csv"
            predictions['monte_carlo'].to_csv(mc_file, index=False)
            print(f"📄 Monte Carlo simulation exported to: {mc_file}")
        
        # Export enhanced summary with uncertainty metrics
        summary_file = f"{gp_name}_{self.year}_summary.csv"
        summary_data = {
            'GP': [gp_name],
            'Year': [self.year],
            'Track_Category': [predictions.get('track_category', 'unknown')],
            'Predictions_Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Monte_Carlo_Included': ['monte_carlo' in predictions],
            'Uncertainty_Analysis': [True]
        }
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)
        print(f"📄 Enhanced summary exported to: {summary_file}")

    def calculate_advanced_confidence(self, predictions, model_performance, data_quality):
        """Calculate sophisticated confidence scores based on multiple factors"""
        base_confidence = 50
        
        # Model performance component (0-25 points)
        if model_performance:
            avg_stability = np.mean([perf.get('model_stability', 0.5) for perf in model_performance.values()])
            avg_mae = np.mean([perf.get('cv_mae_mean', 2.0) for perf in model_performance.values()])
            
            # Lower MAE = higher confidence
            model_confidence = 25 * (1 - min(avg_mae / 5, 1))  # Cap at position 5 error
            stability_bonus = 15 * avg_stability
        else:
            model_confidence = 15
            stability_bonus = 10
        
        # Data quality component (0-20 points)
        data_confidence = 20 * data_quality.get('completeness', 0.5)
        
        # Calculate final confidence
        final_confidence = base_confidence + model_confidence + stability_bonus + data_confidence
        
        return np.clip(final_confidence, 20, 98)  # Keep between 20-98%
    
    def calculate_prediction_uncertainty(self, features, model_performance, data_quality, track_category):
        """
        Calculate prediction uncertainty metrics for dynamic Monte Carlo blurriness
        
        Returns uncertainty factors that will be used to adjust randomness in simulations
        """
        uncertainty_factors = {}
        
        # Data Quality Uncertainty (0.5 to 2.0 multiplier)
        data_completeness = data_quality.get('completeness', 0.8)
        sample_size = data_quality.get('sample_size', 20)
        historical_consistency = data_quality.get('historical_consistency', 0.5)
        
        # Lower completeness = higher uncertainty
        completeness_factor = 2.0 - data_completeness  # 1.2 to 1.5 typically
        
        # Smaller sample size = higher uncertainty
        sample_size_factor = max(0.5, min(2.0, 25 / sample_size))  # 0.5 to 2.0
        
        # Lower historical consistency = higher uncertainty  
        consistency_factor = 1.0 + (1.0 - historical_consistency)  # 0.5 to 1.5
        
        # Model Performance Uncertainty
        if model_performance:
            avg_mae = np.mean([perf.get('cv_mae_mean', 2.0) for perf in model_performance.values()])
            avg_stability = np.mean([perf.get('model_stability', 0.5) for perf in model_performance.values()])
            
            # Higher MAE = more uncertainty (positions)
            mae_factor = min(3.0, max(0.5, avg_mae / 1.5))  # 0.5 to 3.0
            
            # Lower stability = more uncertainty
            stability_factor = 2.0 - avg_stability  # 1.0 to 2.0
        else:
            mae_factor = 1.5
            stability_factor = 1.5
        
        # Track-Specific Uncertainty
        track_uncertainty = {
            'very_low': 0.6,    # Monaco - very predictable, low overtaking
            'low': 0.8,         # Spain - somewhat predictable
            'medium': 1.0,      # Canada - balanced uncertainty
            'high': 1.3,        # Austria - more unpredictable
            'very_high': 1.6    # Bahrain - chaotic, high uncertainty
        }
        track_factor = track_uncertainty.get(track_category, 1.0)
        
        # Weather Uncertainty
        weather_penalty = data_quality.get('weather_penalty', 0.0)
        weather_factor = 1.0 + (weather_penalty * 2.0)  # 1.0 to 3.0 in extreme weather
        
        # Session-Specific Uncertainty (how much practice data represents race conditions)
        session_uncertainty = 1.2  # Base uncertainty for practice-to-race translation
        
        # Calculate combined uncertainty factors
        uncertainty_factors = {
            'position_uncertainty': np.sqrt(
                completeness_factor * sample_size_factor * mae_factor * 
                track_factor * weather_factor * session_uncertainty
            ),
            'dnf_uncertainty': consistency_factor * stability_factor * track_factor,
            'pace_uncertainty': mae_factor * weather_factor * track_factor,
            'strategy_uncertainty': track_factor * weather_factor,
            'safety_car_uncertainty': track_factor * 1.5,  # Always unpredictable
            
            # Individual components for debugging
            'data_quality_factor': completeness_factor * sample_size_factor,
            'model_quality_factor': mae_factor * stability_factor,
            'external_factor': track_factor * weather_factor,
            
            # Confidence intervals
            'confidence_level': min(0.95, max(0.5, 
                1.0 - (uncertainty_factors.get('position_uncertainty', 1.0) - 1.0) / 2.0
            ))
        }
        
        return uncertainty_factors

    def simulate_race_with_dynamic_blurriness(self, race_predictions, features, model_performance, 
                                            data_quality, track_category, num_simulations=1000):
        """
        Enhanced Monte Carlo simulation with dynamic blurriness based on prediction quality
        
        Adjusts randomness/uncertainty based on:
        - Data quality and completeness
        - Model performance and stability  
        - Track characteristics
        - Weather conditions
        - Sample size and historical consistency
        """
        if race_predictions.empty:
            return pd.DataFrame()
        
        print(f"🎲 Running {num_simulations} Monte Carlo simulations with dynamic uncertainty...")
        
        # Calculate uncertainty factors
        uncertainty = self.calculate_prediction_uncertainty(
            features, model_performance, data_quality, track_category
        )
        
        print(f"   📊 Uncertainty Analysis:")
        print(f"      Position Uncertainty: {uncertainty['position_uncertainty']:.2f}x")
        print(f"      Data Quality Factor: {uncertainty['data_quality_factor']:.2f}x")  
        print(f"      Model Quality Factor: {uncertainty['model_quality_factor']:.2f}x")
        print(f"      External Factors: {uncertainty['external_factor']:.2f}x")
        print(f"      Confidence Level: {uncertainty['confidence_level']:.1%}")
        
        simulation_results = []
        
        for sim in range(num_simulations):
            sim_result = race_predictions.copy()
            
            # Dynamic blurriness application
            for idx, row in sim_result.iterrows():
                driver = row['Driver']
                predicted_pos = row['Predicted_Finish']
                dnf_prob = row['DNF_Risk'] / 100
                
                # Get driver-specific factors
                driver_features = features[features['Driver'] == driver].iloc[0] if len(features[features['Driver'] == driver]) > 0 else None
                
                if driver_features is not None:
                    # Driver-specific uncertainty modifiers
                    consistency = driver_features.get('hist_consistency', 0.5)
                    recent_form = driver_features.get('recent_form', 10.0)
                    pace_rank = driver_features.get('race_pace', 0)
                    
                    # More consistent drivers have less uncertainty
                    driver_consistency_factor = 2.0 - consistency  # 1.0 to 2.0
                    
                    # Drivers in poor form have more uncertainty
                    form_factor = 1.0 + max(0, (recent_form - 10) / 20.0)  # 1.0 to 1.5
                    
                    # Pace uncertainty - faster drivers more predictable in some ways
                    if hasattr(pace_rank, 'rank'):
                        pace_uncertainty = 1.0 + (pace_rank / 20.0) * 0.3  # Slight increase for slower drivers
                    else:
                        pace_uncertainty = 1.0
                else:
                    driver_consistency_factor = 1.5
                    form_factor = 1.2
                    pace_uncertainty = 1.2
                
                # Combined uncertainty for this driver
                total_position_uncertainty = (
                    uncertainty['position_uncertainty'] * 
                    driver_consistency_factor * 
                    form_factor * 
                    pace_uncertainty
                )
                
                # Position blurriness - scales with uncertainty
                base_position_std = 1.8  # Base uncertainty in positions
                position_std = base_position_std * total_position_uncertainty
                
                # Apply different uncertainty ranges based on grid position
                if predicted_pos <= 3:
                    # Front runners - slightly less uncertainty in some conditions
                    position_std *= 0.85
                elif predicted_pos >= 15:
                    # Back markers - more uncertainty due to traffic, incidents
                    position_std *= 1.15
                
                # Generate position variation with dynamic blurriness
                position_variation = np.random.normal(0, position_std)
                
                # DNF simulation with blurred probability
                dnf_uncertainty_factor = uncertainty['dnf_uncertainty'] * driver_consistency_factor
                blurred_dnf_prob = dnf_prob * dnf_uncertainty_factor
                blurred_dnf_prob = np.clip(blurred_dnf_prob, 0, 0.35)  # Cap at 35%
                
                # Strategy/incident uncertainty
                strategy_chaos = np.random.normal(0, uncertainty['strategy_uncertainty'] * 0.5)
                
                # Safety car probability affects everyone
                safety_car_effect = 0
                if np.random.random() < 0.25:  # 25% chance of safety car affecting this driver
                    safety_car_effect = np.random.normal(0, uncertainty['safety_car_uncertainty'] * 1.2)
                
                # Weather micro-effects
                weather_effect = np.random.normal(0, uncertainty.get('weather_penalty', 0) * 2.0)
                
                # Combine all uncertainty sources
                total_position_change = (
                    position_variation + 
                    strategy_chaos + 
                    safety_car_effect + 
                    weather_effect
                )
                
                # Apply DNF check
                if np.random.random() < blurred_dnf_prob:
                    sim_result.loc[idx, 'Final_Position'] = 99  # DNF
                else:
                    new_position = predicted_pos + total_position_change
                    # Realistic bounds
                    new_position = max(1, min(20, new_position))
                    sim_result.loc[idx, 'Final_Position'] = new_position
            
            # Re-rank positions to ensure no ties (except DNFs)
            finished = sim_result[sim_result['Final_Position'] < 99].copy()
            if len(finished) > 0:
                finished['Final_Position'] = finished['Final_Position'].rank(method='first')
                sim_result.loc[finished.index, 'Final_Position'] = finished['Final_Position']
            
            simulation_results.append(sim_result[['Driver', 'Final_Position']].copy())
        
        # Aggregate results with uncertainty metrics
        return self._aggregate_blurred_simulations(simulation_results, uncertainty)

    def _aggregate_blurred_simulations(self, simulation_results, uncertainty_factors):
        """
        Aggregate Monte Carlo simulation results with uncertainty quantification
        """
        all_results = pd.concat(simulation_results, ignore_index=True)
        
        summary_data = []
        
        for driver in all_results['Driver'].unique():
            driver_results = all_results[all_results['Driver'] == driver]
            positions = driver_results['Final_Position']
            
            # Separate finished vs DNF
            dnfs = (positions == 99).sum()
            finished = positions[positions < 99]
            
            if len(finished) > 0:
                avg_pos = finished.mean()
                std_dev = finished.std()
                best_result = finished.min()
                worst_result = finished.max()
                
                # Uncertainty-adjusted confidence intervals
                confidence_level = uncertainty_factors.get('confidence_level', 0.8)
                alpha = 1 - confidence_level
                
                # Calculate percentiles for confidence intervals
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                pos_ci_lower = np.percentile(finished, lower_percentile)
                pos_ci_upper = np.percentile(finished, upper_percentile)
            else:
                avg_pos = 20
                std_dev = 0
                best_result = 20
                worst_result = 20
                pos_ci_lower = 20
                pos_ci_upper = 20
            
            driver_summary = {
                'Driver': driver,
                'Avg_Position': avg_pos,
                'Std_Dev': std_dev,
                'Best_Result': best_result,
                'Worst_Result': worst_result,
                'DNF_Rate': (dnfs / len(positions)) * 100,
                'Position_CI_Lower': pos_ci_lower,
                'Position_CI_Upper': pos_ci_upper,
                'Uncertainty_Level': uncertainty_factors['position_uncertainty'],
                'Confidence_Level': uncertainty_factors['confidence_level']
            }
            
            # Position probabilities with uncertainty awareness
            for pos in range(1, 21):
                prob = (finished == pos).sum() / len(simulation_results) * 100
                driver_summary[f'P{pos}_Prob'] = prob
            
            # Key probability milestones
            driver_summary['Win_Prob'] = driver_summary['P1_Prob']
            driver_summary['Podium_Prob'] = sum(driver_summary[f'P{pos}_Prob'] for pos in range(1, 4))
            driver_summary['Points_Prob'] = sum(driver_summary[f'P{pos}_Prob'] for pos in range(1, 11))
            
            summary_data.append(driver_summary)
        
        return pd.DataFrame(summary_data).sort_values('Avg_Position')

    def display_simulation_with_uncertainty(self, simulation_results, track_category):
        """
        Display simulation results with uncertainty visualization
        """
        if simulation_results.empty:
            return
        
        print(f"\n🎲 MONTE CARLO SIMULATION RESULTS")
        print("=" * 80)
        print(f"Track Category: {track_category.replace('_', ' ').title()}")
        print(f"Simulations: 1000 scenarios with dynamic uncertainty")
        
        print(f"\n{'Driver':<12} {'Avg':<6} {'±':<5} {'Best':<6} {'Worst':<7} {'DNF%':<6} {'CI Range':<12} {'Uncertainty':<12}")
        print("-" * 80)
        
        for idx, row in simulation_results.head(15).iterrows():
            uncertainty_desc = self._get_uncertainty_description(row['Uncertainty_Level'])
            ci_range = f"{row['Position_CI_Lower']:.1f}-{row['Position_CI_Upper']:.1f}"
            
            print(f"{row['Driver']:<12} {row['Avg_Position']:<6.1f} {row['Std_Dev']:<5.1f} "
                  f"{int(row['Best_Result']):<6} {int(row['Worst_Result']):<7} "
                  f"{row['DNF_Rate']:<6.1f} {ci_range:<12} {uncertainty_desc:<12}")
        
        # Probability insights
        print(f"\n🏆 WIN PROBABILITIES (Top 5):")
        print("-" * 40)
        win_probs = simulation_results.nlargest(5, 'Win_Prob')
        for idx, row in win_probs.iterrows():
            uncertainty_desc = self._get_uncertainty_description(row['Uncertainty_Level'])
            print(f"   {row['Driver']}: {row['Win_Prob']:.1f}% ({uncertainty_desc} confidence)")
        
        print(f"\n🥉 PODIUM PROBABILITIES (Top 8):")
        print("-" * 40)
        podium_probs = simulation_results.nlargest(8, 'Podium_Prob')
        for idx, row in podium_probs.iterrows():
            print(f"   {row['Driver']}: {row['Podium_Prob']:.1f}%")

    def _get_uncertainty_description(self, uncertainty_level):
        """Convert uncertainty level to human readable description"""
        if uncertainty_level < 0.8:
            return "Very High"
        elif uncertainty_level < 1.2:  
            return "High"
        elif uncertainty_level < 1.8:
            return "Medium"
        elif uncertainty_level < 2.5:
            return "Low"
        else:
            return "Very Low"

    # ...existing code...
def main():
    """
    Main function to run F1 Race Predictions
    
    This function provides an interactive way to generate F1 race predictions
    for any Grand Prix with proper error handling and user guidance.
    """
    print("🏎️  F1 RACE PREDICTOR")
    print("=" * 50)
    print("Welcome to the F1 Race Prediction System!")
    print("This tool analyzes practice session data to predict qualifying and race results.\n")
    
    try:
        # Initialize the predictor
        year = 2025  # Current season
        predictor = F1RacePredictor(year=year)
        
        # Available Grand Prix options
        available_gps = [
            'Australian', 'Chinese', 'Japanese', 'Bahrain', 'Saudi Arabian',
            'Miami', 'Emilia Romagna', 'Monaco', 'Canadian', 'Spanish',
            'Austrian', 'British', 'Belgian', 'Hungarian', 'Dutch',
            'Italian', 'Azerbaijan', 'Singapore', 'United States',
            'Mexico City', 'São Paulo', 'Las Vegas', 'Qatar', 'Abu Dhabi'
        ]
    
        
        
        selected_gps = ['Canadian']
        
        
        
        # Run predictions for selected Grand Prix
        print(f"\n🚀 Starting predictions for {len(selected_gps)} Grand Prix...")
        
        for gp_name in selected_gps:
            try:
                print(f"\n{'='*80}")
                print(f"🏁 PROCESSING: {gp_name.upper()} GRAND PRIX")
                print(f"{'='*80}")
                
                # Generate comprehensive predictions
                predictions = predictor.generate_full_report_v2(gp_name)
                
                if predictions:
                    print(f"\n✅ {gp_name} Grand Prix predictions completed successfully!")
                    
                    # Quick summary
                    if 'qualifying' in predictions and not predictions['qualifying'].empty:
                        pole_prediction = predictions['qualifying'].iloc[0]
                        print(f"🏆 Predicted Pole Position: {pole_prediction['Driver']}")
                        print(f"   Confidence: {pole_prediction['Confidence']:.1f}%")
                    
                    if 'race' in predictions and not predictions['race'].empty:
                        winner_prediction = predictions['race'].iloc[0]
                        print(f"🥇 Predicted Race Winner: {winner_prediction['Driver']}")
                        print(f"   Starting Position: P{winner_prediction['Grid_Position']}")
                        print(f"   Position Change: {winner_prediction['Positions_Change']:+d}")
                else:
                    print(f"❌ Failed to generate predictions for {gp_name} Grand Prix")
                    print("   This might be due to missing practice session data.")
                
            except Exception as e:
                print(f"❌ Error processing {gp_name} Grand Prix: {str(e)}")
                print("   Skipping to next race...")
                continue
        
        print(f"\n🎉 Prediction session completed!")
        print("📄 Check the generated CSV files for detailed results.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Prediction cancelled by user.")
        print("👋 Thank you for using the F1 Race Predictor!")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure you have a stable internet connection")
        print("2. Check that the selected Grand Prix has practice session data")
        print("3. Verify that all required packages are installed")
        print("   Run: pip install fastf1 pandas numpy scikit-learn xgboost")
        
    finally:
        print("\n" + "=" * 50)
        print("🏎️  F1 Race Predictor - Session Ended")
        print("=" * 50)


def demo_mode():
    """
    Demonstration mode showing the predictor's capabilities
    """
    print("🎮 DEMO MODE - F1 Race Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = F1RacePredictor(year=2025)
    
    # Demo with a specific race
    demo_races = ['Canadian', 'Monaco', 'Spanish']
    
    print("🔍 Demo: Analyzing recent Grand Prix races...")
    
    for race in demo_races:
        print(f"\n📊 Analyzing {race} Grand Prix...")
        try:
            # Quick analysis
            predictions = predictor.generate_full_report_v2(race)
            if predictions:
                print(f"✅ {race} GP analysis completed")
            else:
                print(f"⚠️  {race} GP: Limited data available")
        except Exception as e:
            print(f"❌ {race} GP analysis failed: {str(e)[:50]}...")
    
    print("\n🎯 Demo completed! Check the generated files for results.")


if __name__ == "__main__":
    """
    Entry point for the F1 Race Predictor script
    """
    import sys
    
    print("🏎️  F1 RACE PREDICTOR v2.0")
    print("=" * 60)
    print("Enhanced Monte Carlo Simulation with Dynamic Blurriness")
    print("=" * 60)
    
    # Check command line arguments for modes
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'demo':
            print("🎮 Running in Demo Mode...")
            demo_mode()
        elif mode == 'test':
            print("🧪 Running in Test Mode...")
            # Quick test with Canadian GP
            try:
                predictor = F1RacePredictor(2025)
                predictions = predictor.generate_full_report_v2('Canadian')
                if predictions and 'monte_carlo' in predictions:
                    print("✅ Enhanced Monte Carlo system working correctly!")
                    mc_results = predictions['monte_carlo']
                    print(f"📊 Simulated {len(mc_results)} drivers with dynamic uncertainty")
                else:
                    print("⚠️ Monte Carlo results not found in predictions")
            except Exception as e:
                print(f"❌ Test failed: {e}")
        elif mode in ['canadian', 'monaco', 'spanish', 'bahrain', 'austria']:
            # Quick prediction for specific GP
            try:
                predictor = F1RacePredictor(2025)
                gp_name = mode.title()
                print(f"🏁 Running {gp_name} Grand Prix prediction...")
                predictions = predictor.generate_full_report_v2(gp_name)
                if predictions:
                    print(f"✅ {gp_name} GP predictions completed!")
                else:
                    print(f"❌ Failed to generate {gp_name} GP predictions")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: demo, test, canadian, monaco, spanish")
    else:
        # Default: Run interactive main function
        print("🚀 Starting Interactive Mode...")
        main()