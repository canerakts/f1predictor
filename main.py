import os
import warnings
from datetime import datetime

try:
    import fastf1
except ModuleNotFoundError:  # pragma: no cover - fallback for tests
    from types import SimpleNamespace
    class _DummySession:
        laps = []
        def load(self):
            pass
    fastf1 = SimpleNamespace(get_session=lambda *a, **k: _DummySession(),
                             Cache=SimpleNamespace(enable_cache=lambda p: None))

try:
    import pandas as pd
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    from types import SimpleNamespace
    pd = SimpleNamespace(DataFrame=lambda *a, **k: [])
    class _DummyNP:
        def __getattr__(self, name):
            raise ImportError('numpy is required')
    np = _DummyNP()

warnings.filterwarnings('ignore')

cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

class F1RacePredictor:
    """Minimal predictor focusing on qualifying, race prediction and simulations."""
    def __init__(self, year=2025):
        self.year = year

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_practice_data(self, year, gp_name, session_type):
        try:
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()
            laps = session.laps
            laps = laps[laps['LapTime'].notna()]
            laps = laps[laps['PitOutTime'].isna()]
            return laps[['Driver', 'LapTime']]
        except Exception as exc:  # pragma: no cover - network issues
            print(f"Error loading {session_type}: {exc}")
            return pd.DataFrame(columns=['Driver', 'LapTime'])

    def extract_weekend_features(self, gp_name):
        """Return lap aggregates separating quali-style and race-style runs."""
        all_laps = []
        year = self.year if self.year <= datetime.now().year else datetime.now().year
        for session in ['FP1', 'FP2', 'FP3']:
            laps = self.collect_practice_data(year, gp_name, session)
            if not laps.empty:
                all_laps.append(laps)
        if not all_laps:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        laps = pd.concat(all_laps)

        # Split laps into "qualifying" style and "race" style
        driver_best = laps.groupby('Driver')['LapTime'].transform('min')
        quali_mask = laps['LapTime'] <= driver_best * 1.05
        quali_laps = laps[quali_mask]
        race_laps = laps[~quali_mask]

        if quali_laps.empty:
            quali_laps = laps
        if race_laps.empty:
            race_laps = laps

        quali_runs = (
            quali_laps.groupby('Driver')['LapTime'].min().reset_index()
        )
        quali_runs.rename(columns={'LapTime': 'BestLap'}, inplace=True)

        race_runs = (
            race_laps.groupby('Driver')['LapTime'].mean().reset_index()
        )
        race_runs.rename(columns={'LapTime': 'AvgLap'}, inplace=True)

        stats = race_laps.groupby('Driver')['LapTime'].agg(['median', 'std', 'count']).reset_index()
        stats.rename(columns={'median': 'MedianLap', 'std': 'LapStd', 'count': 'NumLaps'}, inplace=True)

        return quali_runs, race_runs, stats

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    def prepare_features_for_ml(self, quali_runs, race_runs, stats):
        """Combine all aggregates into a single feature table."""
        if quali_runs.empty:
            return pd.DataFrame()

        features = quali_runs.merge(race_runs, on='Driver', how='left')
        features = features.merge(stats, on='Driver', how='left')

        # Fallbacks for missing data
        features['AvgLap'].fillna(features['BestLap'], inplace=True)
        features['MedianLap'].fillna(features['BestLap'], inplace=True)
        if 'LapStd' in features:
            if features['LapStd'].isna().all():
                features['LapStd'].fillna(0, inplace=True)
            else:
                features['LapStd'].fillna(features['LapStd'].median(), inplace=True)
        else:
            features['LapStd'] = 0
        features['NumLaps'].fillna(0, inplace=True)

        fastest = features['BestLap'].min()
        features['gap_to_fastest'] = features['BestLap'] - fastest
        return features

    # ------------------------------------------------------------------
    # Qualifying prediction
    # ------------------------------------------------------------------
    def predict_qualifying(self, features):
        if features.empty:
            return pd.DataFrame()
        # Small penalty for drivers with larger lap standard deviation
        features['qual_score'] = features['BestLap'] + 0.05 * features['LapStd']
        features = features.sort_values('qual_score')
        features['Predicted_Position'] = range(1, len(features) + 1)

        base_conf = 100 - features['qual_score'].rank(pct=True) * 50
        lap_factor = features['NumLaps'] / features['NumLaps'].max()
        lap_factor = lap_factor.clip(lower=0.5)
        features['Confidence'] = (base_conf * lap_factor).round(2)

        return features[['Driver', 'Predicted_Position', 'Confidence']]

    # ------------------------------------------------------------------
    # Race prediction
    # ------------------------------------------------------------------
    def predict_race(self, features, quali_predictions):
        if features.empty or quali_predictions.empty:
            return pd.DataFrame()
        grid = dict(zip(quali_predictions['Driver'], quali_predictions['Predicted_Position']))
        features['Grid'] = features['Driver'].map(grid).fillna(20)

        features['pace_rank'] = features['AvgLap'].rank()
        features['consistency_rank'] = features['LapStd'].rank()
        features['laps_rank'] = features['NumLaps'].rank(ascending=False)

        score = (
            features['Grid'] * 0.4
            + features['pace_rank'] * 0.3
            + features['consistency_rank'] * 0.2
            + features['laps_rank'] * 0.1
        )

        features['Predicted_Finish'] = score.rank().astype(int)

        return features[
            ['Driver', 'Predicted_Finish', 'Grid', 'LapStd', 'NumLaps']
        ].sort_values('Predicted_Finish')

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    def simulate_race(self, race_predictions, num_sim=1000):
        """Monte Carlo simulation with uncertainty scaled by data quality."""
        if race_predictions.empty:
            return pd.DataFrame()

        drivers = race_predictions['Driver'].tolist()
        base_pos = race_predictions['Predicted_Finish'].astype(float).values

        lap_std = race_predictions.get('LapStd', pd.Series([pd.Timedelta(0)] * len(base_pos)))
        num_laps = race_predictions.get('NumLaps', pd.Series([0] * len(base_pos)))

        # Convert timedeltas to seconds for scaling
        lap_std_sec = lap_std.dt.total_seconds() if hasattr(lap_std, 'dt') else lap_std
        max_std = lap_std_sec.max() if lap_std_sec.max() != 0 else 1
        max_laps = num_laps.max() if num_laps.max() != 0 else 1

        std_factor = lap_std_sec / max_std
        laps_factor = 1 - (num_laps / max_laps)
        noise_scale = 1.0 + std_factor + laps_factor

        results = []
        for _ in range(num_sim):
            noise = np.random.normal(0, noise_scale, size=len(base_pos))
            pos = np.clip(base_pos + noise, 1, 20)
            results.append(pd.Series(pos).rank().astype(int).values)

        sim_df = pd.DataFrame(results, columns=drivers)
        summary = []
        for d in drivers:
            positions = sim_df[d]
            summary.append({
                'Driver': d,
                'Avg_Position': positions.mean(),
                'Std_Dev': positions.std(),
                'Win_Prob': (positions == 1).mean() * 100,
                'Points_Prob': (positions <= 10).mean() * 100,
            })
        return pd.DataFrame(summary).sort_values('Avg_Position')

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------
    def run_prediction(self, gp_name):
        quali_runs, race_runs, stats = self.extract_weekend_features(gp_name)
        features = self.prepare_features_for_ml(quali_runs, race_runs, stats)
        quali_pred = self.predict_qualifying(features)
        race_pred = self.predict_race(features, quali_pred)
        monte_carlo = self.simulate_race(race_pred)
        return {
            'qualifying': quali_pred,
            'race': race_pred,
            'monte_carlo': monte_carlo,
        }


def main():
    predictor = F1RacePredictor()
    preds = predictor.run_prediction('Canadian')
    if preds['qualifying'].empty:
        print('No data available')
        return
    print('\nQUALIFYING PREDICTION')
    print(preds['qualifying'].to_string(index=False))
    print('\nRACE PREDICTION')
    print(preds['race'].to_string(index=False))
    print('\nMONTE CARLO SUMMARY')
    print(preds['monte_carlo'].head().to_string(index=False))


if __name__ == '__main__':
    main()
