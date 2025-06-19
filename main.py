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
        all_laps = []
        year = self.year if self.year <= datetime.now().year else datetime.now().year
        for session in ['FP1', 'FP2', 'FP3']:
            laps = self.collect_practice_data(year, gp_name, session)
            if not laps.empty:
                all_laps.append(laps)
        if not all_laps:
            return (pd.DataFrame(), pd.DataFrame(),
                    pd.DataFrame(), pd.DataFrame())
        laps = pd.concat(all_laps)
        quali_runs = laps.groupby('Driver')['LapTime'].min().reset_index()
        quali_runs.rename(columns={'LapTime': 'BestLap'}, inplace=True)
        race_runs = laps.groupby('Driver')['LapTime'].mean().reset_index()
        race_runs.rename(columns={'LapTime': 'AvgLap'}, inplace=True)
        lap_counts = laps.groupby('Driver')['LapTime'].count().reset_index()
        lap_counts.rename(columns={'LapTime': 'LapCount'}, inplace=True)
        lap_std = laps.groupby('Driver')['LapTime'].std().reset_index()
        lap_std.rename(columns={'LapTime': 'LapSTD'}, inplace=True)
        return quali_runs, race_runs, lap_counts, lap_std

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    def prepare_features_for_ml(self, quali_runs, race_runs, lap_counts, lap_std):
        if quali_runs.empty:
            return pd.DataFrame()
        features = quali_runs.merge(race_runs, on='Driver', how='left')
        features = features.merge(lap_counts, on='Driver', how='left')
        features = features.merge(lap_std, on='Driver', how='left')
        features['AvgLap'].fillna(features['BestLap'], inplace=True)
        features['LapCount'].fillna(0, inplace=True)
        features['LapSTD'].fillna(pd.Timedelta(0), inplace=True)
        fastest = features['BestLap'].min()
        features['gap_to_fastest'] = features['BestLap'] - fastest
        return features

    # ------------------------------------------------------------------
    # Qualifying prediction
    # ------------------------------------------------------------------
    def predict_qualifying(self, features):
        if features.empty:
            return pd.DataFrame()
        features = features.sort_values('BestLap')
        features['Predicted_Position'] = range(1, len(features) + 1)
        max_std = features['LapSTD'].dt.total_seconds().max() or 1
        conf = 100 - features['gap_to_fastest'].rank(pct=True) * 40
        conf -= (features['LapSTD'].dt.total_seconds() / max_std) * 10
        features['Confidence'] = conf.clip(0, 100)
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
        features['consistency_rank'] = features['LapSTD'].rank()
        score = (features['Grid'] * 0.4 +
                features['pace_rank'] * 0.5 +
                features['consistency_rank'] * 0.1)
        features['Predicted_Finish'] = score.rank().astype(int)
        return features[['Driver', 'Predicted_Finish', 'Grid', 'LapSTD']].sort_values('Predicted_Finish')

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    def simulate_race(self, race_predictions, num_sim=1000):
        if race_predictions.empty:
            return pd.DataFrame()
        results = []
        drivers = race_predictions['Driver'].tolist()
        base_pos = race_predictions['Predicted_Finish'].values.astype(float)
        variability = race_predictions['LapSTD'].fillna(pd.Timedelta(0)).dt.total_seconds()
        max_var = variability.max() if variability.max() > 0 else 1
        variability = variability / max_var
        for _ in range(num_sim):
            noise = np.random.normal(0, 1.5 + variability)
            pos = np.clip(base_pos + noise, 1, 20)
            order = pd.Series(pos).rank().astype(int)
            results.append(order)
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
        quali_runs, race_runs, lap_counts, lap_std = self.extract_weekend_features(gp_name)
        features = self.prepare_features_for_ml(quali_runs, race_runs, lap_counts, lap_std)
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
