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
            return pd.DataFrame(), pd.DataFrame()
        laps = pd.concat(all_laps)
        quali_runs = laps.groupby('Driver')['LapTime'].min().reset_index()
        quali_runs.rename(columns={'LapTime': 'BestLap'}, inplace=True)
        race_runs = laps.groupby('Driver')['LapTime'].mean().reset_index()
        race_runs.rename(columns={'LapTime': 'AvgLap'}, inplace=True)
        return quali_runs, race_runs

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    def prepare_features_for_ml(self, quali_runs, race_runs):
        if quali_runs.empty:
            return pd.DataFrame()
        features = quali_runs.merge(race_runs, on='Driver', how='left')
        features['AvgLap'].fillna(features['BestLap'], inplace=True)
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
        features['Confidence'] = 100 - features['gap_to_fastest'].rank(pct=True) * 50
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
        score = features['Grid'] * 0.4 + features['pace_rank'] * 0.6
        features['Predicted_Finish'] = score.rank().astype(int)
        return features[['Driver', 'Predicted_Finish', 'Grid']].sort_values('Predicted_Finish')

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    def simulate_race(self, race_predictions, num_sim=1000):
        if race_predictions.empty:
            return pd.DataFrame()
        results = []
        drivers = race_predictions['Driver'].tolist()
        base_pos = race_predictions['Predicted_Finish'].values.astype(float)
        for _ in range(num_sim):
            noise = np.random.normal(0, 1.5, size=len(base_pos))
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
        quali_runs, race_runs = self.extract_weekend_features(gp_name)
        features = self.prepare_features_for_ml(quali_runs, race_runs)
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
