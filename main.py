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
            if isinstance(laps, list):  # fallback for dummy session
                return pd.DataFrame(
                    columns=[
                        'Driver',
                        'Team',
                        'LapTime',
                    'Sector1Time',
                    'Sector2Time',
                    'Sector3Time',
                    'LapNumber',
                    'TyreLife',
                    'Compound',
                ]
            )
            laps = laps[laps['LapTime'].notna()]
            laps = laps[laps['PitOutTime'].isna()]
            return laps[
                [
                    'Driver',
                    'Team',
                    'LapTime',
                    'Sector1Time',
                    'Sector2Time',
                    'Sector3Time',
                    'LapNumber',
                    'TyreLife',
                    'Compound',
                ]
            ]
        except Exception as exc:  # pragma: no cover - network issues
            print(f"Error loading {session_type}: {exc}")
            return pd.DataFrame(
                columns=[
                    'Driver',
                    'Team',
                    'LapTime',
                    'Sector1Time',
                    'Sector2Time',
                    'Sector3Time',
                    'LapNumber',
                    'TyreLife',
                    'Compound',
                ]
            )

    def extract_weekend_features(self, gp_name):
        """Separate qualifying-style and race-style laps from practice data."""
        all_laps = []
        year = self.year if self.year <= datetime.now().year else datetime.now().year
        for session in ['FP1', 'FP2', 'FP3']:
            laps = self.collect_practice_data(year, gp_name, session)
            if not laps.empty:
                all_laps.append(laps)
        if not all_laps:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        laps = pd.concat(all_laps)
        team_map = laps[['Driver', 'Team']].drop_duplicates()
        # Determine a per-driver cutoff to split fast laps from long runs
        qcut = laps.groupby('Driver')['LapTime'].transform(lambda x: x.quantile(0.25))
        laps['is_quali'] = laps['LapTime'] <= qcut

        quali_runs = (
            laps[laps['is_quali']]
            .groupby('Driver')['LapTime']
            .min()
            .reset_index(name='BestLap')
            .merge(team_map, on='Driver', how='left')
        )

        race_laps = laps[~laps['is_quali']]
        if race_laps.empty:
            race_laps = laps
        race_runs = (
            race_laps.groupby('Driver')['LapTime']
            .mean()
            .reset_index(name='AvgLap')
            .merge(team_map, on='Driver', how='left')
        )

        def _deg_slope(df):
            if df['TyreLife'].nunique() <= 1:
                return 0.0
            x = df['TyreLife'].fillna(pd.Series(range(len(df)), index=df.index))
            y = df['LapTime'].dt.total_seconds()
            A = np.vstack([x, np.ones(len(x))]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return m

        deg = (
            race_laps.groupby('Driver')
            .apply(_deg_slope)
            .reset_index(name='TyreDeg')
            .merge(team_map, on='Driver', how='left')
        )

        stats = (
            laps.groupby('Driver')['LapTime']
            .agg(['median', 'std', 'count'])
            .reset_index()
            .rename(columns={'median': 'MedianLap', 'std': 'LapStd', 'count': 'NumLaps'})
            .merge(team_map, on='Driver', how='left')
        )

        team_quali = (
            laps[laps['is_quali']]
            .groupby('Team')['LapTime']
            .min()
            .reset_index(name='TeamBestLap')
        )

        team_theoretical = (
            laps.groupby('Team')[['Sector1Time', 'Sector2Time', 'Sector3Time']]
            .min()
            .sum(axis=1)
            .reset_index(name='TeamTheoreticalBest')
        )

        team_stats = (
            laps.groupby('Team')['LapTime']
            .agg(['median', 'std', 'count'])
            .reset_index()
            .rename(
                columns={
                    'median': 'TeamMedianLap',
                    'std': 'TeamLapStd',
                    'count': 'TeamNumLaps',
                }
            )
        )

        teams = (
            team_quali.merge(team_theoretical, on='Team', how='outer')
            .merge(team_stats, on='Team', how='outer')
        )

        return quali_runs, race_runs, stats, teams, deg

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    def prepare_features_for_ml(self, quali_runs, race_runs, stats, team_data, deg_data):
        """Combine all aggregates into a single feature table."""
        if quali_runs.empty:
            return pd.DataFrame()

        features = quali_runs.merge(race_runs, on=['Driver', 'Team'], how='left')
        features = features.merge(stats, on=['Driver', 'Team'], how='left')
        features = features.merge(team_data, on='Team', how='left')
        features = features.merge(deg_data[['Driver', 'TyreDeg']], on='Driver', how='left')

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

        team_cols = [
            'TeamBestLap',
            'TeamTheoreticalBest',
            'TeamMedianLap',
            'TeamLapStd',
            'TeamNumLaps',
        ]
        for c in team_cols:
            if c in features:
                if features[c].isna().all():
                    features[c].fillna(features[c].median() if features[c].dtype != object else 0, inplace=True)
                else:
                    features[c].fillna(features[c].median(), inplace=True)
            else:
                features[c] = 0

        if 'TyreDeg' not in features:
            features['TyreDeg'] = 0.0
        else:
            if features['TyreDeg'].isna().all():
                features['TyreDeg'].fillna(0.0, inplace=True)
            else:
                features['TyreDeg'].fillna(features['TyreDeg'].median(), inplace=True)

        fastest = features['BestLap'].min()
        features['gap_to_fastest'] = features['BestLap'] - fastest
        features['team_gap'] = features['TeamTheoreticalBest'] - features['BestLap']
        features['driver_consistency'] = (
            features['LapStd'].dt.total_seconds() / features['AvgLap'].dt.total_seconds()
        ).fillna(0)
        features['team_consistency'] = (
            features['TeamLapStd'].dt.total_seconds() / features['TeamMedianLap'].dt.total_seconds()
        ).fillna(0)
        return features

    # ------------------------------------------------------------------
    # Qualifying prediction
    # ------------------------------------------------------------------
    def predict_qualifying(self, features):
        if features.empty:
            return pd.DataFrame()
        # Factor in team pace and consistency
        features['qual_score'] = (
            features['BestLap'].dt.total_seconds()
            + 0.05 * features['LapStd'].dt.total_seconds()
            + 0.02 * features['TeamLapStd'].dt.total_seconds()
            + 0.01 * features['team_gap'].dt.total_seconds()
            + 0.1 * features['driver_consistency']
            + 0.05 * features['team_consistency']
        )
        features = features.sort_values('qual_score')
        features['Predicted_Position'] = range(1, len(features) + 1)

        base_conf = 1 - ((features["qual_score"] - features["qual_score"].min()) / (features["qual_score"].max() - features["qual_score"].min() + 1e-6))
        lap_factor = (features["NumLaps"] / features["NumLaps"].max()).clip(lower=0.5)
        team_factor = 1 - (features["TeamLapStd"] / features["TeamLapStd"].max())
        driver_consistency = (features["LapStd"].dt.total_seconds() / features["AvgLap"].dt.total_seconds()).fillna(0)
        consistency_factor = 1 - driver_consistency
        prob = base_conf * lap_factor * team_factor * consistency_factor
        features['Confidence'] = (100 / (1 + np.exp(-5 * (prob - 0.5)))).round(2)

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
        features['team_pace_rank'] = features['TeamTheoreticalBest'].rank()
        features['team_consistency_rank'] = features['TeamLapStd'].rank()
        features['degradation_rank'] = features['TyreDeg'].rank()
        features['driver_consistency'] = (
            features['LapStd'].dt.total_seconds() / features['AvgLap'].dt.total_seconds()
        ).fillna(0)
        features['driver_consistency_rank'] = features['driver_consistency'].rank()

        score = (
            features['Grid'] * 0.25
            + features['pace_rank'] * 0.25
            + features['consistency_rank'] * 0.1
            + features['team_pace_rank'] * 0.15
            + features['team_consistency_rank'] * 0.05
            + features['laps_rank'] * 0.05
            + features['degradation_rank'] * 0.1
            + features['driver_consistency_rank'] * 0.05
        )

        features['Predicted_Finish'] = score.rank().astype(int)

        return features[
            [
                'Driver',
                'Predicted_Finish',
                'Grid',
                'LapStd',
                'NumLaps',
                'TeamLapStd',
                'TeamTheoreticalBest',
                'TyreDeg',
                'driver_consistency',
            ]
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
        team_std = race_predictions.get('TeamLapStd', pd.Series([pd.Timedelta(0)] * len(base_pos)))
        degr = race_predictions.get('TyreDeg', pd.Series([0.0] * len(base_pos)))
        consistency = race_predictions.get('driver_consistency', pd.Series([0.0] * len(base_pos)))

        # Convert timedeltas to seconds for scaling
        lap_std_sec = lap_std.dt.total_seconds() if hasattr(lap_std, 'dt') else lap_std
        team_std_sec = team_std.dt.total_seconds() if hasattr(team_std, 'dt') else team_std
        max_std = lap_std_sec.max() if lap_std_sec.max() != 0 else 1
        max_team_std = team_std_sec.max() if team_std_sec.max() != 0 else 1
        max_laps = num_laps.max() if num_laps.max() != 0 else 1
        max_deg = abs(degr).max() if abs(degr).max() != 0 else 1
        max_cons = consistency.max() if consistency.max() != 0 else 1

        std_factor = lap_std_sec / max_std
        laps_factor = 1 - (num_laps / max_laps)
        team_factor = team_std_sec / max_team_std
        deg_factor = abs(degr) / max_deg
        cons_factor = consistency / max_cons
        noise_scale = 1.0 + std_factor + laps_factor + team_factor + deg_factor + cons_factor

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

    def calibrate_confidence(self, gp_name, quali_df):
        """Calibrate confidence scores using previous year's qualifying data."""
        try:
            prev = fastf1.get_session(self.year - 1, gp_name, 'Q')
            prev.load()
            res = prev.results[['Abbreviation', 'Position']]
            res.rename(columns={'Abbreviation': 'Driver', 'Position': 'ActualPos'}, inplace=True)
            merged = quali_df.merge(res, on='Driver')
            if len(merged) >= 5:
                from sklearn.isotonic import IsotonicRegression
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(merged['Predicted_Position'], merged['ActualPos'])
                adj = iso.transform(quali_df['Predicted_Position'])
                quali_df['Confidence'] = (1 - (adj - 1) / (quali_df['Predicted_Position'].max())) * 100
        except Exception as exc:  # pragma: no cover - network issues
            print(f'Calibration failed: {exc}')
        return quali_df

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------
    def run_prediction(self, gp_name):
        quali_runs, race_runs, stats, teams, deg = self.extract_weekend_features(gp_name)
        features = self.prepare_features_for_ml(quali_runs, race_runs, stats, teams, deg)
        quali_pred = self.predict_qualifying(features)
        quali_pred = self.calibrate_confidence(gp_name, quali_pred)
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
