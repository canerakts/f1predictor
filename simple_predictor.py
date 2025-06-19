import pandas as pd
import numpy as np

class F1RacePredictor:
    """Simplified predictor focusing on qualifying, race result and Monte Carlo."""

    def __init__(self, year=2025):
        self.year = year
        self.track_categories = {
            'very_low': ['Monaco'],
            'low': ['Spanish'],
            'medium': ['Canadian'],
            'high': ['Austrian'],
            'very_high': ['Bahrain']
        }

    # ------------------------------------------------------------------
    # Basic data stubs
    # ------------------------------------------------------------------
    def extract_weekend_features(self, gp_name):
        """Return dummy practice data."""
        drivers = ['VER', 'HAM', 'NOR', 'LEC', 'SAI']
        quali = pd.DataFrame({
            'Driver': drivers,
            'LapTime': np.linspace(88.0, 92.0, len(drivers))
        })
        race = pd.DataFrame({
            'Driver': drivers,
            'AvgLapTime': np.linspace(90.0, 94.0, len(drivers))
        })
        return quali, race

    def load_historical_data(self, gp_name, num_recent_races=5):
        """Return simple historical stats for demo purposes."""
        base = ['VER', 'HAM', 'NOR', 'LEC', 'SAI']
        stats = {}
        for i, drv in enumerate(base):
            stats[drv] = {
                'avg_race_pos': i + 1,
                'consistency': 0.8,
                'dnf_probability': 0.05,
                'recent_form': i + 1,
                'momentum': 0.0,
                'trend': 0.0,
            }
        return stats

    def prepare_features_for_ml(self, quali_runs, race_runs, historical_stats, gp_name=None):
        """Minimal feature preparation."""
        features = quali_runs.copy()
        fastest = features['LapTime'].min()
        features['gap_to_fastest'] = features['LapTime'] - fastest
        features['race_pace'] = race_runs['AvgLapTime']
        for drv in features['Driver']:
            stats = historical_stats.get(drv, {
                'avg_race_pos': 10,
                'consistency': 0.5,
                'dnf_probability': 0.1,
                'recent_form': 10,
                'momentum': 0.0,
                'trend': 0.0,
            })
            features.loc[features['Driver'] == drv, 'hist_avg_finish'] = stats['avg_race_pos']
            features.loc[features['Driver'] == drv, 'hist_consistency'] = stats['consistency']
            features.loc[features['Driver'] == drv, 'hist_dnf_rate'] = stats['dnf_probability']
        return features

    # ------------------------------------------------------------------
    # Simple prediction methods
    # ------------------------------------------------------------------
    def predict_qualifying_with_ml(self, features):
        """Predict qualifying order based on best lap."""
        result = features[['Driver', 'gap_to_fastest']].copy()
        result['Predicted_Position'] = result['gap_to_fastest'].rank().astype(int)
        result['Confidence'] = 90 - result['Predicted_Position'] * 3
        result['Position_Range'] = result['Predicted_Position'].apply(lambda x: f"{max(1, x-1)}-{min(20, x+1)}")
        result['Q3_Probability'] = np.clip(100 - result['Predicted_Position'] * 5, 0, 100)
        result['Pole_Probability'] = np.clip(50 - (result['Predicted_Position']-1)*10, 0, 100)
        return result.sort_values('Predicted_Position')

    def predict_race_with_enhanced_ml(self, features, quali_pred, gp_name):
        """Predict race results using grid position and race pace."""
        merged = features.merge(quali_pred[['Driver', 'Predicted_Position']], on='Driver')
        merged['Grid_Position'] = merged['Predicted_Position']
        pace_rank = merged['race_pace'].rank()
        merged['Predicted_Finish'] = (0.6 * merged['Grid_Position'] + 0.4 * pace_rank).rank().astype(int)
        merged['Positions_Change'] = merged['Grid_Position'] - merged['Predicted_Finish']
        merged['DNF_Risk'] = merged['hist_dnf_rate'] * 100
        merged['Points_Probability'] = np.where(merged['Predicted_Finish'] <= 10, 80, 20)
        merged['Race_Pace_Rank'] = pace_rank.astype(int)
        merged['ML_Race_Score'] = 1 / merged['Predicted_Finish']
        return merged[['Driver', 'Grid_Position', 'Predicted_Finish', 'Positions_Change',
                      'DNF_Risk', 'Points_Probability', 'Race_Pace_Rank', 'ML_Race_Score']].sort_values('Predicted_Finish')

    # ------------------------------------------------------------------
    # Monte Carlo utilities
    # ------------------------------------------------------------------
    def calculate_prediction_uncertainty(self, features, model_performance, data_quality, track_category):
        """Return simple uncertainty metrics."""
        completeness = data_quality.get('completeness', 1.0)
        sample_size = data_quality.get('sample_size', len(features))
        base = max(0.5, 25 / sample_size)
        position_uncertainty = (2.0 - completeness) * base
        return {
            'position_uncertainty': max(0.5, position_uncertainty),
            'dnf_uncertainty': 1.0,
            'data_quality_factor': completeness,
            'model_quality_factor': 1.0,
            'external_factor': 1.0,
            'confidence_level': 0.8,
        }

    def simulate_race_with_dynamic_blurriness(self, race_predictions, features, model_performance,
                                              data_quality, track_category, num_simulations=100):
        if race_predictions.empty:
            return pd.DataFrame()
        uncertainty = self.calculate_prediction_uncertainty(features, model_performance, data_quality, track_category)
        results = []
        for _ in range(num_simulations):
            positions = []
            for pos in race_predictions['Predicted_Finish']:
                noise = np.random.normal(0, uncertainty['position_uncertainty'])
                positions.append(max(1, min(20, pos + noise)))
            results.append(positions)
        sim_df = pd.DataFrame(results, columns=race_predictions['Driver'])
        summary = []
        for drv in sim_df.columns:
            pos = sim_df[drv]
            summary.append({
                'Driver': drv,
                'Avg_Position': pos.mean(),
                'P1_Prob': (pos == 1).mean() * 100,
                'DNF_Rate': 0.0,
            })
        return pd.DataFrame(summary).sort_values('Avg_Position')

    # ------------------------------------------------------------------
    # High level report
    # ------------------------------------------------------------------
    def generate_full_report_v2(self, gp_name):
        quali, race = self.extract_weekend_features(gp_name)
        if quali.empty:
            return None
        hist = self.load_historical_data(gp_name)
        features = self.prepare_features_for_ml(quali, race, hist, gp_name)
        quali_pred = self.predict_qualifying_with_ml(features)
        race_pred = self.predict_race_with_enhanced_ml(features, quali_pred, gp_name)
        mc = self.simulate_race_with_dynamic_blurriness(
            race_pred, features, None,
            {'completeness': 1.0, 'sample_size': len(features)}, 'medium', 100)
        return {'qualifying': quali_pred, 'race': race_pred, 'monte_carlo': mc}
