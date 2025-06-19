#!/usr/bin/env python3
"""Simple test script for Monte Carlo blurriness functionality"""

import pytest

# Skip tests when pandas is unavailable
pytest.importorskip("pandas")

print("🎲 Testing Enhanced Monte Carlo with Dynamic Blurriness...")

try:
    from test import F1RacePredictor
    print("✅ Import successful")
    
    predictor = F1RacePredictor(2025)
    print("✅ Predictor initialized")
    
    # Test the uncertainty calculation method
    import pandas as pd
    import numpy as np
    
    # Create dummy features for testing
    features = pd.DataFrame({
        'Driver': ['VER', 'HAM', 'LEC', 'NOR', 'PIA'],
        'hist_consistency': [0.8, 0.7, 0.6, 0.75, 0.65],
        'recent_form': [5.0, 8.0, 12.0, 6.0, 10.0],
        'race_pace': [90.5, 90.8, 91.2, 90.6, 91.0]
    })
    
    # Test uncertainty calculation
    print("\n🔍 Testing uncertainty calculation...")
    uncertainty = predictor.calculate_prediction_uncertainty(
        features=features,
        model_performance=None,
        data_quality={'completeness': 0.85, 'sample_size': 15, 'weather_penalty': 0.1},
        track_category='medium'
    )
    
    print("✅ Uncertainty calculation works!")
    print(f"📊 Position uncertainty: {uncertainty['position_uncertainty']:.2f}x")
    print(f"📊 DNF uncertainty: {uncertainty['dnf_uncertainty']:.2f}x")
    print(f"📊 Confidence level: {uncertainty['confidence_level']:.1%}")
    print(f"📊 Data quality factor: {uncertainty['data_quality_factor']:.2f}x")
    print(f"📊 Model quality factor: {uncertainty['model_quality_factor']:.2f}x")
    print(f"📊 External factors: {uncertainty['external_factor']:.2f}x")
    
    # Test with different track categories
    print("\n🏁 Testing different track categories...")
    for track_cat in ['very_low', 'low', 'medium', 'high', 'very_high']:
        uncertainty = predictor.calculate_prediction_uncertainty(
            features=features,
            model_performance=None,
            data_quality={'completeness': 0.85, 'sample_size': 15, 'weather_penalty': 0.0},
            track_category=track_cat
        )
        print(f"   {track_cat:10}: {uncertainty['position_uncertainty']:.2f}x uncertainty")
    
    # Test with different weather conditions
    print("\n🌧️ Testing weather impact...")
    for weather_penalty in [0.0, 0.2, 0.5, 1.0]:
        uncertainty = predictor.calculate_prediction_uncertainty(
            features=features,
            model_performance=None,
            data_quality={'completeness': 0.85, 'sample_size': 15, 'weather_penalty': weather_penalty},
            track_category='medium'
        )
        print(f"   Weather {weather_penalty:.1f}: {uncertainty['position_uncertainty']:.2f}x uncertainty")
    
    # Test dummy Monte Carlo simulation
    print("\n🎲 Testing Monte Carlo simulation...")
    race_predictions = pd.DataFrame({
        'Driver': ['VER', 'HAM', 'LEC', 'NOR', 'PIA'],
        'Predicted_Finish': [1, 3, 5, 2, 4],
        'DNF_Risk': [5.0, 8.0, 12.0, 6.0, 10.0]
    })
    
    mc_results = predictor.simulate_race_with_dynamic_blurriness(
        race_predictions=race_predictions,
        features=features,
        model_performance=None,
        data_quality={'completeness': 0.85, 'sample_size': 15, 'weather_penalty': 0.1},
        track_category='medium',
        num_simulations=100  # Small number for testing
    )
    
    print("✅ Monte Carlo simulation works!")
    print(f"📊 Simulation results for top drivers:")
    for idx, row in mc_results.head(3).iterrows():
        print(f"   {row['Driver']}: Avg={row['Avg_Position']:.1f}, "
              f"P1={row['P1_Prob']:.1f}%, DNF={row['DNF_Rate']:.1f}%")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
