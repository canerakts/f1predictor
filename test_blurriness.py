#!/usr/bin/env python3
"""Test script for the enhanced Monte Carlo simulation with dynamic blurriness"""

import pytest

# Skip tests when pandas is unavailable
pytest.importorskip("pandas")

try:
    print("🎲 Testing Enhanced Monte Carlo with Dynamic Blurriness...")
    from test import F1RacePredictor
    print("✅ Import successful")
    
    predictor = F1RacePredictor(2025)
    print("✅ Predictor initialized")
    
    print("🏁 Running Canadian GP prediction...")
    predictions = predictor.generate_full_report_v2('Canadian')
    
    if predictions:
        print("✅ Predictions generated successfully!")
        if 'monte_carlo' in predictions:
            print("🎲 Monte Carlo simulation results included!")
            mc_results = predictions['monte_carlo']
            print(f"   📊 Top 3 average positions:")
            for i, (_, row) in enumerate(mc_results.head(3).iterrows()):
                print(f"   {i+1}. {row['Driver']}: {row['Avg_Position']:.1f} ± {row['Std_Dev']:.1f}")
        else:
            print("⚠️ Monte Carlo simulation not included")
    else:
        print("❌ Failed to generate predictions")

except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    print("📋 Full traceback:")
    traceback.print_exc()
