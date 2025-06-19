#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced qualifying and race run separation
"""

from test import F1RacePredictor

def test_enhanced_separation():
    """Test the enhanced run separation system"""
    
    print("🔬 Testing Enhanced F1 Run Separation System")
    print("=" * 60)
    
    # Initialize predictor
    predictor = F1RacePredictor(year=2025)
    
    # Test with Canadian GP (has practice data)
    gp_name = "Canadian"
    
    print(f"\n📊 Analyzing {gp_name} Grand Prix practice sessions...")
    print("-" * 50)
    
    # Extract weekend features using enhanced system
    quali_runs, race_runs = predictor.extract_weekend_features(gp_name)
    
    if not quali_runs.empty:
        print(f"\n🏁 Qualifying Runs Analysis:")
        print("-" * 40)
        print(f"Total qualifying runs identified: {len(quali_runs)}")
        print("\nQualifying runs by driver:")
        driver_quali_counts = quali_runs['Driver'].value_counts().sort_index()
        for driver, count in driver_quali_counts.items():
            avg_lap_time = quali_runs[quali_runs['Driver'] == driver]['LapTime'].mean()
            print(f"   {driver}: {count} runs (avg: {avg_lap_time:.3f}s)")
        
        # Show run characteristics if available
        if 'RunType' in quali_runs.columns:
            print(f"\nRun types distribution:")
            print(quali_runs['RunType'].value_counts())
        
        if 'Confidence' in quali_runs.columns:
            print(f"\nAverage classification confidence: {quali_runs['Confidence'].mean():.2f}")
    
    if not race_runs.empty:
        print(f"\n🏎️ Race Runs Analysis:")
        print("-" * 40)
        print(f"Total race runs identified: {len(race_runs)}")
        print("\nRace runs by driver:")
        driver_race_counts = race_runs['Driver'].value_counts().sort_index()
        for driver, count in driver_race_counts.items():
            avg_stint_length = race_runs[race_runs['Driver'] == driver]['StintLength'].mean()
            avg_pace = race_runs[race_runs['Driver'] == driver]['AvgLapTime'].mean()
            print(f"   {driver}: {count} stints (avg length: {avg_stint_length:.1f}, avg pace: {avg_pace:.3f}s)")
        
        # Show degradation patterns
        if 'DegradationPattern' in race_runs.columns:
            print(f"\nDegradation patterns:")
            print(race_runs['DegradationPattern'].value_counts())
        
        # Show fuel load distribution
        if 'FuelLoad' in race_runs.columns:
            print(f"\nFuel load distribution:")
            print(race_runs['FuelLoad'].value_counts())
        
        if 'Confidence' in race_runs.columns:
            print(f"\nAverage classification confidence: {race_runs['Confidence'].mean():.2f}")
    
    # Test the enhanced run analysis separately
    print(f"\n🔬 Testing Enhanced Run Analysis System:")
    print("-" * 50)
    
    # Load practice data manually to test enhanced analysis
    try:
        import fastf1
        fastf1.Cache.enable_cache('cache')
        
        all_laps = []
        for session in ['FP1', 'FP2', 'FP3']:
            try:
                session_obj = fastf1.get_session(2025, gp_name, session)
                session_obj.load()
                
                laps = session_obj.laps
                if not laps.empty:
                    laps = laps[laps['LapTime'].notna()]
                    laps = laps[laps['PitOutTime'].isna()]
                    laps['Session'] = session
                    all_laps.append(laps)
                    print(f"   Loaded {len(laps)} laps from {session}")
            except Exception as e:
                print(f"   ⚠️ Could not load {session}: {e}")
        
        if all_laps:
            combined_laps = pd.concat(all_laps, ignore_index=True)
            print(f"\n   Total laps for analysis: {len(combined_laps)}")
            
            # Run enhanced analysis
            enhanced_analysis = predictor.enhanced_run_analysis(combined_laps)
            
            print(f"\n📈 Enhanced Analysis Results:")
            for run_type, runs in enhanced_analysis.items():
                if runs:
                    print(f"   {run_type.replace('_', ' ').title()}: {len(runs)} runs")
                    
                    # Show confidence distribution
                    confidences = [run['confidence'] for run in runs]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        print(f"      Average confidence: {avg_conf:.2f}")
                    
                    # Show run purposes
                    purposes = [run.get('primary_type', 'UNKNOWN') for run in runs]
                    from collections import Counter
                    purpose_counts = Counter(purposes)
                    print(f"      Purposes: {dict(purpose_counts)}")
    
    except ImportError:
        print("   ⚠️ FastF1 not available for detailed analysis")
    except Exception as e:
        print(f"   ⚠️ Enhanced analysis failed: {e}")
    
    print(f"\n✅ Enhanced run separation test completed!")
    print("=" * 60)

if __name__ == "__main__":
    import pandas as pd
    test_enhanced_separation()
