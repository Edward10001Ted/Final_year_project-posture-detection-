#!/usr/bin/env python3
"""Test script to verify bias logic without camera"""

# Simulate different prediction scenarios
EDWARD_BIAS = 0.15

def test_bias_scenario(raw_sena, raw_edward, scenario_name):
    """Test a specific scenario"""
    prediction_counts = {'Sena': raw_sena, 'Edward': raw_edward}
    
    # Apply bias
    biased_counts = prediction_counts.copy()
    biased_counts['Edward'] *= (1 + EDWARD_BIAS)
    
    final_label = max(biased_counts, key=biased_counts.get)
    
    print(f"\n{scenario_name}:")
    print(f"  Raw counts - Sena: {prediction_counts['Sena']}, Edward: {prediction_counts['Edward']}")
    print(f"  With {EDWARD_BIAS*100:.0f}% Edward Bias - Sena: {biased_counts['Sena']:.1f}, Edward: {biased_counts['Edward']:.1f}")
    print(f"  FINAL: {final_label}")
    
    return final_label

# Test scenarios
print("=" * 60)
print("Testing Edward Bias Logic (15% bias)")
print("=" * 60)

# Scenario 1: Model heavily biased toward Sena
test_bias_scenario(70, 15, "Scenario 1: Heavy Sena bias (70 vs 15)")

# Scenario 2: Model moderately biased toward Sena  
test_bias_scenario(60, 30, "Scenario 2: Moderate Sena bias (60 vs 30)")

# Scenario 3: Equal predictions
test_bias_scenario(45, 45, "Scenario 3: Equal predictions (45 vs 45)")

# Scenario 4: Equal but with different total counts
test_bias_scenario(90, 90, "Scenario 4: Equal high counts (90 vs 90)")

# Scenario 5: Edge case - Edward wins originally
test_bias_scenario(40, 60, "Scenario 5: Edward wins originally (40 vs 60)")

print("\n" + "=" * 60)
print("Bias effectiveness check:")
print("=" * 60)

# Check how much bias is needed to overcome different deficits
for sena_count in [50, 60, 70, 80]:
    edward_count = 30
    biased_edward = edward_count * (1 + EDWARD_BIAS)
    print(f"Sena {sena_count} vs Edward {edward_count}: Edward with bias = {biased_edward:.1f} → ", end="")
    if biased_edward > sena_count:
        print("Edward WINS ✓")
    else:
        print(f"Sena still wins (need {((sena_count/edward_count - 1)*100):.1f}% bias for Edward)")

print(f"\nConclusion: 15% bias helps Edward but may not be enough for large deficits.")
print(f"Consider increasing to 20-25% if needed.")
