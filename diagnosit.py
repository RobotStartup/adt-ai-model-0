import pandas as pd

df = pd.read_csv("ai_model_sales.csv")

print("="*60)
print("CONVERSION PATTERNS ANALYSIS")
print("="*60)

# Overall conversion rate
conv_rate = df['conversion'].mean()
print(f"\nOverall conversion rate: {conv_rate:.1%}")
print(f"Total records: {len(df)}")
print(f"Conversions: {df['conversion'].sum()}")

# By recency
print("\n--- BY DAYS SINCE DISCONNECT ---")
df['recency_bin'] = pd.cut(df['days_since_disconnect'], 
                           bins=[0, 30, 90, 180, 365, 999], 
                           labels=['<30d', '30-90d', '90-180d', '180-365d', '>365d'])
print(df.groupby('recency_bin')['conversion'].agg(['mean', 'count']))

# By RMR
print("\n--- BY HISTORICAL RMR ---")
df['rmr_bin'] = pd.cut(df['historical_rmr'], 
                       bins=[-1, 0, 100, 200, 999], 
                       labels=['None', 'Low', 'Medium', 'High'])
print(df.groupby('rmr_bin')['conversion'].agg(['mean', 'count']))

# By equipment type  
print("\n--- BY EQUIPMENT TYPE ---")
print(df.groupby('equipment_service type')['conversion'].agg(['mean', 'count']))

# By purchase history
print("\n--- BY SERVICE COUNT ---")
df['has_services'] = df['purchase_history'] != 'none'
print(df.groupby('has_services')['conversion'].agg(['mean', 'count']))

# By multiple locations
print("\n--- BY MULTIPLE LOCATIONS ---")
print(df.groupby('multiple_locations')['conversion'].agg(['mean', 'count']))

# Check for any strong signal
print("\n" + "="*60)
print("LOOKING FOR STRONG SIGNALS")
print("="*60)

# Recent + high RMR
recent_high_value = df[(df['days_since_disconnect'] < 90) & (df['historical_rmr'] > 150)]
print(f"\nRecent (<90d) + High RMR (>$150):")
print(f"  Count: {len(recent_high_value)}")
print(f"  Conversion rate: {recent_high_value['conversion'].mean():.1%}")

# Compare to old + no services
old_no_service = df[(df['days_since_disconnect'] > 300) & (df['historical_rmr'] == 0)]
print(f"\nOld (>300d) + No RMR:")
print(f"  Count: {len(old_no_service)}")
print(f"  Conversion rate: {old_no_service['conversion'].mean():.1%}")