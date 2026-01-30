import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime 
 
print("="*80) 
print("?? AFRICAN AGRICULTURE DATA SCIENCE REVOLUTION") 
print("?? SIMPLE WORKING VERSION") 
print("="*80) 
 
# Create directories 
os.makedirs("results", exist_ok=True) 
os.makedirs("results/visualizations", exist_ok=True) 
 
try: 
    # Load data 
    print("\\n?? LOADING DATA...") 
    df_strawberry = pd.read_csv("data/raw/strawberry_farmers_morogoro.csv") 
    df_arusha = pd.read_csv("data/raw/both_crops_farmers_arusha.csv") 
 
    print(f"? SUCCESS! Data loaded:") 
    print(f"   ?? Strawberry farmers: {len(df_strawberry)}") 
    print(f"   ?? Arusha farmers: {len(df_arusha)}") 
    print(f"   ????? Total farmers: {len(df_strawberry) + len(df_arusha)}") 
 
    # Analyze social capital 
    print("\\n?? ANALYZING SOCIAL CAPITAL...") 
 
    social_terms = ["trust", "share", "group", "network", "cooperat", "meeting"] 
    social_cols_strawberry = [] 
    social_cols_arusha = [] 
 
    for col in df_strawberry.columns: 
        col_lower = str(col).lower() 
        if any(term in col_lower for term in social_terms): 
            social_cols_strawberry.append(col) 
 
    for col in df_arusha.columns: 
        col_lower = str(col).lower() 
        if any(term in col_lower for term in social_terms): 
            social_cols_arusha.append(col) 
 
    print(f"   Found {len(social_cols_strawberry)} social capital indicators in strawberry data") 
    print(f"   Found {len(social_cols_arusha)} social capital indicators in Arusha data") 
 
    # Create visualization 
    print("\\n?? CREATING VISUALIZATION...") 
 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    regions = ["Morogoro\\nStrawberry", "Arusha\\nMulti-Crop"] 
    farmer_counts = [len(df_strawberry), len(df_arusha)] 
 
    bars = ax.bar(regions, farmer_counts, color=["red", "green"], alpha=0.7) 
    ax.set_title("Tanzanian Farmers Surveyed", fontsize=14, fontweight="bold") 
    ax.set_ylabel("Number of Farmers") 
 
    # Add value labels on bars 
    for bar, count in zip(bars, farmer_counts): 
        height = bar.get_height() 
        ax.text(bar.get_x() + bar.get_width()/2, height + 3, 
                str(count), ha="center", va="bottom", fontweight="bold") 
 
    plt.tight_layout() 
    plt.savefig("results/visualizations/farmers_by_region.png", dpi=100) 
    print(f"   ? Saved: results/visualizations/farmers_by_region.png") 
 
    # Generate revenue report 
    print("\\n?? GENERATING REVENUE STRATEGY...") 
 
    report = f'''?? AFRICAN AGRICULTURE REVOLUTION - REVENUE REPORT 
Generated: {datetime.now().strftime("%%Y-%%m-%%d %%H:%%M:%%S")} 
 
?? ANALYSIS SUMMARY: 
 Tanzanian farmers analyzed: {len(df_strawberry) + len(df_arusha)} 
 Social capital indicators found: {len(social_cols_strawberry) + len(social_cols_arusha)} 
 Regions: Morogoro (Strawberry), Arusha (Multi-Crop) 
 
?? SERVICE OFFERINGS: 
1. NGO Dashboard: $2,000 - $5,000 
2. Predictive Analytics: $300 - $1,000/month   
3. Research Consulting: $5,000 - $20,000/project 
4. Training Programs: $3,000 - $8,000 
 
?? REVENUE PROJECTION (Year 1): $25,000 - $75,000 
 
?? NEXT STEPS: 
1. Update GitHub repository 
2. Share on LinkedIn 
3. Contact Tanzanian NGOs 
4. Apply for agriculture data science positions 
''' 
 
    with open("results/revenue_report.md", "w") as f: 
        f.write(report) 
 
    print(f"   ? Saved: results/revenue_report.md") 
 
    print("\\n" + "="*80) 
    print("?? REVOLUTION COMPLETE!") 
    print("="*80) 
    print(f"?? Farmers analyzed: {len(df_strawberry) + len(df_arusha)}") 
    print("?? Revenue potential: $25,000 - $75,000") 
    print("?? Update GitHub: git add . && git commit -m 'Complete' && git push") 
    print("="*80) 
 
except Exception as e: 
    print(f"? ERROR: {e}") 
    import traceback 
    traceback.print_exc() 
