"""
🍓 STRAWBERRY FARMER SPECIFIC ANALYSIS
Deep dive analysis for your specific dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_strawberry_data(data_path):
    """
    Complete analysis of strawberry farmer survey data
    """
    print("="*80)
    print("🍓 STRAWBERRY FARMER COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_excel(data_path)
    
    # Create shorter column names for readability
    column_rename = {
        '_1_What_is_your_age': 'age_group',
        '_2_What_is_your_level_of_education': 'education',
        '_3_How_many_years_have_you_been_engaged_in_strawberry_farming': 'farming_experience',
        '_4_How_large_is_your_strawberry_farm': 'farm_size',
        '_5_Are_you_a_member_of_any_farmer_group': 'group_member',
        '_6_How_often_do_you_attend_meeting_for_your_farmer_association': 'meeting_frequency',
        '_7_Do_you_think_being_member_improved_your_farming_practices': 'membership_benefit',
        '_8_How_do_you_rate_the_support_you_received_from_your_association_interms_of_resources_g_seeds_tools': 'group_support',
        '_9_How_much_do_you_trust_other_member_in_your_community': 'trust_level',
        '_10_How_often_do_you_exchange_farming_inforamtions_with_other_members': 'info_exchange_freq',
        '_11_Do_you_believe_the_information_shared_by_other_farmers_has_positively_impacted_your_strawberry_yields': 'info_impact',
        '_12_How_willing_are_you_to_share_farming_knowledge_with_other_strawberry_farmers': 'willingness_to_share',
        '_13_Do_you_participate_in_any_group_activities_related_to_strawberry_farming': 'group_activities',
        '_14_How_often_does_your_community_engage_in_collective_action_to_solve _agricultural_challenges': 'collective_action_freq',
        '_15_How_effective_has_collective_action_been_in_improving_strawberry_farming_outcomes_in_your_community': 'collective_effectiveness',
        '_16_Does_your_community_provide_any_form_of_support_during_farming_crises': 'crisis_support',
        '_17_Do_you_use_organic_or_ecological_farming_practices': 'organic_farming',
        '_18_How_satisfied_are_you_with_the_quality_of_your_soil_and_land_for_strawberry_farming': 'soil_satisfaction',
        '_19_How_accessible_are_agricultural_resources_in_your_region': 'resource_accessibility',
        '_20_Have_you_received_any_formal_training_on_sustainable_strawberry_farming_techniques': 'received_training',
        '_21_Which_of_the_following_sustainable_techniques_do_you_currently_use_in_your_strawberry_farming': 'sustainable_techniques',
        '_22_How_effective_do_you_find_these_sustainable_techniques_in_improving_your_strawberry_yields': 'technique_effectiveness',
        '_23_What_challenges_do_you_face_in_adopting_sustainable_techniques_in_your_farming_practices': 'challenges',
        '_24_How would you describe your strawberry yields over the past year': 'yield_trend'
    }
    
    df = df.rename(columns=column_rename)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total farmers: {len(df)}")
    print(f"   Variables: {len(df.columns)}")
    
    # 1. DEMOGRAPHIC ANALYSIS
    print("\n👥 DEMOGRAPHIC ANALYSIS:")
    
    # Age distribution
    print("\n📊 Age Distribution:")
    age_counts = df['age_group'].value_counts()
    for age_group, count in age_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {age_group}: {count} farmers ({percentage:.1f}%)")
    
    # Education distribution
    print("\n🎓 Education Levels:")
    edu_counts = df['education'].value_counts()
    for edu, count in edu_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {edu}: {count} farmers ({percentage:.1f}%)")
    
    # 2. SOCIAL CAPITAL ANALYSIS
    print("\n🤝 SOCIAL CAPITAL ANALYSIS:")
    
    # Group membership
    group_members = df['group_member'].value_counts(normalize=True) * 100
    print(f"\n📊 Group Membership:")
    print(f"   Members: {group_members.get('yes', 0):.1f}%")
    print(f"   Non-members: {group_members.get('no', 0):.1f}%")
    
    # Trust levels
    print("\n💖 Trust Levels:")
    if 'trust_level' in df.columns:
        trust_counts = df['trust_level'].value_counts()
        for level, count in trust_counts.items():
            print(f"   • {level}: {count} farmers")
    
    # 3. YIELD ANALYSIS
    print("\n📈 YIELD TREND ANALYSIS:")
    
    if 'yield_trend' in df.columns:
        yield_counts = df['yield_trend'].value_counts()
        total = len(df)
        
        print("\n📊 Yield Changes:")
        for trend, count in yield_counts.items():
            percentage = (count / total) * 100
            print(f"   • {trend}: {count} farmers ({percentage:.1f}%)")
    
    # 4. CHALLENGES ANALYSIS
    print("\n🎯 CHALLENGES ANALYSIS:")
    
    if 'challenges' in df.columns:
        # Extract all challenges
        all_challenges = []
        for challenges in df['challenges'].dropna():
            if isinstance(challenges, str):
                # Split by space or underscore
                items = challenges.replace('_', ' ').split()
                all_challenges.extend(items)
        
        from collections import Counter
        challenge_counts = Counter(all_challenges)
        
        print("\n📊 Most Common Challenges:")
        for challenge, count in challenge_counts.most_common(5):
            percentage = (count / len(df)) * 100
            print(f"   • {challenge}: {count} mentions ({percentage:.1f}%)")
    
    # 5. VISUALIZATIONS
    print("\n📊 CREATING VISUALIZATIONS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Age Distribution
    age_order = ['below_30', '30_40', '41_50', 'above_50']
    age_counts = df['age_group'].value_counts().reindex(age_order)
    axes[0, 0].bar(age_counts.index, age_counts.values, color='skyblue')
    axes[0, 0].set_title('Age Distribution of Farmers', fontweight='bold')
    axes[0, 0].set_xlabel('Age Group')
    axes[0, 0].set_ylabel('Number of Farmers')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Education Distribution
    edu_counts = df['education'].value_counts()
    axes[0, 1].pie(edu_counts.values, labels=edu_counts.index, autopct='%1.1f%%',
                  colors=['lightgreen', 'lightblue', 'gold', 'salmon'])
    axes[0, 1].set_title('Education Level Distribution', fontweight='bold')
    
    # Plot 3: Group Membership
    group_counts = df['group_member'].value_counts()
    axes[0, 2].bar(['Members', 'Non-members'], group_counts.values, 
                  color=['green', 'red'])
    axes[0, 2].set_title('Farmer Group Membership', fontweight='bold')
    axes[0, 2].set_ylabel('Number of Farmers')
    
    # Plot 4: Yield Trends
    if 'yield_trend' in df.columns:
        yield_counts = df['yield_trend'].value_counts()
        colors = ['red', 'yellow', 'green']
        axes[1, 0].bar(yield_counts.index, yield_counts.values, color=colors[:len(yield_counts)])
        axes[1, 0].set_title('Strawberry Yield Trends', fontweight='bold')
        axes[1, 0].set_xlabel('Yield Change')
        axes[1, 0].set_ylabel('Number of Farmers')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Farm Size
    if 'farm_size' in df.columns:
        size_counts = df['farm_size'].value_counts()
        axes[1, 1].bar(size_counts.index, size_counts.values, color='purple')
        axes[1, 1].set_title('Farm Size Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Farm Size')
        axes[1, 1].set_ylabel('Number of Farmers')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Challenges Word Cloud
    axes[1, 2].axis('off')
    if 'challenges' in df.columns and len(all_challenges) > 0:
        from wordcloud import WordCloud
        
        text = ' '.join(all_challenges)
        wordcloud = WordCloud(width=400, height=300, 
                            background_color='white',
                            max_words=50).generate(text)
        
        axes[1, 2].imshow(wordcloud, interpolation='bilinear')
        axes[1, 2].set_title('Common Challenges Word Cloud', fontweight='bold')
    
    plt.suptitle('Strawberry Farmer Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 6. KEY INSIGHTS
    print("\n💡 KEY INSIGHTS:")
    
    # Insight 1: Group membership benefits
    if 'group_member' in df.columns and 'yield_trend' in df.columns:
        group_yield = df.groupby('group_member')['yield_trend'].apply(
            lambda x: (x == 'increased').sum() / len(x) * 100
        )
        
        print(f"\n1. GROUP MEMBERSHIP IMPACT:")
        print(f"   • Members with increased yields: {group_yield.get('yes', 0):.1f}%")
        print(f"   • Non-members with increased yields: {group_yield.get('no', 0):.1f}%")
    
    # Insight 2: Education impact
    if 'education' in df.columns and 'yield_trend' in df.columns:
        print(f"\n2. EDUCATION IMPACT:")
        for edu_level in df['education'].unique():
            edu_df = df[df['education'] == edu_level]
            if len(edu_df) > 0:
                increased_pct = (edu_df['yield_trend'] == 'increased').sum() / len(edu_df) * 100
                print(f"   • {edu_level}: {increased_pct:.1f}% had increased yields")
    
    # Insight 3: Trust and yields
    if 'trust_level' in df.columns and 'yield_trend' in df.columns:
        trust_order = ['not_at_all', 'slightly', 'moderately', 'significantly', 'completely']
        print(f"\n3. TRUST AND YIELD RELATIONSHIP:")
        
        for trust_level in trust_order:
            if trust_level in df['trust_level'].values:
                trust_df = df[df['trust_level'] == trust_level]
                increased_pct = (trust_df['yield_trend'] == 'increased').sum() / len(trust_df) * 100
                print(f"   • {trust_level}: {increased_pct:.1f}% had increased yields")
    
    # 7. RECOMMENDATIONS
    print("\n🎯 ACTIONABLE RECOMMENDATIONS:")
    print("   1. Strengthen farmer groups - they correlate with better yields")
    print("   2. Focus on trust-building activities in communities")
    print("   3. Provide targeted training for farmers with lower education")
    print("   4. Address specific challenges: climate factors, resource access")
    print("   5. Promote sustainable techniques among experienced farmers")
    
    # Save analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/strawberry_analysis_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("STRAWBERRY FARMER ANALYSIS REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Farmers: {len(df)}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*30 + "\n")
        
        # Add key findings
        f.write(f"1. Age Distribution:\n")
        for age_group, count in age_counts.items():
            f.write(f"   {age_group}: {count} farmers\n")
        
        f.write(f"\n2. Education Levels:\n")
        for edu, count in edu_counts.items():
            f.write(f"   {edu}: {count} farmers\n")
        
        f.write(f"\n3. Yield Trends:\n")
        if 'yield_trend' in df.columns:
            for trend, count in df['yield_trend'].value_counts().items():
                f.write(f"   {trend}: {count} farmers\n")
    
    print(f"\n📄 Analysis report saved to: {report_path}")
    
    return df

# Main execution
if __name__ == "__main__":
    print("🍓 STRAWBERRY FARMER ANALYSIS")
    
    # Try default path
    import os
    default_path = "C:/Users/hp/OneDrive/Documents/African_Agriculture_Revolution/data/Strawberry_Farmer_Data.xlsx"
    
    if os.path.exists(default_path):
        data_path = default_path
        print(f"📁 Loading data from: {data_path}")
        results = analyze_strawberry_data(data_path)
    else:
        print("❌ Data file not found at default path.")
        print("💡 Please update the path in the script or enter manually.")