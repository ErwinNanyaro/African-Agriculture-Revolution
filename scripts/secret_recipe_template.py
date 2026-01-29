"""
🚀 AFRICAN AGRICULTURE REVOLUTION: SECRET RECIPE TEMPLATE
Transform SPSS/Stata analyses into modern data science pipelines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RevolutionMindset:
    """The revolutionary mindset shift for African agriculture data science"""
    
    @staticmethod
    def manifesto():
        print("="*80)
        print("🌍 AFRICAN AGRICULTURE DATA SCIENCE REVOLUTION")
        print("="*80)
        print("\n💡 MANIFESTO:")
        print("   • FROM SPSS/Stata → TO Python ML pipelines")
        print("   • FROM Static reports → TO Interactive dashboards")
        print("   • FROM Academic papers → TO Business insights")
        print("   • FROM One-time analysis → TO Continuous value creation")
        print("\n🎯 MISSION: Transform African agriculture through data science")
        print("💰 VALUE: Create solutions organizations will pay for")
        print("="*80)
        return True

def revolutionary_diagnose(df, dataset_name):
    """
    Comprehensive data diagnosis for agricultural surveys
    """
    print(f"\n🔍 STEP 1: REVOLUTIONARY DATA DIAGNOSIS - {dataset_name}")
    print("-"*60)
    
    insights = {'shape': df.shape, 'columns': list(df.columns)}
    
    print(f"📊 Dataset: {df.shape[0]} farmers × {df.shape[1]} variables")
    print(f"📅 Time collected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-categorize columns
    categories = {
        'demographic': [],
        'social_capital': [],
        'production': [],
        'challenges': [],
        'outcomes': []
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        if any(word in col_lower for word in ['age', 'education', 'gender']):
            categories['demographic'].append(col)
        elif any(word in col_lower for word in ['trust', 'group', 'share', 'cooperat']):
            categories['social_capital'].append(col)
        elif any(word in col_lower for word in ['farm', 'yield', 'acre', 'production']):
            categories['production'].append(col)
        elif any(word in col_lower for word in ['challenge', 'problem', 'difficult']):
            categories['challenges'].append(col)
        elif any(word in col_lower for word in ['satisfaction', 'improve', 'increase']):
            categories['outcomes'].append(col)
    
    print("\n📋 VARIABLE CATEGORIES:")
    for cat, cols in categories.items():
        if cols:
            print(f"   • {cat.title()}: {len(cols)} variables")
    
    return insights

def revolutionary_clean(df):
    """
    Intelligent cleaning for agricultural survey data
    """
    print(f"\n🧹 STEP 2: REVOLUTIONARY DATA CLEANING")
    print("-"*60)
    
    df_clean = df.copy()
    
    # Define encoding schemes
    LIKERT_MAP = {
        'not_at_all': 1, 'never': 1, 'not_effective': 1,
        'slightly': 2, 'rarely': 2, 'slightly_effective': 2,
        'moderately': 3, 'occasionally': 3, 'moderately_effective': 3,
        'significantly': 4, 'frequently': 4, 'very_effective': 4,
        'completely': 5, 'always': 5
    }
    
    YIELD_MAP = {
        'decreased': -1,
        'remained_the_same': 0,
        'increased': 1
    }
    
    AGE_MAP = {
        'below_30': 25,
        '30_40': 35,
        '41_50': 45,
        'above_50': 55
    }
    
    # 1. Encode categorical variables
    print("🔧 Encoding categorical variables...")
    
    for col in df_clean.columns:
        # Check for Likert scales
        sample_val = str(df_clean[col].iloc[0]) if len(df_clean) > 0 else ""
        
        for key in LIKERT_MAP.keys():
            if key in sample_val.lower():
                df_clean[f"{col}_encoded"] = df_clean[col].map(LIKERT_MAP)
                break
        
        # Check for yield changes
        if 'yield' in col.lower() and any(key in sample_val.lower() for key in YIELD_MAP.keys()):
            df_clean[f"{col}_encoded"] = df_clean[col].map(YIELD_MAP)
        
        # Check for age groups
        if 'age' in col.lower() and any(key in sample_val.lower() for key in AGE_MAP.keys()):
            df_clean[f"{col}_numeric"] = df_clean[col].map(AGE_MAP)
    
    # 2. Create composite indices
    print("\n📊 Creating composite indices...")
    
    # Social Capital Index
    social_cols = [col for col in df_clean.columns if any(term in str(col).lower() 
                  for term in ['trust', 'share', 'group', 'cooperat', 'network', 'member'])]
    social_encoded = [f"{col}_encoded" for col in social_cols if f"{col}_encoded" in df_clean.columns]
    
    if social_encoded:
        df_clean['social_capital_index'] = df_clean[social_encoded].mean(axis=1)
        print(f"   Created Social Capital Index from {len(social_encoded)} variables")
    
    # Resource Access Index
    resource_cols = [col for col in df_clean.columns if any(term in str(col).lower()
                    for term in ['access', 'resource', 'tool', 'input', 'seed', 'training'])]
    resource_encoded = [f"{col}_encoded" for col in resource_cols if f"{col}_encoded" in df_clean.columns]
    
    if resource_encoded:
        df_clean['resource_access_index'] = df_clean[resource_encoded].mean(axis=1)
        print(f"   Created Resource Access Index from {len(resource_encoded)} variables")
    
    # 3. Handle missing values
    print("\n🧪 Handling missing values...")
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    return df_clean

def revolutionary_features(df):
    """
    Create powerful features for agricultural analysis
    """
    print(f"\n🔧 STEP 3: REVOLUTIONARY FEATURE ENGINEERING")
    print("-"*60)
    
    df_features = df.copy()
    
    # 1. Extract farm size
    size_cols = [col for col in df_features.columns if any(term in str(col).lower() 
                 for term in ['acre', 'farm', 'size', 'large'])]
    
    if size_cols:
        size_col = size_cols[0]
        
        def parse_farm_size(size_str):
            if pd.isna(size_str):
                return np.nan
            
            size_str = str(size_str).lower()
            
            if 'less_than' in size_str and '1' in size_str:
                return 0.5
            elif '1_3' in size_str:
                return 2.0
            elif '4_6' in size_str:
                return 5.0
            elif 'more_than' in size_str and '6' in size_str:
                return 8.0
            else:
                return np.nan
        
        df_features['farm_size_acres'] = df_features[size_col].apply(parse_farm_size)
        print(f"   Created: farm_size_acres")
    
    # 2. Create farmer segments
    print("\n🎯 Creating farmer segments...")
    
    if 'social_capital_index' in df_features.columns and 'resource_access_index' in df_features.columns:
        # Segment by social capital and resource access
        conditions = [
            (df_features['social_capital_index'] >= 3) & (df_features['resource_access_index'] >= 3),
            (df_features['social_capital_index'] >= 3) & (df_features['resource_access_index'] < 3),
            (df_features['social_capital_index'] < 3) & (df_features['resource_access_index'] >= 3),
            (df_features['social_capital_index'] < 3) & (df_features['resource_access_index'] < 3)
        ]
        
        choices = ['HighSocial_HighResource', 'HighSocial_LowResource', 
                  'LowSocial_HighResource', 'LowSocial_LowResource']
        
        df_features['farmer_segment'] = np.select(conditions, choices, default='Other')
        print(f"   Created {df_features['farmer_segment'].nunique()} farmer segments")
    
    # 3. Create performance score
    yield_cols = [col for col in df_features.columns if 'yield' in str(col).lower() 
                  and 'encoded' in str(col)]
    
    if yield_cols:
        df_features['performance_score'] = df_features[yield_cols].mean(axis=1)
        print(f"   Created performance_score from {len(yield_cols)} yield variables")
    
    return df_features

def revolutionary_visualize(df, context="Strawberry Farmers"):
    """
    Create publication-quality visualizations
    """
    print(f"\n📊 STEP 4: REVOLUTIONARY VISUALIZATIONS")
    print("-"*60)
    
    # Create figure with fewer plots if data is limited
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'African Agriculture Analytics: {context}', fontsize=20, fontweight='bold')
    
    plot_count = 0
    
    # Plot 1: Social Capital Distribution
    if 'social_capital_index' in df.columns:
        plot_count += 1
        ax1 = plt.subplot(3, 3, plot_count)
        sns.histplot(df['social_capital_index'], kde=True, color='blue', ax=ax1, bins=20)
        ax1.axvline(df['social_capital_index'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["social_capital_index"].mean():.2f}')
        ax1.set_title('Social Capital Distribution', fontweight='bold')
        ax1.set_xlabel('Social Capital Index')
        ax1.set_ylabel('Frequency')
        ax1.legend()
    
    # Plot 2: Performance Score Distribution
    if 'performance_score' in df.columns:
        plot_count += 1
        ax2 = plt.subplot(3, 3, plot_count)
        sns.histplot(df['performance_score'], kde=True, color='green', ax=ax2, bins=20)
        ax2.set_title('Performance Score Distribution', fontweight='bold')
        ax2.set_xlabel('Performance Score')
        ax2.set_ylabel('Frequency')
    
    # Plot 3: Resource Access Distribution
    if 'resource_access_index' in df.columns:
        plot_count += 1
        ax3 = plt.subplot(3, 3, plot_count)
        # Fixed boxplot
        box_data = [df['resource_access_index'].dropna().values]
        ax3.boxplot(box_data, patch_artist=True, 
                   boxprops=dict(facecolor='orange'))
        ax3.set_title('Resource Access Distribution', fontweight='bold')
        ax3.set_ylabel('Resource Access Index')
        ax3.set_xticklabels(['Resource Access'])
    
    # Plot 4: Education Distribution
    edu_cols = [col for col in df.columns if 'education' in str(col).lower()]
    if edu_cols:
        plot_count += 1
        ax4 = plt.subplot(3, 3, plot_count)
        edu_col = edu_cols[0]
        edu_counts = df[edu_col].value_counts()
        edu_counts.plot(kind='bar', color='green', ax=ax4)
        ax4.set_title('Education Level Distribution', fontweight='bold')
        ax4.set_xlabel('Education Level')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Farm Size Distribution
    if 'farm_size_acres' in df.columns:
        plot_count += 1
        ax5 = plt.subplot(3, 3, plot_count)
        df['farm_size_acres'].hist(bins=20, color='purple', ax=ax5, edgecolor='black')
        ax5.set_title('Farm Size Distribution', fontweight='bold')
        ax5.set_xlabel('Farm Size (acres)')
        ax5.set_ylabel('Count')
    
    # Plot 6: Age Distribution
    age_cols = [col for col in df.columns if 'age_numeric' in str(col)]
    if age_cols:
        plot_count += 1
        ax6 = plt.subplot(3, 3, plot_count)
        df[age_cols[0]].hist(bins=15, color='teal', ax=ax6, edgecolor='black')
        ax6.set_title('Age Distribution', fontweight='bold')
        ax6.set_xlabel('Age (years)')
        ax6.set_ylabel('Count')
    
    # Plot 7: Social Capital vs Performance
    if 'social_capital_index' in df.columns and 'performance_score' in df.columns:
        plot_count += 1
        ax7 = plt.subplot(3, 3, plot_count)
        sns.scatterplot(x='social_capital_index', y='performance_score', 
                       data=df, alpha=0.6, ax=ax7)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['social_capital_index'], df['performance_score'], 1)
            p = np.poly1d(z)
            ax7.plot(df['social_capital_index'], p(df['social_capital_index']), 
                    "r--", alpha=0.8)
        
        ax7.set_title('Social Capital vs Performance', fontweight='bold')
        ax7.set_xlabel('Social Capital Index')
        ax7.set_ylabel('Performance Score')
    
    # Plot 8: Challenges Analysis
    challenge_cols = [col for col in df.columns if 'challenge' in str(col).lower()]
    if challenge_cols:
        plot_count += 1
        ax8 = plt.subplot(3, 3, plot_count)
        challenge_col = challenge_cols[0]
        
        # Extract individual challenges
        all_challenges = []
        for val in df[challenge_col].dropna():
            if isinstance(val, str):
                challenges = val.replace('_', ' ').split()
                all_challenges.extend(challenges)
        
        from collections import Counter
        challenge_counts = Counter(all_challenges)
        top_challenges = dict(challenge_counts.most_common(5))
        
        if top_challenges:
            ax8.barh(list(top_challenges.keys()), list(top_challenges.values()), color='salmon')
            ax8.set_title('Top 5 Challenges', fontweight='bold')
            ax8.set_xlabel('Frequency')
    
    # Plot 9: Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 2:
        plot_count += 1
        ax9 = plt.subplot(3, 3, plot_count)
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax9, cbar_kws={"shrink": 0.8})
        ax9.set_title('Feature Correlation Matrix', fontweight='bold')
    
    # Hide empty subplots
    for i in range(plot_count + 1, 10):
        ax = plt.subplot(3, 3, i)
        ax.axis('off')
    
    # ==== FIXED SECTION ====
    plt.tight_layout()
    plt.show(block=False)  # This allows script to continue
    plt.pause(5)  # Show graphs for 5 seconds
    plt.close('all')  # Close all figures
    # =======================
    
    print(f"✅ Created {plot_count} revolutionary visualizations")

def revolutionary_model(df):
    """
    Build predictive models for agricultural insights
    """
    print(f"\n🤖 STEP 5: PREDICTIVE MODELING")
    print("-"*60)
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Create target variable: High performance farmers
        if 'performance_score' in df.columns:
            median_perf = df['performance_score'].median()
            df['high_performer'] = (df['performance_score'] > median_perf).astype(int)
            target_col = 'high_performer'
        elif 'social_capital_index' in df.columns:
            median_sc = df['social_capital_index'].median()
            df['high_social_capital'] = (df['social_capital_index'] > median_sc).astype(int)
            target_col = 'high_social_capital'
        else:
            print("⚠️ No suitable target variable found. Creating synthetic target.")
            return None
        
        # Prepare features
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Model Setup:")
        print(f"   Features: {X.shape[1]}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        print(f"   Positive class: {y.mean():.1%}")
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"\n🏆 Model Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importances.head(10).to_string(index=False))
        
        # ==== FIXED: Visualize feature importance ====
        plt.figure(figsize=(12, 6))
        top_features = importances.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.gca().invert_yaxis()
        plt.title('Top 10 Most Important Features', fontweight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show(block=False)  # Don't block execution
        plt.pause(3)  # Show for 3 seconds
        plt.close()  # Close the figure
        # ============================================
        
        return model
        
    except Exception as e:
        print(f"⚠️ Modeling error: {str(e)}")
        return None

def revolutionary_monetize(df, context):
    """
    Create monetization strategy from insights
    """
    print(f"\n💰 STEP 6: MONETIZATION STRATEGY")
    print("-"*60)
    
    print("🚀 MONEY-MAKING OPPORTUNITIES:")
    
    opportunities = [
        {
            'name': 'Farmer Segmentation Dashboard',
            'description': 'Interactive dashboard showing farmer segments and performance',
            'clients': ['NGOs', 'Government', 'Agribusiness'],
            'price': '$1,500 - $5,000',
            'timeline': '2-3 weeks'
        },
        {
            'name': 'Social Capital Assessment Service',
            'description': 'Comprehensive analysis of farmer networks and trust',
            'clients': ['Development Organizations', 'Cooperatives'],
            'price': '$2,000 - $8,000',
            'timeline': '3-4 weeks'
        },
        {
            'name': 'Yield Prediction Model',
            'description': 'ML model predicting yield changes based on farmer characteristics',
            'clients': ['Insurance Companies', 'Input Suppliers'],
            'price': '$300/month subscription',
            'timeline': '4-5 weeks'
        }
    ]
    
    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. {opp['name']}")
        print(f"   📝 {opp['description']}")
        print(f"   🎯 Clients: {', '.join(opp['clients'])}")
        print(f"   💰 Price: {opp['price']}")
        print(f"   ⏱️ Timeline: {opp['timeline']}")
    
    print("\n🎯 IMMEDIATE ACTION STEPS:")
    print("   1. Create GitHub portfolio with this analysis")
    print("   2. Build Streamlit dashboard")
    print("   3. Write blog post on LinkedIn")
    print("   4. Reach out to 3 potential clients")
    
    return opportunities

def execute_revolution(data_path, dataset_name="Agricultural Data"):
    """
    Execute the complete revolutionary pipeline
    """
    print("\n" + "="*80)
    print("🚀 EXECUTING AFRICAN AGRICULTURE DATA REVOLUTION")
    print("="*80)
    
    # Step 0: Mindset
    RevolutionMindset.manifesto()
    
    # Load data
    try:
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Use .xlsx or .csv files")
        
        print(f"\n✅ Data loaded: {len(df)} records")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Execute pipeline
    insights = revolutionary_diagnose(df, dataset_name)
    df_clean = revolutionary_clean(df)
    df_features = revolutionary_features(df_clean)
    revolutionary_visualize(df_features, dataset_name)
    model = revolutionary_model(df_features)
    opportunities = revolutionary_monetize(df_features, dataset_name)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/results_{dataset_name.replace(' ', '_')}_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_features.to_excel(writer, sheet_name='Processed_Data', index=False)
        
        # Save summary
        summary_data = {
            'Metric': ['Total Farmers', 'Variables Processed', 
                      'Social Capital Avg', 'Resource Access Avg'],
            'Value': [len(df_features), df_features.shape[1],
                     df_features.get('social_capital_index', pd.Series([0])).mean(),
                     df_features.get('resource_access_index', pd.Series([0])).mean()]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\n✅ REVOLUTION COMPLETE!")
    print(f"📁 Results saved to: {output_path}")
    
    return {
        'raw_data': df,
        'processed_data': df_features,
        'model': model,
        'opportunities': opportunities,
        'output_file': output_path
    }

# Google Colab integration
def setup_colab():
    """
    Setup code for Google Colab
    """
    colab_code = """
# AFRICAN AGRICULTURE REVOLUTION - COLAB SETUP
# ============================================

# 1. Install required packages
!pip install pandas numpy matplotlib seaborn scikit-learn openpyxl -q
!pip install streamlit -q

# 2. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 3. Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# 4. Clone your GitHub repository
!git clone https://github.com/ErwinNanyaro/African-Agriculture-Revolution.git

# 5. Navigate to your project
%cd /content/African-Agriculture-Revolution

print("✅ Environment setup complete!")
print("📁 Files available:")
!ls -la
    """
    return colab_code

def get_github_guide():
    """
    GitHub integration guide - SIMPLIFIED VERSION
    """
    guide = """
# GITHUB + GOOGLE COLAB INTEGRATION GUIDE

## 1. INITIAL GITHUB SETUP (Already done)

## 2. DAILY WORKFLOW:

### On your laptop:
1. Save scripts to: C:\\Users\\hp\\OneDrive\\Documents\\African_Agriculture_Revolution
2. Commit changes:
   git add .
   git commit -m "Update analysis"
   git push origin main

### On Google Colab:
1. Open https://colab.research.google.com
2. Clone your repository:
   !git clone https://github.com/ErwinNanyaro/African-Agriculture-Revolution.git
   %cd African-Agriculture-Revolution
3. Run analysis
4. Push results back:
   !git add results/
   !git commit -m "Colab analysis results"
   !git push origin main

## 3. FOLDER STRUCTURE:
African-Agriculture-Revolution/
├── data/              # Your datasets
├── scripts/           # Python scripts
├── notebooks/         # Colab notebooks
├── results/           # Analysis outputs
└── README.md          # Project documentation
    """
    return guide

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("🌍 AFRICAN AGRICULTURE DATA SCIENCE REVOLUTION")
    print("="*80)
    
    # Get data path
    import os
    
    # Try default path first
    default_path = "C:/Users/hp/OneDrive/Documents/African_Agriculture_Revolution/data/Strawberry_Farmer_Data.xlsx"
    
    if os.path.exists(default_path):
        data_path = default_path
        print(f"📁 Using data at: {data_path}")
    else:
        print("📁 Please enter path to your data file:")
        data_path = input("Path: ").strip()
    
    # Execute revolution
    try:
        results = execute_revolution(
            data_path=data_path,
            dataset_name="Tanzania Strawberry Farmers"
        )
        
        if results:
            print("\n" + "="*80)
            print("🎉 REVOLUTION SUCCESSFUL!")
            print("="*80)
            
            print("\n📋 NEXT STEPS:")
            print("   1. Review the Excel file in 'results/' folder")
            print("   2. Run on 2 more datasets to build portfolio")
            print("   3. Create GitHub repository (you already have)")
            print("   4. Write blog post about your findings")
            
            # Show integration guide
            print("\n" + "="*80)
            print(get_github_guide())
            print("="*80)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Install packages: pip install pandas openpyxl matplotlib seaborn scikit-learn")
        print("   2. Check file path exists")
        print("   3. Ensure correct file format (.xlsx or .csv)")