"""
🚀 REVOLUTIONARY DATA SCIENCE TEMPLATE FOR AFRICAN AGRICULTURE
From SPSS/Stata to Modern Data Science - The Complete Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# STEP 0: THE REVOLUTIONARY MINDSET
# ============================================================

class AfricanAgricultureRevolution:
    """The mindset shift from old methods to modern data science"""
    
    @staticmethod
    def mindset():
        print("="*80)
        print("🌍 AFRICAN AGRICULTURE DATA SCIENCE REVOLUTION")
        print("="*80)
        print("\n💡 MINDSET SHIFT:")
        print("   • FROM: SPSS/Stata → TO: Python/R with ML pipelines")
        print("   • FROM: Static reports → TO: Interactive dashboards")
        print("   • FROM: Academic papers → TO: Actionable business insights")
        print("   • FROM: One-time analysis → TO: Continuous value creation")
        print("\n🎯 MISSION: Transform African agriculture through data science")
        print("💰 GOAL: Create solutions organizations will pay for")
        print("="*80)
        return True

# ============================================================
# STEP 1: DATA UNDERSTANDING & DIAGNOSTICS
# ============================================================

def diagnose_data(df, dataset_name="Agricultural Dataset"):
    """
    Revolutionary data understanding - Beyond basic EDA
    """
    print(f"\n🔍 STEP 1: COMPREHENSIVE DATA DIAGNOSIS - {dataset_name}")
    print("-"*60)
    
    insights = {
        'basic_info': {},
        'quality_metrics': {},
        'revolutionary_insights': []
    }
    
    # 1A: Basic Information
    insights['basic_info']['shape'] = df.shape
    insights['basic_info']['memory_mb'] = df.memory_usage(deep=True).sum() / (1024**2)
    
    print(f"📊 Dataset Shape: {df.shape[0]} farmers × {df.shape[1]} variables")
    print(f"💾 Memory Usage: {insights['basic_info']['memory_mb']:.2f} MB")
    
    # 1B: Column Intelligence
    print("\n📋 COLUMN INTELLIGENCE:")
    column_types = {
        'demographic': [],
        'behavioral': [],
        'outcome': [],
        'resource': [],
        'social': []
    }
    
    for col in df.columns:
        # Auto-categorize columns
        col_lower = str(col).lower()
        
        if any(word in col_lower for word in ['age', 'education', 'gender', 'year']):
            column_types['demographic'].append(col)
        elif any(word in col_lower for word in ['yield', 'income', 'profit', 'satisfaction']):
            column_types['outcome'].append(col)
        elif any(word in col_lower for word in ['group', 'share', 'trust', 'cooperat']):
            column_types['social'].append(col)
        elif any(word in col_lower for word in ['access', 'resource', 'tool', 'input']):
            column_types['resource'].append(col)
        else:
            column_types['behavioral'].append(col)
    
    for cat, cols in column_types.items():
        if cols:
            print(f"   • {cat.title()}: {len(cols)} variables")
    
    # 1C: Data Quality Revolution
    print("\n🧪 DATA QUALITY METRICS:")
    
    quality_issues = []
    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        unique_pct = df[col].nunique() / len(df) * 100
        
        if null_pct > 20:
            quality_issues.append(f"High missing values in {col} ({null_pct:.1f}%)")
        
        if unique_pct > 90 and df[col].dtype == 'object':
            quality_issues.append(f"Potential free-text in {col}")
    
    if quality_issues:
        print("   ⚠️ Issues found:")
        for issue in quality_issues[:3]:  # Show top 3
            print(f"     - {issue}")
    else:
        print("   ✅ Good data quality")
    
    # 1D: First 5 Revolutionary Insights
    print("\n💡 FIRST 5 REVOLUTIONARY INSIGHTS:")
    
    # Insight 1: Data collection patterns
    if 'start' in df.columns and 'end' in df.columns:
        df['duration_minutes'] = (pd.to_datetime(df['end']) - pd.to_datetime(df['start'])).dt.total_seconds() / 60
        avg_duration = df['duration_minutes'].mean()
        insights['revolutionary_insights'].append(f"Average survey duration: {avg_duration:.1f} minutes")
        print(f"   1. Surveys took {avg_duration:.1f} minutes on average")
    
    # Insight 2: Response patterns
    for col in df.columns[:3]:  # First 3 columns
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
            insights['revolutionary_insights'].append(f"Most common response in {col}: {mode_val}")
            print(f"   2. Most common in {col[:20]}...: {mode_val}")
            break
    
    return insights

# ============================================================
# STEP 2: REVOLUTIONARY DATA CLEANING
# ============================================================

def revolutionary_clean(df, agricultural_context=True):
    """
    Clean agricultural survey data intelligently
    """
    print(f"\n🧹 STEP 2: REVOLUTIONARY DATA CLEANING")
    print("-"*60)
    
    df_clean = df.copy()
    transformations = []
    
    # 2A: Intelligent Missing Value Imputation
    print("🔧 Intelligent Missing Value Handling:")
    
    for col in df_clean.columns:
        null_count = df_clean[col].isnull().sum()
        
        if null_count > 0:
            null_pct = (null_count / len(df_clean)) * 100
            
            # Strategy based on column type and missing percentage
            if df_clean[col].dtype in ['int64', 'float64']:
                if null_pct < 5:
                    # Small missing, use median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    transformations.append(f"Numeric {col}: Filled {null_count} missing with median")
                else:
                    # Large missing, use advanced imputation
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    transformations.append(f"Numeric {col}: Filled {null_count} missing with mean")
            
            elif df_clean[col].dtype == 'object':
                if 'not' in str(col).lower() or 'no' in str(col).lower():
                    # Likely negative response
                    df_clean[col] = df_clean[col].fillna('no')
                else:
                    # Use mode
                    if not df_clean[col].mode().empty:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                        transformations.append(f"Categorical {col}: Filled {null_count} missing with mode")
    
    # 2B: Revolutionary Encoding for Agricultural Data
    print("\n🎯 Revolutionary Encoding for Survey Data:")
    
    # Define comprehensive encoding schemes
    likert_5_mapping = {
        'not_at_all': 1, 'never': 1,
        'slightly': 2, 'rarely': 2,
        'moderately': 3, 'occasionally': 3,
        'significantly': 4, 'frequently': 4,
        'completely': 5, 'always': 5
    }
    
    satisfaction_mapping = {
        'not_satisfied': 1,
        'slightly_satisfied': 2,
        'moderately_satisfied': 3,
        'very_satisfied': 4
    }
    
    accessibility_mapping = {
        'not_accessible': 1,
        'slightly_accessible': 2,
        'moderately_accessible': 3,
        'very_accessible': 4
    }
    
    # Auto-detect and encode
    encoded_columns = []
    for col in df_clean.columns:
        col_str = str(df_clean[col].iloc[0]) if len(df_clean) > 0 else ""
        
        # Check for Likert scales
        if any(term in col_str for term in likert_5_mapping.keys()):
            df_clean[f"{col}_encoded"] = df_clean[col].map(likert_5_mapping)
            encoded_columns.append(col)
        
        # Check for satisfaction
        elif any(term in col_str for term in satisfaction_mapping.keys()):
            df_clean[f"{col}_encoded"] = df_clean[col].map(satisfaction_mapping)
            encoded_columns.append(col)
    
    print(f"   Encoded {len(encoded_columns)} Likert/satisfaction variables")
    
    # 2C: Create Composite Indicators
    print("\n📊 Creating Composite Indicators:")
    
    # Social Capital Index
    social_cols = [col for col in df_clean.columns if any(term in str(col).lower() 
                  for term in ['trust', 'share', 'group', 'cooperat', 'network'])]
    
    if social_cols:
        # Get encoded versions
        social_encoded = [f"{col}_encoded" for col in social_cols if f"{col}_encoded" in df_clean.columns]
        if social_encoded:
            df_clean['social_capital_index'] = df_clean[social_encoded].mean(axis=1)
            print(f"   Created Social Capital Index from {len(social_encoded)} variables")
    
    # Resource Access Index
    resource_cols = [col for col in df_clean.columns if any(term in str(col).lower()
                    for term in ['access', 'resource', 'tool', 'input', 'seed'])]
    
    if resource_cols:
        resource_encoded = [f"{col}_encoded" for col in resource_cols if f"{col}_encoded" in df_clean.columns]
        if resource_encoded:
            df_clean['resource_access_index'] = df_clean[resource_encoded].mean(axis=1)
            print(f"   Created Resource Access Index from {len(resource_encoded)} variables")
    
    # 2D: Create Target Variable for Predictive Modeling
    if agricultural_context:
        # Look for yield/income/production columns
        outcome_cols = [col for col in df_clean.columns if any(term in str(col).lower()
                       for term in ['yield', 'income', 'production', 'profit', 'revenue'])]
        
        if outcome_cols:
            # Create binary target: high vs low performance
            for col in outcome_cols:
                if f"{col}_encoded" in df_clean.columns:
                    median_val = df_clean[f"{col}_encoded"].median()
                    df_clean['high_performance'] = (df_clean[f"{col}_encoded"] > median_val).astype(int)
                    print(f"   Created binary target: high_performance")
                    break
    
    return df_clean, transformations

# ============================================================
# STEP 3: REVOLUTIONARY FEATURE ENGINEERING
# ============================================================

def revolutionary_features(df, context='strawberry_farmers'):
    """
    Create powerful features for agricultural analysis
    """
    print(f"\n🔧 STEP 3: REVOLUTIONARY FEATURE ENGINEERING - {context}")
    print("-"*60)
    
    df_features = df.copy()
    created_features = []
    
    # 3A: Demographic Power Features
    print("👥 Demographic Power Features:")
    
    # Age parsing
    age_cols = [col for col in df_features.columns if 'age' in str(col).lower()]
    if age_cols:
        age_col = age_cols[0]
        
        # Extract numeric age from ranges
        def extract_age_numeric(age_str):
            if pd.isna(age_str):
                return np.nan
            age_str = str(age_str)
            
            if 'below' in age_str:
                return 25  # Approximate
            elif '30_40' in age_str:
                return 35
            elif '41_50' in age_str:
                return 45
            elif 'above' in age_str:
                return 55
            else:
                # Try to extract numbers
                import re
                numbers = re.findall(r'\d+', age_str)
                if numbers:
                    return float(numbers[0])
                return np.nan
        
        df_features['age_numeric'] = df_features[age_col].apply(extract_age_numeric)
        created_features.append('age_numeric')
        print(f"   Created: age_numeric from {age_col}")
    
    # 3B: Experience Features
    exp_cols = [col for col in df_features.columns if any(term in str(col).lower() 
                for term in ['experience', 'year', 'season'])]
    
    if exp_cols:
        exp_col = exp_cols[0]
        
        def extract_experience_years(exp_str):
            if pd.isna(exp_str):
                return np.nan
            
            exp_str = str(exp_str).lower()
            
            if 'less_than' in exp_str:
                return 1
            elif '2_5' in exp_str:
                return 3.5
            elif '6_10' in exp_str:
                return 8
            elif 'more_than' in exp_str or '10' in exp_str:
                return 12
            else:
                # Try to extract numbers
                import re
                numbers = re.findall(r'\d+', exp_str)
                if numbers:
                    return float(numbers[0])
                return np.nan
        
        df_features['experience_years'] = df_features[exp_col].apply(extract_experience_years)
        created_features.append('experience_years')
        print(f"   Created: experience_years from {exp_col}")
    
    # 3C: Farm Size Features
    size_cols = [col for col in df_features.columns if any(term in str(col).lower() 
                 for term in ['acre', 'hectare', 'size', 'farm'])]
    
    if size_cols:
        size_col = size_cols[0]
        
        def extract_farm_size_acres(size_str):
            if pd.isna(size_str):
                return np.nan
            
            size_str = str(size_str).lower()
            
            if 'less_than' in size_str and 'acre' in size_str:
                return 0.5
            elif '1_3' in size_str:
                return 2
            elif '4_6' in size_str:
                return 5
            elif 'more_than' in size_str and '6' in size_str:
                return 8
            else:
                return np.nan
        
        df_features['farm_size_acres'] = df_features[size_col].apply(extract_farm_size_acres)
        created_features.append('farm_size_acres')
        print(f"   Created: farm_size_acres from {size_col}")
    
    # 3D: Create Powerful Interaction Features
    print("\n💫 Creating Interaction Features:")
    
    # Farmer Sophistication Score
    if 'age_numeric' in df_features.columns and 'experience_years' in df_features.columns:
        df_features['farmer_sophistication'] = (
            df_features['age_numeric'] * 0.3 + 
            df_features['experience_years'] * 0.7
        )
        created_features.append('farmer_sophistication')
        print("   Created: farmer_sophistication (age × experience)")
    
    # Resource-Experience Interaction
    if 'experience_years' in df_features.columns and 'resource_access_index' in df_features.columns:
        df_features['experienced_resource_access'] = (
            df_features['experience_years'] * df_features['resource_access_index']
        )
        created_features.append('experienced_resource_access')
        print("   Created: experienced_resource_access")
    
    # 3E: Create Segments/Clusters
    print("\n🎯 Creating Farmer Segments:")
    
    # Segment by experience and resources
    if 'experience_years' in df_features.columns and 'resource_access_index' in df_features.columns:
        conditions = [
            (df_features['experience_years'] >= 5) & (df_features['resource_access_index'] >= 3),
            (df_features['experience_years'] < 5) & (df_features['resource_access_index'] >= 3),
            (df_features['experience_years'] >= 5) & (df_features['resource_access_index'] < 3),
            (df_features['experience_years'] < 5) & (df_features['resource_access_index'] < 3)
        ]
        
        choices = ['Experienced_HighResource', 'New_HighResource', 
                  'Experienced_LowResource', 'New_LowResource']
        
        df_features['farmer_segment'] = np.select(conditions, choices, default='Other')
        created_features.append('farmer_segment')
        print(f"   Created: farmer_segment with {df_features['farmer_segment'].nunique()} segments")
    
    print(f"\n✅ Created {len(created_features)} revolutionary features")
    
    return df_features, created_features

# ============================================================
# STEP 4: REVOLUTIONARY VISUALIZATION
# ============================================================

def revolutionary_visualizations(df, context='Strawberry Farmers'):
    """
    Create publication-quality visualizations
    """
    print(f"\n📊 STEP 4: REVOLUTIONARY VISUALIZATIONS - {context}")
    print("-"*60)
    
    # Create professional figure
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'African Agriculture Analytics: {context}', 
                 fontsize=24, fontweight='bold', y=1.02)
    
    # 4A: Social Capital Distribution
    ax1 = plt.subplot(3, 3, 1)
    if 'social_capital_index' in df.columns:
        sns.histplot(df['social_capital_index'], kde=True, color='blue', ax=ax1)
        mean_sc = df['social_capital_index'].mean()
        ax1.axvline(mean_sc, color='red', linestyle='--', label=f'Mean: {mean_sc:.2f}')
        ax1.set_title('Social Capital Index Distribution', fontweight='bold')
        ax1.set_xlabel('Social Capital Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
    
    # 4B: Resource Access by Experience
    ax2 = plt.subplot(3, 3, 2)
    if 'experience_years' in df.columns and 'resource_access_index' in df.columns:
        # Create experience bins
        df['exp_bin'] = pd.cut(df['experience_years'], bins=5)
        sns.boxplot(x='exp_bin', y='resource_access_index', data=df, ax=ax2, palette='viridis')
        ax2.set_title('Resource Access by Farming Experience', fontweight='bold')
        ax2.set_xlabel('Experience (years)')
        ax2.set_ylabel('Resource Access Index')
        ax2.tick_params(axis='x', rotation=45)
    
    # 4C: Farmer Segments
    ax3 = plt.subplot(3, 3, 3)
    if 'farmer_segment' in df.columns:
        segment_counts = df['farmer_segment'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        wedges, texts, autotexts = ax3.pie(segment_counts.values, labels=segment_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Farmer Segments Distribution', fontweight='bold')
        plt.setp(autotexts, size=10, weight="bold")
    
    # 4D: Correlation Heatmap
    ax4 = plt.subplot(3, 3, 4)
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 2:
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax4, cbar_kws={"shrink": 0.8})
        ax4.set_title('Feature Correlation Matrix', fontweight='bold')
    
    # 4E: Age vs Social Capital
    ax5 = plt.subplot(3, 3, 5)
    if 'age_numeric' in df.columns and 'social_capital_index' in df.columns:
        sns.scatterplot(x='age_numeric', y='social_capital_index', data=df, 
                       hue='farmer_segment' if 'farmer_segment' in df.columns else None,
                       alpha=0.7, ax=ax5)
        ax5.set_title('Age vs Social Capital', fontweight='bold')
        ax5.set_xlabel('Age (years)')
        ax5.set_ylabel('Social Capital Index')
    
    # 4F: Performance by Segment
    ax6 = plt.subplot(3, 3, 6)
    if 'farmer_segment' in df.columns and 'high_performance' in df.columns:
        performance_by_segment = df.groupby('farmer_segment')['high_performance'].mean().sort_values()
        performance_by_segment.plot(kind='barh', color='green', ax=ax6)
        ax6.set_title('High Performance Rate by Segment', fontweight='bold')
        ax6.set_xlabel('Proportion High Performing')
        ax6.set_ylabel('Farmer Segment')
    
    plt.tight_layout()
    plt.show()
    
    # 4G: Advanced Analysis Visualization
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Network Analysis (if group membership data exists)
    ax7 = axes[0]
    group_cols = [col for col in df.columns if any(term in str(col).lower() 
                 for term in ['group', 'member', 'association'])]
    
    if group_cols:
        group_col = group_cols[0]
        if df[group_col].dtype == 'object':
            group_counts = df[group_col].value_counts()
            group_counts.plot(kind='bar', color='purple', ax=ax7)
            ax7.set_title('Farmer Group Membership', fontweight='bold')
            ax7.set_xlabel('Group Membership')
            ax7.set_ylabel('Count')
            ax7.tick_params(axis='x', rotation=45)
    
    # Training Impact Analysis
    ax8 = axes[1]
    training_cols = [col for col in df.columns if any(term in str(col).lower() 
                    for term in ['training', 'train', 'workshop'])]
    
    if training_cols and 'high_performance' in df.columns:
        training_col = training_cols[0]
        if df[training_col].nunique() <= 5:  # Likely yes/no or frequency
            performance_by_training = df.groupby(training_col)['high_performance'].mean()
            performance_by_training.plot(kind='bar', color='orange', ax=ax8)
            ax8.set_title('Impact of Training on Performance', fontweight='bold')
            ax8.set_xlabel('Received Training')
            ax8.set_ylabel('Proportion High Performing')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Created 8 revolutionary visualizations")

# ============================================================
# STEP 5: PREDICTIVE MODELING REVOLUTION
# ============================================================

def revolutionary_modeling(df, target_col='high_performance'):
    """
    Advanced ML modeling for agricultural predictions
    """
    print(f"\n🤖 STEP 5: PREDICTIVE MODELING REVOLUTION")
    print("-"*60)
    
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        import xgboost as xgb
        
        # Prepare data
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df[target_col] if target_col in df.columns else None
        
        if y is None:
            print("⚠️ No target column found. Creating synthetic target for demonstration.")
            # Create synthetic target for demonstration
            if 'social_capital_index' in df.columns:
                median_sc = df['social_capital_index'].median()
                y = (df['social_capital_index'] > median_sc).astype(int)
                X = X.drop(columns=['social_capital_index'], errors='ignore')
            else:
                print("❌ Cannot create synthetic target. Skipping modeling.")
                return None
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Modeling Setup:")
        print(f"   Features: {X.shape[1]}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        print(f"   Positive class proportion: {y.mean():.2%}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        
        # Train and evaluate models
        results = {}
        best_model = None
        best_score = 0
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(models.items()):
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            # Update best model
            if auc > best_score:
                best_score = auc
                best_model = model
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.3f}, AUC: {auc:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print results
        print("\n🏆 MODEL PERFORMANCE SUMMARY:")
        print("-"*50)
        print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<10} {'CV Mean':<10} {'CV Std':<10}")
        print("-"*50)
        
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['auc']:<10.3f} "
                  f"{metrics['cv_mean']:<10.3f} {metrics['cv_std']:<10.3f}")
        
        # Feature Importance from best model
        if hasattr(best_model, 'feature_importances_'):
            print("\n📊 TOP 10 IMPORTANT FEATURES:")
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(10).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            top_features = feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.gca().invert_yaxis()
            plt.title('Top 10 Most Important Features', fontweight='bold')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
        
        return results, best_model
        
    except Exception as e:
        print(f"⚠️ Modeling skipped due to: {str(e)}")
        return None

# ============================================================
# STEP 6: MONETIZATION & VALUE CREATION
# ============================================================

def revolutionary_monetization(df, insights, model_results, context):
    """
    Turn analysis into revenue-generating opportunities
    """
    print(f"\n💰 STEP 6: REVOLUTIONARY MONETIZATION STRATEGY")
    print("-"*60)
    
    # 6A: Identify Money-Making Opportunities
    opportunities = [
        {
            'product': 'Farmer Segmentation Dashboard',
            'description': 'Interactive dashboard showing farmer segments, performance, and recommendations',
            'target_clients': ['NGOs', 'Government Agencies', 'Agribusiness Companies'],
            'pricing': '$1,500 - $5,000 one-time + $200/month maintenance',
            'development_time': '2-3 weeks',
            'resources_needed': ['Streamlit', 'Plotly', 'Heroku/AWS']
        },
        {
            'product': 'Predictive Analytics Service',
            'description': 'ML model predicting which farmers need intervention, with API access',
            'target_clients': ['Microfinance Institutions', 'Input Suppliers', 'Insurance Companies'],
            'pricing': '$300/month subscription',
            'development_time': '3-4 weeks',
            'resources_needed': ['FastAPI', 'Scikit-learn/XGBoost', 'Docker']
        },
        {
            'product': 'Research & Impact Report',
            'description': 'Professional report with actionable insights for donors/funders',
            'target_clients': ['International Donors', 'Research Institutions', 'Government'],
            'pricing': '$2,000 - $10,000 per report',
            'development_time': '1-2 weeks',
            'resources_needed': ['LaTeX/R Markdown', 'Data visualization skills']
        },
        {
            'product': 'Training Program Design',
            'description': 'Customized training curriculum based on data insights',
            'target_clients': ['Extension Services', 'Farmer Cooperatives', 'NGOs'],
            'pricing': '$3,000 - $8,000 per program',
            'development_time': '2-3 weeks',
            'resources_needed': ['Instructional design', 'Subject matter expertise']
        }
    ]
    
    # 6B: Print Opportunities
    print("🚀 MONEY-MAKING OPPORTUNITIES:\n")
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp['product']}")
        print(f"   📝 {opp['description']}")
        print(f"   🎯 Clients: {', '.join(opp['target_clients'])}")
        print(f"   💰 Price: {opp['pricing']}")
        print(f"   ⏱️ Development: {opp['development_time']}")
        print()
    
    # 6C: Immediate Action Plan
    print("🎯 IMMEDIATE 7-DAY ACTION PLAN:")
    print("   Day 1-2: Create GitHub portfolio with this analysis")
    print("   Day 3: Write LinkedIn post about findings")
    print("   Day 4: Create simple Streamlit dashboard")
    print("   Day 5: Reach out to 3 potential clients")
    print("   Day 6: Write blog post on Medium")
    print("   Day 7: Join agricultural data science communities")
    
    # 6D: Pricing Strategy
    print("\n💵 PRICING STRATEGY FOR AFRICAN MARKET:")
    print("   • NGOs/Government: $1,000 - $5,000 per project")
    print("   • Agribusiness: $300 - $1,500 monthly subscription")
    print("   • International Donors: $5,000 - $20,000 per research")
    print("   • Consulting: $50 - $150 per hour")
    
    return opportunities

# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def execute_revolution(data_path, dataset_name="Agricultural Data"):
    """
    Execute the complete revolutionary pipeline
    """
    print("\n" + "="*80)
    print("🚀 EXECUTING AFRICAN AGRICULTURE DATA SCIENCE REVOLUTION")
    print("="*80)
    
    # Load data
    try:
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Step 0: Mindset
    AfricanAgricultureRevolution.mindset()
    
    # Step 1: Diagnose
    insights = diagnose_data(df, dataset_name)
    
    # Step 2: Clean
    df_clean, transformations = revolutionary_clean(df)
    
    # Step 3: Engineer Features
    df_features, features_created = revolutionary_features(df_clean, dataset_name)
    
    # Step 4: Visualize
    revolutionary_visualizations(df_features, dataset_name)
    
    # Step 5: Model
    model_results = revolutionary_modeling(df_features)
    
    # Step 6: Monetize
    opportunities = revolutionary_monetization(df_features, insights, model_results, dataset_name)
    
    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results_{dataset_name.replace(' ', '_')}_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_features.to_excel(writer, sheet_name='Processed_Data', index=False)
        
        # Save summary
        summary = pd.DataFrame({
            'Metric': ['Total Farmers', 'Features Created', 'Social Capital Avg', 
                      'Resource Access Avg', 'High Performance %'],
            'Value': [len(df_features), len(features_created),
                     df_features.get('social_capital_index', pd.Series([0])).mean(),
                     df_features.get('resource_access_index', pd.Series([0])).mean(),
                     df_features.get('high_performance', pd.Series([0])).mean() * 100]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\n✅ REVOLUTION COMPLETE!")
    print(f"📁 Results saved to: {output_path}")
    print(f"📊 Processed {len(df_features)} farmers with {df_features.shape[1]} features")
    
    return {
        'raw_data': df,
        'cleaned_data': df_clean,
        'featured_data': df_features,
        'model_results': model_results,
        'opportunities': opportunities,
        'output_file': output_path
    }

# ============================================================
# GOOGLE COLAB INTEGRATION FUNCTIONS
# ============================================================

def setup_colab_environment():
    """
    Setup Google Colab environment for African agriculture analysis
    """
    print("Setting up Google Colab environment...")
    
    colab_code = """
# ============================================
# AFRICAN AGRICULTURE DATA SCIENCE - COLAB SETUP
# ============================================

# Install required packages
!pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost openpyxl -q
!pip install streamlit -q

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone your GitHub repository
!git clone https://github.com/YOUR_USERNAME/African-Agriculture-Revolution.git

# Navigate to your project
%cd /content/African-Agriculture-Revolution

print("✅ Environment setup complete!")
print("📁 Files available:")
!ls -la
    """
    
    return colab_code

def create_colab_notebook():
    """
    Create a Google Colab notebook template
    """
    notebook_content = {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '# 🌍 African Agriculture Data Science Revolution\n',
                    '## From SPSS/Stata to Modern Data Science\n',
                    '### Google Colab + GitHub Integration\n'
                ]
            },
            {
                'cell_type': 'code',
                'metadata': {},
                'source': [
                    '# Install and import all required packages\n',
                    '!pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost streamlit -q\n',
                    'import pandas as pd\n',
                    'import numpy as np\n',
                    'import matplotlib.pyplot as plt\n',
                    'import seaborn as sns\n',
                    'import warnings\n',
                    'warnings.filterwarnings("ignore")\n',
                    '\n',
                    'print("✅ Packages installed and imported!")'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 🔗 Connect to GitHub\n',
                    'Clone your repository to access all scripts'
                ]
            },
            {
                'cell_type': 'code',
                'metadata': {},
                'source': [
                    '# Clone your GitHub repository\n',
                    '!git clone https://github.com/YOUR_USERNAME/African-Agriculture-Revolution.git\n',
                    '%cd African-Agriculture-Revolution\n',
                    '!ls'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 📊 Load Your Strawberry Farmer Data\n',
                    'Replace with your actual data path'
                ]
            },
            {
                'cell_type': 'code',
                'metadata': {},
                'source': [
                    '# Load your data\n',
                    'from scripts.secret_recipe_template import execute_revolution\n',
                    '\n',
                    '# Execute the complete pipeline\n',
                    'results = execute_revolution(\n',
                    '    data_path="data/Strawberry_Farmer_Data.xlsx",\n',
                    '    dataset_name="Tanzania Strawberry Farmers"\n',
                    ')'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 📈 Create Interactive Dashboard\n',
                    'Generate an interactive report'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Create interactive visualizations\n',
                    'import plotly.express as px\n',
                    '\n',
                    '# Assuming df_features is in results\n',
                    'df_features = results[\'featured_data\']\n',
                    '\n',
                    '# Interactive scatter plot\n',
                    'if \'age_numeric\' in df_features.columns and \'social_capital_index\' in df_features.columns:\n',
                    '    fig = px.scatter(df_features, x=\'age_numeric\', y=\'social_capital_index\',\n',
                    '                     color=\'farmer_segment\' if \'farmer_segment\' in df_features.columns else None,\n',
                    '                     title=\'Age vs Social Capital Index\',\n',
                    '                     hover_data=[\'experience_years\', \'resource_access_index\'])\n',
                    '    fig.show()'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 💰 Monetization Strategy\n',
                    'Review the money-making opportunities'
                ]
            },
            {
                'cell_type': 'code',
                'metadata': {},
                'source': [
                    '# Review monetization opportunities\n',
                    'opportunities = results[\'opportunities\']\n',
                    'print("🚀 MONEY-MAKING OPPORTUNITIES FOUND:")\n',
                    'for i, opp in enumerate(opportunities, 1):\n',
                    '    print(f"{i}. {opp[\'product\']} - {opp[\'pricing\']}")'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 📤 Save and Share Results\n',
                    'Save to GitHub and create shareable reports'
                ]
            },
            {
                'cell_type': 'code',
                'metadata': {},
                'source': [
                    '# Save to GitHub\n',
                    '!git add .\n',
                    '!git commit -m "Colab analysis: Strawberry farmer insights"\n',
                    '!git push origin main\n',
                    '\n',
                    'print("✅ Results saved to GitHub!")'
                ]
            }
        ]
    }
    
    return notebook_content

# ============================================================
# GITHUB INTEGRATION GUIDE
# ============================================================

def github_integration_guide():
    """
    Guide for integrating Google Colab with GitHub
    """
    guide = """
# 📚 GOOGLE COLAB + GITHUB INTEGRATION GUIDE

## 1. INITIAL SETUP ON YOUR LAPTOP
1. Save all scripts to: C:\\Users\\hp\\OneDrive\\Documents\\African_Agriculture_Revolution
2. Initialize Git repository: