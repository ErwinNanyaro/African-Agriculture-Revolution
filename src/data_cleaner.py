"""
🧹 REVOLUTIONARY DATA CLEANER FOR AFRICAN AGRICULTURE SURVEYS
Intelligent cleaning preserving farmers' voices
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

class AfricanAgricultureDataCleaner:
    """Revolutionary cleaner for agricultural survey data"""
    
    def __init__(self):
        # Comprehensive encoding schemes for African agricultural data
        self.likert_5_mapping = {
            'not_at_all': 1, 'never': 1, 'not_effective': 1,
            'slightly': 2, 'rarely': 2, 'slightly_effective': 2,
            'moderately': 3, 'occasionally': 3, 'moderately_effective': 3,
            'significantly': 4, 'frequently': 4, 'very_effective': 4,
            'completely': 5, 'always': 5
        }
        
        self.satisfaction_mapping = {
            'not_satisfied': 1,
            'slightly_satisfied': 2,
            'moderately_satisfied': 3,
            'very_satisfied': 4
        }
        
        self.accessibility_mapping = {
            'not_accessible': 1,
            'slightly_accessible': 2,
            'moderately_accessible': 3,
            'very_accessible': 4
        }
        
        self.yield_mapping = {
            'very_low': 1,
            'low': 2,
            'moderate': 3,
            'high': 4,
            'very_high': 5
        }
        
    def clean_dataset(self, df: pd.DataFrame, dataset_type: str = 'strawberry') -> Tuple[pd.DataFrame, Dict]:
        """
        Revolutionary cleaning pipeline
        
        Args:
            df: Raw DataFrame
            dataset_type: Type of dataset ('strawberry' or 'arusha')
            
        Returns:
            Cleaned DataFrame and cleaning report
        """
        print(f"\n🧹 REVOLUTIONARY CLEANING PIPELINE - {dataset_type.upper()}")
        print("-"*60)
        
        df_clean = df.copy()
        cleaning_report = {
            'missing_values_before': df.isnull().sum().sum(),
            'transformations_applied': [],
            'issues_fixed': []
        }
        
        # Step 1: Intelligent Missing Value Imputation
        df_clean = self._handle_missing_values(df_clean, cleaning_report)
        
        # Step 2: Revolutionary Encoding
        df_clean = self._encode_survey_responses(df_clean, cleaning_report)
        
        # Step 3: Create Social Capital Index
        df_clean = self._create_social_capital_index(df_clean, cleaning_report)
        
        # Step 4: Create Target Variables
        df_clean = self._create_target_variables(df_clean, cleaning_report)
        
        # Step 5: Standardize Column Names
        df_clean = self._standardize_columns(df_clean)
        
        # Generate final report
        cleaning_report['missing_values_after'] = df_clean.isnull().sum().sum()
        cleaning_report['cleaning_success'] = (
            cleaning_report['missing_values_before'] - cleaning_report['missing_values_after']
        ) / cleaning_report['missing_values_before'] * 100 if cleaning_report['missing_values_before'] > 0 else 100
        
        self._print_cleaning_report(cleaning_report)
        
        return df_clean, cleaning_report
    
    def _handle_missing_values(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Intelligent missing value imputation"""
        print("   🔧 Intelligent Missing Value Handling...")
        
        df_clean = df.copy()
        strategies_applied = []
        
        for col in df_clean.columns:
            null_count = df_clean[col].isnull().sum()
            
            if null_count > 0:
                null_pct = (null_count / len(df_clean)) * 100
                
                # Strategy based on column type and content
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Numeric columns
                    if null_pct < 10:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        strategies_applied.append(f"Numeric {col}: Median imputation ({null_count} values)")
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                        strategies_applied.append(f"Numeric {col}: Mean imputation ({null_count} values)")
                
                elif df_clean[col].dtype == 'object':
                    # Categorical columns - strategy based on content
                    col_str = str(df_clean[col].iloc[0]) if len(df_clean[col].dropna()) > 0 else ""
                    
                    if 'no' in col_str.lower() or 'not' in col_str.lower():
                        df_clean[col] = df_clean[col].fillna('no')
                        strategies_applied.append(f"Categorical {col}: Filled with 'no' ({null_count} values)")
                    elif any(term in col_str.lower() for term in ['yes', 'always', 'completely']):
                        df_clean[col] = df_clean[col].fillna('no')  # Conservative approach
                        strategies_applied.append(f"Categorical {col}: Conservative fill ({null_count} values)")
                    else:
                        if not df_clean[col].mode().empty:
                            mode_val = df_clean[col].mode()[0]
                            df_clean[col] = df_clean[col].fillna(mode_val)
                            strategies_applied.append(f"Categorical {col}: Mode imputation ({null_count} values)")
        
        report['transformations_applied'].extend(strategies_applied)
        print(f"      Applied {len(strategies_applied)} imputation strategies")
        
        return df_clean
    
    def _encode_survey_responses(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Revolutionary encoding of survey responses"""
        print("   🎯 Revolutionary Survey Response Encoding...")
        
        df_clean = df.copy()
        encoded_columns = []
        
        # Auto-detect and encode Likert scales
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Check sample values
                sample_values = df_clean[col].dropna().unique()[:5]
                
                # Check for Likert 5-point scale
                if any(str(val) in self.likert_5_mapping for val in sample_values):
                    df_clean[f"{col}_encoded"] = df_clean[col].map(self.likert_5_mapping)
                    encoded_columns.append(col)
                
                # Check for satisfaction scale
                elif any(str(val) in self.satisfaction_mapping for val in sample_values):
                    df_clean[f"{col}_encoded"] = df_clean[col].map(self.satisfaction_mapping)
                    encoded_columns.append(col)
                
                # Check for accessibility scale
                elif any(str(val) in self.accessibility_mapping for val in sample_values):
                    df_clean[f"{col}_encoded"] = df_clean[col].map(self.accessibility_mapping)
                    encoded_columns.append(col)
                
                # Check for yield scale
                elif any(str(val) in self.yield_mapping for val in sample_values):
                    df_clean[f"{col}_encoded"] = df_clean[col].map(self.yield_mapping)
                    encoded_columns.append(col)
        
        # Encode binary variables
        binary_cols = [col for col in df_clean.columns if df_clean[col].nunique() == 2]
        for col in binary_cols:
            if df_clean[col].dtype == 'object':
                unique_vals = df_clean[col].unique()
                if 'yes' in unique_vals or 'no' in unique_vals:
                    df_clean[f"{col}_encoded"] = df_clean[col].map({'yes': 1, 'no': 0})
                    encoded_columns.append(col)
        
        report['transformations_applied'].append(f"Encoded {len(encoded_columns)} survey response columns")
        print(f"      Encoded {len(encoded_columns)} columns")
        
        return df_clean
    
    def _create_social_capital_index(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Create comprehensive social capital index"""
        print("   🤝 Creating Social Capital Index...")
        
        df_clean = df.copy()
        
        # Identify social capital components
        social_components = []
        
        # Trust-related columns
        trust_cols = [col for col in df_clean.columns if 'trust' in str(col).lower()]
        if trust_cols:
            trust_encoded = [f"{col}_encoded" for col in trust_cols if f"{col}_encoded" in df_clean.columns]
            social_components.extend(trust_encoded)
        
        # Information sharing
        share_cols = [col for col in df_clean.columns if any(term in str(col).lower() 
                     for term in ['share', 'exchange', 'information'])]
        if share_cols:
            share_encoded = [f"{col}_encoded" for col in share_cols if f"{col}_encoded" in df_clean.columns]
            social_components.extend(share_encoded)
        
        # Group participation
        group_cols = [col for col in df_clean.columns if any(term in str(col).lower() 
                     for term in ['group', 'meeting', 'association', 'network'])]
        if group_cols:
            group_encoded = [f"{col}_encoded" for col in group_cols if f"{col}_encoded" in df_clean.columns]
            social_components.extend(group_encoded)
        
        # Collective action
        collective_cols = [col for col in df_clean.columns if 'collective' in str(col).lower()]
        if collective_cols:
            collective_encoded = [f"{col}_encoded" for col in collective_cols if f"{col}_encoded" in df_clean.columns]
            social_components.extend(collective_encoded)
        
        # Calculate social capital index if we have components
        if social_components:
            # Filter to columns that exist
            existing_components = [col for col in social_components if col in df_clean.columns]
            
            if existing_components:
                df_clean['social_capital_index'] = df_clean[existing_components].mean(axis=1)
                
                # Normalize to 0-100 scale for interpretability
                df_clean['social_capital_score'] = (
                    (df_clean['social_capital_index'] - df_clean['social_capital_index'].min()) / 
                    (df_clean['social_capital_index'].max() - df_clean['social_capital_index'].min())
                ) * 100
                
                report['transformations_applied'].append(
                    f"Created Social Capital Index from {len(existing_components)} components"
                )
                print(f"      Created index from {len(existing_components)} social capital components")
        
        return df_clean
    
    def _create_target_variables(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Create target variables for predictive modeling"""
        print("   🎯 Creating Target Variables...")
        
        df_clean = df.copy()
        
        # Look for yield-related columns
        yield_cols = [col for col in df_clean.columns if 'yield' in str(col).lower()]
        
        for col in yield_cols:
            if f"{col}_encoded" in df_clean.columns:
                # Create binary target: high vs low yield
                median_yield = df_clean[f"{col}_encoded"].median()
                df_clean['high_yield'] = (df_clean[f"{col}_encoded"] > median_yield).astype(int)
                report['transformations_applied'].append(f"Created binary target: high_yield from {col}")
                print(f"      Created high_yield target variable")
                break
        
        # Look for yield change columns
        change_cols = [col for col in df_clean.columns if 'change' in str(col).lower()]
        
        for col in change_cols:
            if col in df_clean.columns and df_clean[col].dtype == 'object':
                # Map yield changes to numeric
                change_mapping = {
                    'decreased': -1,
                    'remained_the_same': 0,
                    'increased': 1
                }
                
                # Check if values match our mapping
                sample_values = df_clean[col].dropna().unique()
                if any(str(val) in change_mapping for val in sample_values):
                    df_clean['yield_change_numeric'] = df_clean[col].map(change_mapping)
                    report['transformations_applied'].append(f"Mapped yield change to numeric from {col}")
                    print(f"      Created yield_change_numeric variable")
                    break
        
        return df_clean
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency"""
        df_clean = df.copy()
        
        # Remove special characters and standardize
        new_columns = {}
        for col in df_clean.columns:
            new_col = str(col).lower().replace(' ', '_').replace('-', '_')
            new_col = re.sub(r'[^a-z0-9_]', '', new_col)
            new_columns[col] = new_col
        
        df_clean = df_clean.rename(columns=new_columns)
        
        return df_clean
    
    def _print_cleaning_report(self, report: Dict):
        """Print comprehensive cleaning report"""
        print("\n📋 CLEANING REPORT:")
        print(f"   Missing values before: {report['missing_values_before']}")
        print(f"   Missing values after: {report['missing_values_after']}")
        print(f"   Cleaning success: {report['cleaning_success']:.1f}%")
        
        print("\n🔄 Transformations Applied:")
        for transformation in report['transformations_applied'][:10]:  # Show first 10
            print(f"   • {transformation}")
        
        if len(report['transformations_applied']) > 10:
            print(f"   ... and {len(report['transformations_applied']) - 10} more")

# Example usage
if __name__ == "__main__":
    # Load sample data
    from data_loader import AfricanAgricultureDataLoader
    
    loader = AfricanAgricultureDataLoader()
    data = loader.load_strawberry_data(sample_size=100)
    
    if not data.empty:
        cleaner = AfricanAgricultureDataCleaner()
        cleaned_data, report = cleaner.clean_dataset(data, 'strawberry')
        
        print(f"\n✅ Revolution complete!")
        print(f"   Original shape: {data.shape}")
        print(f"   Cleaned shape: {cleaned_data.shape}")
        print(f"   New columns: {[col for col in cleaned_data.columns if 'encoded' in col or 'index' in col or 'score' in col]}")