"""
🌍 REVOLUTIONARY DATA LOADER FOR AFRICAN AGRICULTURE
Transforming raw survey data into actionable intelligence
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AfricanAgricultureDataLoader:
    """Revolutionary data loader for Tanzanian farmer data"""
    
    def __init__(self, config_path: str = "config/paths.yaml"):
        """
        Initialize with revolutionary mindset
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_paths = self.config['data_paths']
        
        print("="*80)
        print("🌍 AFRICAN AGRICULTURE DATA SCIENCE REVOLUTION")
        print("="*80)
        print("\n💡 MINDSET: We're listening to farmers through their data")
        print("🎯 MISSION: Transform African agriculture with data science")
        print("💰 GOAL: Create solutions that organizations will pay for")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Create default config
            default_config = {
                'data_paths': {
                    'strawberry': 'data/raw/strawberry_farmers_Morogoro.csv',
                    'arusha': 'data/raw/both_crops_farmers_Arusha.csv'
                },
                'analysis': {
                    'target_column': 'yield_change_numeric',
                    'problem_type': 'classification'
                }
            }
            return default_config
    
    def load_strawberry_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load Morogoro strawberry farmer data with revolutionary insights
        
        Args:
            sample_size: Number of samples to load (None for all)
            
        Returns:
            DataFrame with farmer survey data
        """
        print("\n🍓 LOADING MOROGORO STRAWBERRY FARMER DATA")
        print("-"*60)
        
        try:
            df = pd.read_csv(self.data_paths['strawberry'])
            
            # Rename columns for revolutionary analysis
            df = self._rename_strawberry_columns(df)
            
            print(f"✅ Data loaded successfully!")
            print(f"📊 Shape: {df.shape[0]} farmers × {df.shape[1]} variables")
            print(f"📅 Columns: {list(df.columns[:5])}...")
            
            # Initial revolutionary insights
            self._generate_initial_insights(df, "Morogoro Strawberry Farmers")
            
            return df.sample(sample_size) if sample_size else df
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure your data is in: C:/Users/hp/OneDrive/Documents/African_Agriculture_Revolution/data/")
            return pd.DataFrame()
    
    def load_arusha_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load Arusha multi-crop farmer data
        
        Args:
            sample_size: Number of samples to load
            
        Returns:
            DataFrame with farmer survey data
        """
        print("\n🌽 LOADING ARUSHA MULTI-CROP FARMER DATA")
        print("-"*60)
        
        try:
            df = pd.read_csv(self.data_paths['arusha'])
            
            # Rename columns for consistency
            df = self._rename_arusha_columns(df)
            
            print(f"✅ Data loaded successfully!")
            print(f"📊 Shape: {df.shape[0]} farmers × {df.shape[1]} variables")
            print(f"📅 Columns: {list(df.columns[:5])}...")
            
            # Initial revolutionary insights
            self._generate_initial_insights(df, "Arusha Multi-Crop Farmers")
            
            return df.sample(sample_size) if sample_size else df
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()
    
    def _rename_strawberry_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revolutionary column renaming for strawberry data"""
        column_mapping = {
            '_1_What_is_your_age': 'age_group',
            '_2_What_is_your_level_of_education': 'education',
            '_3_How_many_years_have_you_been_engaged_in_strawberry_farming': 'experience',
            '_4_How_large_is_your_strawberry_farm': 'farm_size',
            '_5_Are_you_a_member_of_any_farmer_network_or_association': 'group_member',
            '_6_How_often_do_you_attend_meetings_or_activities_organized_by_your_farmer_association': 'meeting_frequency',
            '_7_Do_you_think_being_a_part_of_a_farmer_network_has_improved_your_farming_practices': 'network_benefit',
            '_8_How_do_you_rate_the_support_you_receive_from_farmer_associations_in_terms_of_resources': 'support_rating',
            '_9_How_much_do_you_trust_other_strawberry_farmers_in_your_community': 'trust_level',
            '_10_How_often_do_you_exchange_farming_information_with_other_farmers': 'info_exchange_freq',
            '_11_Do_you_believe_the_information_shared_by_other_farmers_has_positively_impacted_your_strawberry_yields': 'info_impact',
            '_12_How_willing_are_you_to_share_your_own_farming_knowledge_with_other_strawberry_farmers': 'sharing_willingness',
            '_13_Do_you_participate_in_any_group_activities_related_to_strawberry_farming': 'group_activities',
            '_14_How_often_does_your_community_engage_in_collective_action_to_solve_agricultural_challenges': 'collective_action_freq',
            '_15_How_effective_has_collective_action_been_in_improving_strawberry_farming_outcomes_in_your_community': 'collective_action_effect',
            '_16_Does_your_community_provide_any_form_of_support_during_farming_crises': 'crisis_support',
            '_17_Do_you_use_organic_or_ecological_farming_practices': 'organic_farming',
            '_18_How_satisfied_are_you_with_the_quality_of_your_soil_and_land_for_strawberry_farming': 'soil_satisfaction',
            '_19_How_accessible_are_agricultural_resources_in_your_region': 'resource_accessibility',
            '_20_Have_you_received_any_formal_training_on_sustainable_strawberry_farming_techniques': 'received_training',
            '_21_How_would_you_describe_your_strawberry_yields_over_the_past_year': 'yield_description',
            '_22_Has_your_strawberry_yield_increased_or_decreased_over_the_past_3_years': 'yield_change_3yrs',
            '_23_What_do_you_think_is_the_most_significant_factor_limiting_your_strawberry_yields': 'yield_limiting_factor',
            '_24_How_would_you_describe_your_strawberry_yields_over_the_past year': 'yield_trend'
        }
        
        # Apply mapping for columns that exist
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
        
        print(f"📝 Renamed {len(existing_mapping)} columns for revolutionary analysis")
        
        return df
    
    def _rename_arusha_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revolutionary column renaming for Arusha data"""
        # Simplified renaming - you'll need to customize based on actual columns
        column_mapping = {}
        
        # Auto-detect and rename
        for col in df.columns:
            if 'age' in str(col).lower():
                column_mapping[col] = 'age'
            elif 'education' in str(col).lower():
                column_mapping[col] = 'education'
            elif 'size' in str(col).lower() and 'land' in str(col).lower():
                column_mapping[col] = 'farm_size'
            elif 'extension' in str(col).lower():
                column_mapping[col] = 'extension_access'
            elif 'training' in str(col).lower():
                column_mapping[col] = 'training_frequency'
            elif 'climate' in str(col).lower():
                column_mapping[col] = 'climate_strategy'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"📝 Renamed {len(column_mapping)} columns")
        
        return df
    
    def _generate_initial_insights(self, df: pd.DataFrame, dataset_name: str):
        """Generate revolutionary initial insights"""
        print(f"\n💡 INITIAL REVOLUTIONARY INSIGHTS - {dataset_name}:")
        
        # Insight 1: Dataset size
        print(f"   1. 📊 {df.shape[0]} farmers represented in the data")
        
        # Insight 2: Missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        print(f"   2. 🧹 Data completeness: {100 - missing_pct:.1f}%")
        
        # Insight 3: Data types
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = df.select_dtypes(include=['object']).shape[1]
        print(f"   3. 🔢 Data composition: {numeric_cols} numeric, {categorical_cols} categorical variables")
        
        # Insight 4: Unique farmers by key characteristic
        if 'age_group' in df.columns:
            age_groups = df['age_group'].nunique()
            print(f"   4. 👥 {age_groups} different age groups represented")
        
        # Insight 5: Social capital indicators
        social_cols = [col for col in df.columns if any(term in str(col).lower() 
                      for term in ['trust', 'share', 'group', 'cooperat'])]
        if social_cols:
            print(f"   5. 🤝 {len(social_cols)} social capital indicators detected")
    
    def merge_datasets(self, strawberry_df: pd.DataFrame, arusha_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge datasets for comprehensive analysis
        
        Args:
            strawberry_df: Morogoro strawberry data
            arusha_df: Arusha multi-crop data
            
        Returns:
            Merged DataFrame with Tanzanian farmers
        """
        print("\n🔄 MERGING TANZANIAN FARMER DATASETS")
        print("-"*60)
        
        # Add region identifier
        strawberry_df['region'] = 'Morogoro'
        strawberry_df['crop_type'] = 'Strawberry'
        
        arusha_df['region'] = 'Arusha'
        arusha_df['crop_type'] = 'Mixed Crops'
        
        # Find common columns
        common_cols = list(set(strawberry_df.columns) & set(arusha_df.columns))
        
        if common_cols:
            print(f"🤝 Found {len(common_cols)} common columns for merging")
            
            # Merge on common columns
            merged_df = pd.concat([
                strawberry_df[common_cols + ['region', 'crop_type']],
                arusha_df[common_cols + ['region', 'crop_type']]
            ], ignore_index=True)
            
            print(f"✅ Merged dataset: {len(merged_df)} total farmers")
            print(f"   Morogoro: {len(strawberry_df)} strawberry farmers")
            print(f"   Arusha: {len(arusha_df)} multi-crop farmers")
            
            return merged_df
        else:
            print("⚠️ No common columns found for merging")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Initialize revolutionary loader
    loader = AfricanAgricultureDataLoader()
    
    # Load strawberry data
    strawberry_data = loader.load_strawberry_data()
    
    # Load Arusha data
    arusha_data = loader.load_arusha_data()
    
    # Merge if both loaded successfully
    if not strawberry_data.empty and not arusha_data.empty:
        all_farmers = loader.merge_datasets(strawberry_data, arusha_data)
        print(f"\n🎯 Total Tanzanian farmers in analysis: {len(all_farmers)}")