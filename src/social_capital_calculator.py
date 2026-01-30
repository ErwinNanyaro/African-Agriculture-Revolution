"""
🤝 REVOLUTIONARY SOCIAL CAPITAL CALCULATOR
Quantifying the invisible networks that drive agricultural success
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
import matplotlib.pyplot as plt

class SocialCapitalCalculator:
    """
    Revolutionary calculator for social capital in African agriculture
    
    Based on your conceptual framework:
    - Networks
    - Trust
    - Information Sharing
    - Cooperation
    - Inclusion
    """
    
    def __init__(self):
        self.components = {
            'structural': ['network_size', 'network_density', 'centrality'],
            'relational': ['trust', 'norms', 'obligations'],
            'cognitive': ['shared_language', 'shared_narratives', 'shared_goals']
        }
        
        self.metrics = {}
    
    def calculate_comprehensive_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive social capital index
        
        Args:
            df: DataFrame with cleaned survey data
            
        Returns:
            DataFrame with social capital metrics
        """
        print("\n🤝 CALCULATING REVOLUTIONARY SOCIAL CAPITAL INDEX")
        print("-"*60)
        
        df_sc = df.copy()
        
        # 1. Structural Social Capital (Networks)
        print("   📊 Calculating Structural Social Capital...")
        df_sc = self._calculate_structural_capital(df_sc)
        
        # 2. Relational Social Capital (Trust)
        print("   🤝 Calculating Relational Social Capital...")
        df_sc = self._calculate_relational_capital(df_sc)
        
        # 3. Cognitive Social Capital (Shared Understanding)
        print("   🧠 Calculating Cognitive Social Capital...")
        df_sc = self._calculate_cognitive_capital(df_sc)
        
        # 4. Composite Index
        print("   📈 Creating Composite Index...")
        df_sc = self._create_composite_index(df_sc)
        
        # 5. Farmer Typology
        print("   🎯 Creating Farmer Social Capital Typology...")
        df_sc = self._create_farmer_typology(df_sc)
        
        self._print_social_capital_report(df_sc)
        
        return df_sc
    
    def _calculate_structural_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate structural social capital (networks)"""
        
        # Network membership score
        network_cols = [col for col in df.columns if any(term in str(col).lower() 
                       for term in ['group_member', 'association', 'network'])]
        
        if network_cols:
            for col in network_cols:
                if f"{col}_encoded" in df.columns:
                    df['structural_network_membership'] = df[f"{col}_encoded"]
                    break
        
        # Meeting participation frequency
        meeting_cols = [col for col in df.columns if 'meeting' in str(col).lower()]
        if meeting_cols:
            for col in meeting_cols:
                if f"{col}_encoded" in df.columns:
                    df['structural_meeting_participation'] = df[f"{col}_encoded"]
                    break
        
        # Group activities participation
        activity_cols = [col for col in df.columns if 'activity' in str(col).lower()]
        if activity_cols:
            for col in activity_cols:
                if f"{col}_encoded" in df.columns:
                    df['structural_group_activities'] = df[f"{col}_encoded"]
                    break
        
        # Composite structural score
        structural_cols = [col for col in df.columns if 'structural_' in col]
        if structural_cols:
            df['structural_capital_score'] = df[structural_cols].mean(axis=1)
        
        return df
    
    def _calculate_relational_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relational social capital (trust)"""
        
        # Trust level
        trust_cols = [col for col in df.columns if 'trust' in str(col).lower()]
        if trust_cols:
            for col in trust_cols:
                if f"{col}_encoded" in df.columns:
                    df['relational_trust'] = df[f"{col}_encoded"]
                    break
        
        # Information sharing willingness
        share_cols = [col for col in df.columns if any(term in str(col).lower() 
                       for term in ['share', 'willing'])]
        if share_cols:
            for col in share_cols:
                if f"{col}_encoded" in df.columns:
                    df['relational_sharing_willingness'] = df[f"{col}_encoded"]
                    break
        
        # Support rating
        support_cols = [col for col in df.columns if 'support' in str(col).lower()]
        if support_cols:
            for col in support_cols:
                if f"{col}_encoded" in df.columns:
                    df['relational_support'] = df[f"{col}_encoded"]
                    break
        
        # Composite relational score
        relational_cols = [col for col in df.columns if 'relational_' in col]
        if relational_cols:
            df['relational_capital_score'] = df[relational_cols].mean(axis=1)
        
        return df
    
    def _calculate_cognitive_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cognitive social capital (shared understanding)"""
        
        # Information exchange frequency
        exchange_cols = [col for col in df.columns if 'exchange' in str(col).lower()]
        if exchange_cols:
            for col in exchange_cols:
                if f"{col}_encoded" in df.columns:
                    df['cognitive_info_exchange'] = df[f"{col}_encoded"]
                    break
        
        # Information impact perception
        impact_cols = [col for col in df.columns if 'impact' in str(col).lower()]
        if impact_cols:
            for col in impact_cols:
                if f"{col}_encoded" in df.columns:
                    df['cognitive_info_impact'] = df[f"{col}_encoded"]
                    break
        
        # Collective action effectiveness
        collective_cols = [col for col in df.columns if 'collective' in str(col).lower()]
        if collective_cols:
            for col in collective_cols:
                if f"{col}_encoded" in df.columns:
                    df['cognitive_collective_action'] = df[f"{col}_encoded"]
                    break
        
        # Composite cognitive score
        cognitive_cols = [col for col in df.columns if 'cognitive_' in col]
        if cognitive_cols:
            df['cognitive_capital_score'] = df[cognitive_cols].mean(axis=1)
        
        return df
    
    def _create_composite_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive social capital index"""
        
        # Check which components we have
        component_scores = []
        
        if 'structural_capital_score' in df.columns:
            component_scores.append('structural_capital_score')
        
        if 'relational_capital_score' in df.columns:
            component_scores.append('relational_capital_score')
        
        if 'cognitive_capital_score' in df.columns:
            component_scores.append('cognitive_capital_score')
        
        if component_scores:
            # Calculate weighted average (equal weights for now)
            df['social_capital_composite'] = df[component_scores].mean(axis=1)
            
            # Create quartiles
            df['social_capital_quartile'] = pd.qcut(
                df['social_capital_composite'], 
                4, 
                labels=['Very Low', 'Low', 'High', 'Very High']
            )
            
            # Create binary classification
            median_sc = df['social_capital_composite'].median()
            df['high_social_capital'] = (df['social_capital_composite'] > median_sc).astype(int)
            
            print(f"      Created composite index from {len(component_scores)} components")
        
        return df
    
    def _create_farmer_typology(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create farmer typology based on social capital patterns"""
        
        if 'social_capital_composite' in df.columns:
            # Create typology based on social capital and other factors
            conditions = []
            choices = []
            
            # Type 1: Network Leaders (High social capital, high group participation)
            if 'structural_capital_score' in df.columns:
                conditions.append(
                    (df['social_capital_composite'] >= df['social_capital_composite'].quantile(0.75)) &
                    (df['structural_capital_score'] >= df['structural_capital_score'].quantile(0.75))
                )
                choices.append('Network Leader')
            
            # Type 2: Trusted Advisors (High relational capital)
            if 'relational_capital_score' in df.columns:
                conditions.append(
                    (df['relational_capital_score'] >= df['relational_capital_score'].quantile(0.75)) &
                    (df['social_capital_composite'] >= df['social_capital_composite'].median())
                )
                choices.append('Trusted Advisor')
            
            # Type 3: Information Brokers (High cognitive capital)
            if 'cognitive_capital_score' in df.columns:
                conditions.append(
                    (df['cognitive_capital_score'] >= df['cognitive_capital_score'].quantile(0.75)) &
                    (df['social_capital_composite'] >= df['social_capital_composite'].median())
                )
                choices.append('Information Broker')
            
            # Type 4: Isolated Farmers (Low social capital)
            conditions.append(
                df['social_capital_composite'] <= df['social_capital_composite'].quantile(0.25)
            )
            choices.append('Isolated Farmer')
            
            # Type 5: Average Participants (everything else)
            conditions.append(True)
            choices.append('Average Participant')
            
            if conditions and choices:
                df['farmer_social_type'] = np.select(conditions, choices, default='Unknown')
                print(f"      Created {df['farmer_social_type'].nunique()} farmer social types")
        
        return df
    
    def _print_social_capital_report(self, df: pd.DataFrame):
        """Print social capital analysis report"""
        
        print("\n📊 SOCIAL CAPITAL ANALYSIS REPORT:")
        
        if 'social_capital_composite' in df.columns:
            print(f"   Composite Score Range: {df['social_capital_composite'].min():.2f} - {df['social_capital_composite'].max():.2f}")
            print(f"   Average Score: {df['social_capital_composite'].mean():.2f}")
            print(f"   Standard Deviation: {df['social_capital_composite'].std():.2f}")
        
        if 'social_capital_quartile' in df.columns:
            print(f"\n   Social Capital Distribution:")
            for quartile in ['Very Low', 'Low', 'High', 'Very High']:
                count = (df['social_capital_quartile'] == quartile).sum()
                pct = (count / len(df)) * 100
                print(f"     {quartile}: {count} farmers ({pct:.1f}%)")
        
        if 'farmer_social_type' in df.columns:
            print(f"\n   Farmer Social Typology:")
            for social_type in df['farmer_social_type'].unique():
                count = (df['farmer_social_type'] == social_type).sum()
                pct = (count / len(df)) * 100
                print(f"     {social_type}: {count} farmers ({pct:.1f}%)")
        
        # Correlation with yield if available
        if 'high_yield' in df.columns and 'social_capital_composite' in df.columns:
            high_yield_sc = df[df['high_yield'] == 1]['social_capital_composite'].mean()
            low_yield_sc = df[df['high_yield'] == 0]['social_capital_composite'].mean()
            
            print(f"\n   💡 KEY INSIGHT: Social Capital vs Yield")
            print(f"     High yield farmers: {high_yield_sc:.2f} average social capital")
            print(f"     Low yield farmers: {low_yield_sc:.2f} average social capital")
            print(f"     Difference: {high_yield_sc - low_yield_sc:.2f} points")
    
    def visualize_social_network(self, df: pd.DataFrame, sample_size: int = 50):
        """
        Visualize social network (simulated for survey data)
        
        Args:
            df: DataFrame with social capital data
            sample_size: Number of farmers to visualize
        """
        print("\n🕸️  Creating Social Network Visualization...")
        
        # Create simulated network based on social capital scores
        G = nx.Graph()
        
        # Sample farmers
        sample_df = df.sample(min(sample_size, len(df)))
        
        # Add nodes (farmers)
        for idx, row in sample_df.iterrows():
            G.add_node(idx, 
                      social_capital=row.get('social_capital_composite', 0.5),
                      farmer_type=row.get('farmer_social_type', 'Unknown'))
        
        # Add edges based on similarity in social capital
        nodes = list(G.nodes())
        
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                
                # Calculate similarity
                sc_i = G.nodes[node_i]['social_capital']
                sc_j = G.nodes[node_j]['social_capital']
                similarity = 1 - abs(sc_i - sc_j)
                
                # Add edge if similarity above threshold
                if similarity > 0.7:
                    G.add_edge(node_i, node_j, weight=similarity)
        
        # Visualize
        plt.figure(figsize=(12, 10))
        
        # Node colors by social capital
        node_colors = [G.nodes[node]['social_capital'] for node in G.nodes()]
        
        # Node sizes by degree
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, 
                                      node_color=node_colors,
                                      node_size=node_sizes,
                                      cmap=plt.cm.YlOrRd,
                                      alpha=0.8)
        
        # Draw edges
        edges = nx.draw_networkx_edges(G, pos, 
                                      width=1,
                                      alpha=0.2,
                                      edge_color='gray')
        
        # Draw labels for high social capital farmers
        labels = {}
        for node in G.nodes():
            if G.nodes[node]['social_capital'] > 0.7:
                labels[node] = f"SC: {G.nodes[node]['social_capital']:.2f}"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Farmer Social Network (Simulated from Survey Data)', 
                 fontsize=14, fontweight='bold')
        plt.colorbar(nodes, label='Social Capital Score')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Network created with {G.number_of_nodes()} farmers and {G.number_of_edges()} connections")