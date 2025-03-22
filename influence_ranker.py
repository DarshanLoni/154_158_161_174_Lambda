import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

class InfluenceRanker:
    def __init__(self, processed_df):
        """
        Initialize the InfluenceRanker with processed data
        
        Parameters:
        processed_df (pd.DataFrame): Preprocessed dataframe from the preprocessing step
        """
        self.data = processed_df
        self.weights = {
            'credibility': 0.3,  # Reduced from 0.4
            'longevity': 0.2,     # Reduced from 0.3
            'engagement': 0.5     # Increased from 0.3
        }
        print(f"InfluenceRanker initialized with {len(self.data)} influencers")
    
    def detect_fake_fame(self):
        """
        Detect potentially fake or artificially inflated influence
        Uses anomaly detection to identify suspicious patterns
        """
        try:
            # Features to use for anomaly detection - use only available features
            potential_features = ['followers', 'avg_likes', '60_day_eng_rate', 'posts']
            features = [f for f in potential_features if f in self.data.columns]
            
            if len(features) < 2:
                print("Warning: Not enough features for anomaly detection")
                return 0
            
            # Normalize features for anomaly detection
            X = self.data[features].copy()
            for col in X.columns:
                if X[col].std() > 0:  # Avoid division by zero
                    X[col] = (X[col] - X[col].mean()) / X[col].std()
                else:
                    X[col] = 0
            
            # Apply Isolation Forest for anomaly detection
            clf = IsolationForest(contamination=0.1, random_state=42)
            self.data['anomaly_score'] = clf.fit_predict(X)
            
            # Flag accounts with suspicious patterns
            # -1 indicates an anomaly
            self.data['fake_fame_flag'] = (self.data['anomaly_score'] == -1).astype(int)
            
            # Additional rule-based flags if we have the necessary columns
            if all(col in self.data.columns for col in ['60_day_eng_rate', 'followers']):
                # Unusually high engagement rate compared to follower count
                engagement_to_follower_ratio = self.data['60_day_eng_rate'] / np.log10(self.data['followers'] + 10)
                threshold = engagement_to_follower_ratio.quantile(0.95)
                self.data.loc[engagement_to_follower_ratio > threshold, 'fake_fame_flag'] = 1
            
            # Z-score for follower counts
            if 'followers' in self.data.columns:
                self.data['follower_z_score'] = np.abs((self.data['followers'] - self.data['followers'].mean()) / 
                                                      (self.data['followers'].std() + 1e-10))  # Avoid division by zero
                self.data.loc[self.data['follower_z_score'] > 2.5, 'fake_fame_flag'] = 1
            
            num_flagged = self.data['fake_fame_flag'].sum()
            print(f"Detected {num_flagged} accounts with potential fake influence")
            return num_flagged
            
        except Exception as e:
            print(f"Error during fake fame detection: {e}")
            self.data['fake_fame_flag'] = 0
            return 0
    
    def calculate_final_scores(self, adjust_for_fake=True):
        """
        Calculate final InfluenceIQ scores based on weighted components
        
        Parameters:
        adjust_for_fake (bool): Whether to penalize scores for suspected fake fame
        
        Returns:
        pd.DataFrame: Dataframe with final scores
        """
        try:
            # Calculate the weighted score
            self.data['influenceiq_score'] = (
                self.weights['credibility'] * self.data['credibility_score'] +
                self.weights['longevity'] * self.data['longevity_score'] +
                self.weights['engagement'] * self.data['engagement_score']
            )
            
            # Adjust scores for accounts flagged as potentially fake
            if adjust_for_fake and 'fake_fame_flag' in self.data.columns:
                penalty_factor = 0.3  # 30% penalty for suspicious accounts
                self.data.loc[self.data['fake_fame_flag'] == 1, 'influenceiq_score'] *= (1 - penalty_factor)
            
            # Scale to 0-100 for readability
            self.data['influenceiq_score'] = (self.data['influenceiq_score'] * 100).round(1)
            
            # Update rankings based on new scores
            self.data = self.data.sort_values('influenceiq_score', ascending=False).reset_index(drop=True)
            self.data['influenceiq_rank'] = self.data.index + 1
            
            print("Final scores calculated successfully")
            return self.data
            
        except Exception as e:
            print(f"Error during score calculation: {e}")
            raise
    
    def get_top_influencers(self, n=10, category=None, country=None):
        """
        Get top influencers, optionally filtered by category or country
        
        Parameters:
        n (int): Number of influencers to return
        category (str): Filter by category if provided
        country (str): Filter by country if provided
        
        Returns:
        pd.DataFrame: Top n influencers
        """
        try:
            filtered_data = self.data.copy()
            
            if category is not None and 'category' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['category'] == category]
            
            if country is not None and 'country' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['country'] == country]
            
            return filtered_data.head(n)
            
        except Exception as e:
            print(f"Error getting top influencers: {e}")
            return self.data.head(n)
    
    def generate_insights(self):
        """
        Generate insights about the influencer landscape
        
        Returns:
        dict: Dictionary of insights
        """
        try:
            insights = {}
            
            if 'country' in self.data.columns:
                insights['top_countries'] = self.data['country'].value_counts().head(5).to_dict()
            
            if '60_day_eng_rate' in self.data.columns:
                insights['avg_engagement_rate'] = self.data['60_day_eng_rate'].mean()
            
            if 'influenceiq_score' in self.data.columns:
                insights['median_influenceiq'] = self.data['influenceiq_score'].median()
            
            if 'fake_fame_flag' in self.data.columns:
                insights['fake_fame_percentage'] = (self.data['fake_fame_flag'].sum() / len(self.data)) * 100
            
            if 'credibility_score' in self.data.columns and 'channel_info' in self.data.columns:
                insights['credibility_leaders'] = self.data.sort_values(
                    'credibility_score', ascending=False)['channel_info'].head(3).tolist()
            
            if 'longevity_score' in self.data.columns and 'channel_info' in self.data.columns:
                insights['longevity_leaders'] = self.data.sort_values(
                    'longevity_score', ascending=False)['channel_info'].head(3).tolist()
            
            if 'engagement_score' in self.data.columns and 'channel_info' in self.data.columns:
                insights['engagement_leaders'] = self.data.sort_values(
                    'engagement_score', ascending=False)['channel_info'].head(3).tolist()
            
            print("Insights generated successfully")
            return insights
            
        except Exception as e:
            print(f"Error generating insights: {e}")
            return {"error": str(e)}
    
    def visualize_scores(self, save_path="influenceiq_analysis.png"):
        """
        Create visualizations of the InfluenceIQ scores and components
        """
        try:
            plt.figure(figsize=(12, 10))
            
            # Plot 1: InfluenceIQ Score Distribution
            plt.subplot(2, 2, 1)
            if 'influenceiq_score' in self.data.columns:
                sns.histplot(self.data['influenceiq_score'], kde=True)
                plt.title('Distribution of InfluenceIQ Scores')
                plt.xlabel('Score')
            
            # Plot 2: Component Comparison for Top 10
            plt.subplot(2, 2, 2)
            if all(col in self.data.columns for col in ['channel_info', 'credibility_score', 
                                                       'longevity_score', 'engagement_score']):
                top10 = self.data.head(10)
                
                # Reshape for seaborn
                plot_data = pd.melt(
                    top10[['channel_info', 'credibility_score', 'longevity_score', 'engagement_score']],
                    id_vars=['channel_info'],
                    var_name='Component',
                    value_name='Score'
                )
                
                sns.barplot(x='channel_info', y='Score', hue='Component', data=plot_data)
                plt.title('Score Components for Top 10 Influencers')
                plt.xticks(rotation=45, ha='right')
            
            # Plot 3: Followers vs InfluenceIQ Score (with fake detection)
            plt.subplot(2, 2, 3)
            if all(col in self.data.columns for col in ['followers', 'influenceiq_score', 'fake_fame_flag']):
                sns.scatterplot(
                    x='followers', 
                    y='influenceiq_score', 
                    hue='fake_fame_flag',
                    data=self.data,
                    palette={0: 'blue', 1: 'red'}
                )
                plt.title('Followers vs InfluenceIQ Score')
                plt.xlabel('Followers (log scale)')
                plt.xscale('log')
                plt.legend(title='Potential Fake Fame')
            
            # Plot 4: Country Distribution
            plt.subplot(2, 2, 4)
            if 'country' in self.data.columns:
                top_countries = self.data['country'].value_counts().head(5)
                sns.barplot(x=top_countries.index, y=top_countries.values)
                plt.title('Top 5 Countries by Influencer Count')
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"Visualizations saved to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return None