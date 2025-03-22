import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_data  # Import your preprocessing function
from influence_ranker import InfluenceRanker  # Import your InfluenceRanker class

# Set page configuration
st.set_page_config(
    page_title="InfluenceIQ - AI-Powered Influencer Ranking System",
    page_icon="🚀",
    layout="wide"
)

# App title and description
st.title("🚀 InfluenceIQ")
st.markdown("### The AI-Powered System That Ranks Who Really Matters!")

st.markdown("""
This platform ranks public figures based on:
- ⭐ **Credibility & Trustworthiness** – How credible are they in their field?
- ⏳ **Fame Longevity** – How long have they remained relevant?
- 📈 **Meaningful Engagement** – Are they influencing for the better or just trending for the moment?
""")

# Sidebar
st.sidebar.title("⚙️ Controls")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your influencer dataset (CSV)", type="csv")

# Sample data option
use_sample_data = st.sidebar.checkbox("Use sample data instead", value=True)

# Filter options
st.sidebar.subheader("Filter Options")
filter_country = st.sidebar.checkbox("Filter by Country")
filter_category = st.sidebar.checkbox("Filter by Category")

# Process data
# Process data section in app.py
# Process data section in app.py
if uploaded_file is not None:
    # If a file is uploaded, uncheck the sample data option
    use_sample_data = False
    
    # Process the uploaded file
    df = pd.read_csv(uploaded_file)
    processed_df = preprocess_data(df)
    ranker = InfluenceRanker(processed_df)
    ranker.detect_fake_fame()
    ranked_data = ranker.calculate_final_scores()
    
    st.success("Dataset processed successfully!")
    
elif use_sample_data:
    # Use the sample data - here we're using a path, which also works
    try:
        sample_data_path = "dataset.csv"  # Path to sample data
        processed_df = preprocess_data(sample_data_path)
        ranker = InfluenceRanker(processed_df)
        ranker.detect_fake_fame()
        ranked_data = ranker.calculate_final_scores()
        
        st.info("Using sample dataset")
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        st.stop()
else:
    st.warning("Please upload a dataset or use the sample data")
    st.stop()

# Main content area - Ranking display
st.header("📊 Influencer Rankings")

# Apply filters if selected
filtered_data = ranked_data.copy()

if filter_country and 'country' in filtered_data.columns:
    countries = ['All'] + sorted(filtered_data['country'].unique().tolist())
    selected_country = st.selectbox("Select a country:", countries)
    
    if selected_country != 'All':
        filtered_data = filtered_data[filtered_data['country'] == selected_country]

if filter_category and 'category' in filtered_data.columns:
    categories = ['All'] + sorted(filtered_data['category'].unique().tolist())
    selected_category = st.selectbox("Select a category:", categories)
    
    if selected_category != 'All':
        filtered_data = filtered_data[filtered_data['category'] == selected_category]

# Number of influencers to display
num_influencers = st.slider("Number of influencers to display", 5, 100, 20)

# Format the table display
display_df = filtered_data.head(num_influencers).copy()

# Ensure these columns exist
if 'channel_info' in display_df.columns:
    display_df = display_df.rename(columns={'channel_info': 'Influencer'})

if 'influenceiq_score' in display_df.columns:
    display_df['InfluenceIQ Score'] = display_df['influenceiq_score'].round(1)

if 'credibility_score' in display_df.columns:
    display_df['Credibility Score'] = (display_df['credibility_score'] * 100).round(1)

if 'longevity_score' in display_df.columns:
    display_df['Longevity Score'] = (display_df['longevity_score'] * 100).round(1)

if 'engagement_score' in display_df.columns:
    display_df['Engagement Score'] = (display_df['engagement_score'] * 100).round(1)

if 'fake_fame_flag' in display_df.columns:
    display_df['Fake Fame Suspected'] = display_df['fake_fame_flag'].map({0: '✅ No', 1: '⚠️ Yes'})

# Select columns to display
columns_to_display = [col for col in ['Influencer', 'InfluenceIQ Score', 'Credibility Score', 
                                    'Longevity Score', 'Engagement Score', 'Fake Fame Suspected',
                                    'country', 'followers']
                    if col in display_df.columns]

# Display the table
st.dataframe(display_df[columns_to_display], width=1000)

# Visualizations
st.header("📈 Insights & Visualizations")

# Create two columns for visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Score Distribution")
    
    # Create a histogram of the InfluenceIQ scores
    if 'influenceiq_score' in ranked_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(ranked_data['influenceiq_score'], kde=True, ax=ax)
        ax.set_title('Distribution of InfluenceIQ Scores')
        ax.set_xlabel('Score')
        st.pyplot(fig)

with col2:
    st.subheader("Top 10 Comparison")
    
    # Create a comparison of the top 10 influencers
    if all(col in ranked_data.columns for col in ['channel_info', 'credibility_score', 
                                               'longevity_score', 'engagement_score']):
        top10 = ranked_data.head(10)
        
        # Reshape for seaborn
        plot_data = pd.melt(
            top10[['channel_info', 'credibility_score', 'longevity_score', 'engagement_score']],
            id_vars=['channel_info'],
            var_name='Component',
            value_name='Score'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='channel_info', y='Score', hue='Component', data=plot_data, ax=ax)
        ax.set_title('Score Components for Top 10 Influencers')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

# Create a second row of visualizations
col3, col4 = st.columns(2)

with col3:
    st.subheader("Followers vs. InfluenceIQ Score")
    
    # Create a scatter plot of followers vs. InfluenceIQ score
    if all(col in ranked_data.columns for col in ['followers', 'influenceiq_score', 'fake_fame_flag']):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='followers', 
            y='influenceiq_score', 
            hue='fake_fame_flag',
            data=ranked_data,
            palette={0: 'blue', 1: 'red'},
            ax=ax
        )
        ax.set_title('Followers vs. InfluenceIQ Score')
        ax.set_xlabel('Followers (log scale)')
        ax.set_xscale('log')
        ax.legend(title='Potential Fake Fame', labels=['Genuine', 'Suspicious'])
        st.pyplot(fig)

with col4:
    st.subheader("Country Distribution")
    
    # Create a bar chart of the top countries
    if 'country' in ranked_data.columns:
        top_countries = ranked_data['country'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_countries.index, y=top_countries.values, ax=ax)
        ax.set_title('Top 5 Countries by Influencer Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

# System insights
st.header("🔍 System Insights")

insights = ranker.generate_insights()

# Display insights in an expandable section
with st.expander("View Detailed Insights"):
    col_a, col_b = st.columns(2)
    
    with col_a:
        if 'avg_engagement_rate' in insights:
            st.metric("Average Engagement Rate", f"{insights['avg_engagement_rate']:.2%}")
        
        if 'median_influenceiq' in insights:
            st.metric("Median InfluenceIQ Score", f"{insights['median_influenceiq']:.1f}")
        
        if 'fake_fame_percentage' in insights:
            st.metric("Suspicious Accounts", f"{insights['fake_fame_percentage']:.1f}%")
    
    with col_b:
        if 'credibility_leaders' in insights:
            st.write("**Credibility Leaders:**")
            for leader in insights['credibility_leaders']:
                st.write(f"- {leader}")
        
        if 'longevity_leaders' in insights:
            st.write("**Longevity Leaders:**")
            for leader in insights['longevity_leaders']:
                st.write(f"- {leader}")
        
        if 'engagement_leaders' in insights:
            st.write("**Engagement Leaders:**")
            for leader in insights['engagement_leaders']:
                st.write(f"- {leader}")

# Download section
st.header("📥 Download Results")
csv = ranked_data.to_csv(index=False)
st.download_button(
    label="Download full rankings as CSV",
    data=csv,
    file_name="influenceiq_rankings.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.markdown("Built for the InfluenceIQ Challenge | ⭐ The AI-Powered System That Ranks Who Really Matters!")