# sentiment_app.py
# INSTRUCTIONS:
# 1. Save this entire file as sentiment_app.py
# 2. Run: pip install streamlit pandas scikit-learn nltk matplotlib
# 3. Run: python -c "import nltk; nltk.download('vader_lexicon')"
# 4. Run: streamlit run sentiment_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from sklearn.feature_extraction.text import CountVectorizer
import time
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="SentimentTracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more beautiful
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4E79A7;
        --secondary-color: #F28E2B;
        --accent-color: #59A14F;
        --background-color: #f8f9fa;
        --text-color: #212529;
    }
    
    /* Page background */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Header styling */
    h1 {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    h2, h3, h4, h5 {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Card styling */
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: scale(1.05);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 5px;
        border: 1px solid #ced4da;
        padding: 0.5rem;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        border-radius: 5px 5px 0 0;
        gap: 0.5rem;
        font-weight: 600;
    }
    
    /* Positive/Negative/Neutral colors */
    .positive {
        background-color: rgba(40, 167, 69, 0.2);
        border-left: 5px solid #28a745;
    }
    
    .negative {
        background-color: rgba(220, 53, 69, 0.2);
        border-left: 5px solid #dc3545;
    }
    
    .neutral {
        background-color: rgba(108, 117, 125, 0.2);
        border-left: 5px solid #6c757d;
    }
    
    /* Animation for results */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    #resultSection {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f1f3f9;
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        padding-left: 1rem;
    }
    
    /* Footer styling */
    footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}
    
    return sia.polarity_scores(cleaned_text)

# Function to classify sentiment based on compound score
def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to get color based on sentiment
def get_sentiment_color(sentiment):
    if sentiment == "Positive":
        return "rgba(40, 167, 69, 0.2)", "#28a745"
    elif sentiment == "Negative":
        return "rgba(220, 53, 69, 0.2)", "#dc3545"
    else:
        return "rgba(108, 117, 125, 0.2)", "#6c757d"

# Function to generate sample data
def generate_sample_data(num_days=30):
    sources = ['Twitter', 'News Articles', 'Customer Reviews', 'Support Tickets']
    companies = ['TechCorp', 'EcoSolutions', 'HealthPlus', 'FinanceHub']
    
    data = []
    
    for i in range(num_days):
        date = datetime.now() - timedelta(days=i)
        for source in sources:
            for company in companies:
                # Generate random number of entries
                num_entries = np.random.randint(5, 20)
                
                for _ in range(num_entries):
                    # Generate random sentiment with bias
                    if company == 'TechCorp':
                        # TechCorp has mostly positive reviews
                        compound = np.random.normal(0.3, 0.4)
                    elif company == 'FinanceHub':
                        # FinanceHub has mixed reviews
                        compound = np.random.normal(0, 0.5)
                    elif company == 'HealthPlus' and source == 'Support Tickets':
                        # HealthPlus has negative support tickets
                        compound = np.random.normal(-0.3, 0.3)
                    else:
                        # Others are more random
                        compound = np.random.normal(0.1, 0.6)
                    
                    # Clamp compound score between -1 and 1
                    compound = max(min(compound, 1.0), -1.0)
                    
                    # Classify sentiment
                    sentiment = classify_sentiment(compound)
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'source': source,
                        'company': company,
                        'compound_score': compound,
                        'sentiment': sentiment
                    })
    
    return pd.DataFrame(data)

# Generate and cache sample data
@st.cache_data
def get_sample_data():
    return generate_sample_data()

# App title and description with improved styling
st.markdown("""
# 📊 SentimentTracker
<div class="card">
    <p style="font-size: 1.2rem; line-height: 1.6;">
        Welcome to <span style="font-weight: 700; color: #4E79A7;">SentimentTracker</span> – an advanced analytics platform that helps you understand customer sentiment across multiple channels. Monitor brand perception, track satisfaction trends, and gain actionable insights from text data.
    </p>
    <p style="color: #6c757d; font-style: italic; margin-top: 0.5rem;">
        Powered by machine learning and natural language processing.
    </p>
</div>
""", unsafe_allow_html=True)

# Create tabs with custom styling
tab1, tab2, tab3 = st.tabs(["✨ Sentiment Analyzer", "📊 Dashboard", "ℹ️ About"])

with tab1:
    st.markdown("""
    <h2 style="margin-bottom: 1.5rem;">✨ Analyze Text Sentiment</h2>
    <div class="card">
        <p style="margin-bottom: 1rem;">
            Enter any text below to analyze its sentiment. Our AI will evaluate the emotional tone and provide detailed insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Text input area with improved styling
    text_input = st.text_area("Enter text to analyze:", height=150, 
                             placeholder="Paste a product review, social media post, email, or any text you want to analyze...")
    
    # Analysis button with custom styling
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        analyze_button = st.button("✨ Analyze Sentiment", use_container_width=True)
    
    if analyze_button:
        if text_input:
            with st.spinner("⏳ Analyzing sentiment..."):
                # Add slight delay to simulate processing
                time.sleep(0.5)
                
                # Get sentiment scores
                sentiment_scores = analyze_sentiment(text_input)
                
                # Classify sentiment
                sentiment = classify_sentiment(sentiment_scores['compound'])
                
                # Get appropriate colors
                bg_color, border_color = get_sentiment_color(sentiment)
                
                # Add emoji based on sentiment
                sentiment_emoji = "😃" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
                
                # Display result with improved styling
                st.markdown(f"""
                <div style="padding: 25px; border-radius: 15px; background-color: {bg_color}; 
                border-left: 8px solid {border_color}; margin: 25px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="color: {border_color}; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        {sentiment_emoji} Sentiment: {sentiment}
                    </h2>
                    <p style="font-size: 1.2rem; font-weight: 500;">Confidence: {abs(sentiment_scores['compound'])*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed scores with improved metrics
                st.markdown("<h3 style='margin: 1.5rem 0 1rem 0;'>Sentiment Breakdown</h3>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                col1.markdown(f"""
                <div style="background-color: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #28a745; margin-bottom: 10px;">Positive</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: #28a745;">{sentiment_scores['pos']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                col2.markdown(f"""
                <div style="background-color: rgba(108, 117, 125, 0.1); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #6c757d; margin-bottom: 10px;">Neutral</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: #6c757d;">{sentiment_scores['neu']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                col3.markdown(f"""
                <div style="background-color: rgba(220, 53, 69, 0.1); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #dc3545; margin-bottom: 10px;">Negative</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: #dc3545;">{sentiment_scores['neg']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Word analysis
                if len(text_input.split()) > 3:
                    st.subheader("Word Impact Analysis")
                    
                    # Tokenize and analyze individual words
                    words = re.findall(r'\b\w+\b', clean_text(text_input))
                    word_impacts = []
                    
                    for word in set(words):
                        if len(word) > 2:  # Skip very short words
                            # Calculate impact by removing the word
                            text_without_word = re.sub(r'\b' + word + r'\b', '', text_input)
                            score_without_word = analyze_sentiment(text_without_word)['compound']
                            impact = sentiment_scores['compound'] - score_without_word
                            word_impacts.append((word, impact))
                    
                    # Sort by absolute impact
                    word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Display top influential words
                    if word_impacts:
                        impact_df = pd.DataFrame(word_impacts[:10], columns=['Word', 'Sentiment Impact'])
                        impact_df['Impact Type'] = impact_df['Sentiment Impact'].apply(
                            lambda x: "Positive" if x > 0 else "Negative")
                        
                        fig, ax = plt.figure(figsize=(10, 5)), plt.subplot()
                        
                        # Create horizontal bar chart
                        bars = ax.barh(impact_df['Word'], impact_df['Sentiment Impact'])
                        
                        # Color bars based on impact
                        for i, bar in enumerate(bars):
                            if impact_df.iloc[i]['Sentiment Impact'] > 0:
                                bar.set_color('#28a745')
                            else:
                                bar.set_color('#dc3545')
                        
                        ax.set_xlabel('Sentiment Impact')
                        ax.set_title('Words with Most Impact on Sentiment')
                        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                        
                        st.pyplot(fig)
                
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Sentiment Dashboard")
    
    # Load sample data
    df = get_sample_data()
    
    # Sidebar filters for dashboard with improved styling
    st.sidebar.markdown("""
    <h2 style="color: #4E79A7; margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid #4E79A7;">
        Dashboard Filters
    </h2>
    <p style="margin-bottom: 1.5rem; color: #6c757d;">
        Customize your view to focus on specific data points and trends.
    </p>
    """, unsafe_allow_html=True)
    
    # Date range filter
    unique_dates = sorted(df['date'].unique(), reverse=True)
    start_date, end_date = st.sidebar.select_slider(
        "Select Date Range",
        options=unique_dates,
        value=(unique_dates[-min(len(unique_dates), 10)], unique_dates[0])
    )
    
    # Company filter
    companies = sorted(df['company'].unique())
    selected_companies = st.sidebar.multiselect(
        "Select Companies",
        options=companies,
        default=companies
    )
    
    # Source filter
    sources = sorted(df['source'].unique())
    selected_sources = st.sidebar.multiselect(
        "Select Sources",
        options=sources,
        default=sources
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date) & 
        (df['company'].isin(selected_companies)) &
        (df['source'].isin(selected_sources))
    ]
    
    # Overview metrics
    st.subheader("Sentiment Overview")
    
    # Calculate metrics
    total_entries = len(filtered_df)
    
    # Check if there are any entries to avoid division by zero
    if total_entries > 0:
        positive_pct = len(filtered_df[filtered_df['sentiment'] == 'Positive']) / total_entries * 100
        negative_pct = len(filtered_df[filtered_df['sentiment'] == 'Negative']) / total_entries * 100
        neutral_pct = len(filtered_df[filtered_df['sentiment'] == 'Neutral']) / total_entries * 100
    else:
        # Default values when no data is available
        positive_pct = 0
        negative_pct = 0
        neutral_pct = 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", f"{total_entries:,}")
    col2.metric("Positive", f"{positive_pct:.1f}%")
    col3.metric("Neutral", f"{neutral_pct:.1f}%")
    col4.metric("Negative", f"{negative_pct:.1f}%")
    
    # Sentiment trend over time
    st.subheader("Sentiment Trends Over Time")
    
    # Check if there's data to plot
    if not filtered_df.empty:
        # Group by date and calculate sentiment percentages
        trend_df = filtered_df.groupby('date')['sentiment'].value_counts().unstack().fillna(0)
        if 'Positive' not in trend_df.columns:
            trend_df['Positive'] = 0
        if 'Negative' not in trend_df.columns:
            trend_df['Negative'] = 0
        if 'Neutral' not in trend_df.columns:
            trend_df['Neutral'] = 0
        
        # Calculate percentages
        trend_total = trend_df.sum(axis=1)
        
        # Check if there are any non-zero totals
        if (trend_total > 0).any():
            trend_pct = trend_df.div(trend_total, axis=0) * 100
            
            # Plot trend
            fig, ax = plt.figure(figsize=(10, 6)), plt.subplot()
            trend_pct.plot(kind='line', ax=ax, marker='o', linewidth=2.5)
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
            ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
            ax.legend(title='Sentiment', title_fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Customize colors
            for i, line in enumerate(ax.get_lines()):
                if i == 0 and line.get_label() == 'Negative':
                    line.set_color('#dc3545')
                elif i == 1 and line.get_label() == 'Neutral':
                    line.set_color('#6c757d')
                elif i == 2 and line.get_label() == 'Positive':
                    line.set_color('#28a745')
            
            # Add data points with hover information
            for i, line in enumerate(ax.get_lines()):
                ax.scatter(line.get_xdata(), line.get_ydata(), color=line.get_color(), s=50, zorder=10)
            
            # Enhance the background and styling
            ax.set_facecolor('#f8f9fa')
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            st.pyplot(fig)
        else:
            st.info("No trend data available with current filters.")
    else:
        st.info("No data available for trend visualization. Try adjusting your filters.")
    
    # Company comparison
    if len(selected_companies) > 1:
        st.subheader("Company Comparison")
        
        # Check if there's data to plot
        if not filtered_df.empty:
            # Group by company and calculate average compound score
            company_df = filtered_df.groupby('company')['compound_score'].mean().sort_values()
            
            if not company_df.empty:
                # Plot company comparison
                fig, ax = plt.figure(figsize=(10, 5)), plt.subplot()
                bars = ax.barh(company_df.index, company_df.values)
                
                # Color bars based on sentiment and add gradient effect
                for i, bar in enumerate(bars):
                    if company_df.iloc[i] > 0.05:
                        bar.set_color('#28a745')
                    elif company_df.iloc[i] < -0.05:
                        bar.set_color('#dc3545')
                    else:
                        bar.set_color('#6c757d')
                
                # Add data labels
                for i, v in enumerate(company_df.values):
                    ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')
                
                ax.set_xlabel('Average Sentiment Score', fontsize=12, fontweight='bold')
                ax.set_title('Company Sentiment Comparison', fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.6)
                
                # Remove spines and set background
                ax.set_facecolor('#f8f9fa')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Add grid for better readability
                ax.xaxis.grid(True, linestyle='--', alpha=0.6)
                
                # Add custom styling
                plt.tight_layout()
                
                st.pyplot(fig)
            else:
                st.info("No company comparison data available with current filters.")
        else:
            st.info("No data available for company comparison. Try adjusting your filters.")
    
    # Source analysis
    st.subheader("Sentiment by Source")
    
    # Check if there's data to plot
    if not filtered_df.empty and len(filtered_df['source'].unique()) > 0:
        # Group by source and calculate sentiment counts
        source_df = filtered_df.groupby(['source', 'sentiment']).size().unstack().fillna(0)
        
        if not source_df.empty:
            # Calculate percentages
            source_total = source_df.sum(axis=1)
            
            # Check if there are any non-zero totals
            if (source_total > 0).any():
                source_pct = source_df.div(source_total, axis=0) * 100
                
                # Plot source analysis with enhanced styling
                fig, ax = plt.figure(figsize=(12, 6)), plt.subplot()
                
                # Use specific colors for sentiment categories
                colors = ['#dc3545', '#6c757d', '#28a745'] if all(s in source_pct.columns for s in ['Negative', 'Neutral', 'Positive']) else None
                
                # Create stacked bar chart with enhanced styling
                source_pct.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)
                
                # Add percentage labels on bars
                for i, source in enumerate(source_pct.index):
                    cumulative = 0
                    for col in source_pct.columns:
                        value = source_pct.loc[source, col]
                        if value > 10:  # Only show labels if there's enough space
                            ax.text(i, cumulative + value/2, f"{value:.0f}%", 
                                   ha='center', va='center', fontweight='bold', color='white')
                        cumulative += value
                
                ax.set_xlabel('Source', fontsize=12, fontweight='bold')
                ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
                ax.set_title('Sentiment Distribution by Source', fontsize=14, fontweight='bold')
                ax.legend(title='Sentiment', title_fontsize=12)
                
                # Set background and remove spines for modern look
                ax.set_facecolor('#f8f9fa')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Add a grid for better readability
                ax.yaxis.grid(True, linestyle='--', alpha=0.3)
                
                # Customize tick labels
                plt.xticks(rotation=0, fontweight='bold')
                plt.yticks(fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No source analysis data available with current filters.")
        else:
            st.info("No source data available for visualization. Try adjusting your filters.")
    else:
        st.info("No data available for source analysis. Try adjusting your filters.")
    
    # Raw data (expandable)
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)

with tab3:
    st.header("About SentimentTracker")
    
    st.markdown("""
    ## Project Description
    
    SentimentTracker is a powerful sentiment analysis tool designed to help businesses monitor and understand public sentiment across multiple platforms. This tool leverages natural language processing techniques to analyze text data and provide valuable insights into customer opinions and market trends.
    
    ## Features
    
    - **Real-time sentiment analysis** of any text input
    - **Multi-platform monitoring** across social media, news, reviews, and support tickets
    - **Interactive dashboard** with filtering options and data visualization
    - **Comparative analysis** between companies and data sources
    - **Word impact analysis** to identify key terms affecting sentiment
    - **Historical trend tracking** to monitor sentiment changes over time
    
    ## Technical Implementation
    
    - **Backend:** Python with NLTK's VADER sentiment analyzer
    - **Frontend:** Streamlit for interactive web interface
    - **Data Processing:** Pandas and NumPy for data manipulation
    - **Visualization:** Matplotlib for charts and graphs
    
    ## Future Enhancements
    
    - Integration with live data sources (Twitter API, RSS feeds, etc.)
    - Advanced NLP models for improved accuracy
    - Topic modeling to identify common themes
    - Emotion detection beyond simple sentiment
    - Anomaly detection for sudden sentiment shifts
    
    ## Created By
    
    [Your Name] - Data Scientist & ML Engineer
    
    GitHub: [Your GitHub Username]
    """)

# Add styled footer
st.markdown("""
<footer>
    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div style="background-color: #f8f9fa; padding: 0.5rem 1rem; border-radius: 5px;">
            <span style="font-weight: bold; color: #4E79A7;">ML Engineer</span>
        </div>
        <div style="background-color: #f8f9fa; padding: 0.5rem 1rem; border-radius: 5px;">
            <span style="font-weight: bold; color: #4E79A7;">Data Scientist</span>
        </div>
        <div style="background-color: #f8f9fa; padding: 0.5rem 1rem; border-radius: 5px;">
            <span style="font-weight: bold; color: #4E79A7;">NLP Specialist</span>
        </div>
    </div>
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Developed with ❤️ by <strong>[Your Name]</strong></p>
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 0.5rem;">
        <a href="https://github.com/yourusername" target="_blank" style="text-decoration: none; color: #4E79A7;">
            <span style="font-weight: bold;">GitHub</span>
        </a>
        <a href="https://linkedin.com/in/yourprofile" target="_blank" style="text-decoration: none; color: #4E79A7;">
            <span style="font-weight: bold;">LinkedIn</span>
        </a>
        <a href="mailto:your.email@example.com" target="_blank" style="text-decoration: none; color: #4E79A7;">
            <span style="font-weight: bold;">Email</span>
        </a>
    </div>
    <p style="color: #6c757d; font-size: 0.9rem; margin-top: 1rem;">© 2025 SentimentTracker</p>
</footer>
""", unsafe_allow_html=True)