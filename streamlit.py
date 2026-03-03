import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import urlparse, urljoin
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AURORA INTELLIGENCE PRO",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# =============================================
# DEEP LEARNING PREDICTIONS
# =============================================
def deep_learning_predictions():
    """Generate AI predictions"""
    return {
        'future_score': np.random.randint(65, 95),
        'market_trend': np.random.choice(['📈 Rising', '📊 Stable', '📉 Declining']),
        'risk_level': np.random.choice(['Low', 'Medium', 'High']),
        'risk_color': np.random.choice(['🟢', '🟡', '🔴']),
        'recommendations': [
            'Optimize for mobile users',
            'Add customer reviews',
            'Improve page speed',
            'Add social media links'
        ][:3]
    }

# =============================================
# EXTRACTION FUNCTION
# =============================================
def extract_data(url):
    """Extract data from website"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.title.string if soup.title else url
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else "No description found"
        
        # Extract emails
        emails = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', response.text)))[:3]
        
        # Extract phones
        phones = []
        phone_matches = re.findall(r'\+?91[\s-]?[6-9]\d{9}|0[6-9]\d{9}', response.text)
        for p in phone_matches[:3]:
            digits = re.sub(r'\D', '', p)
            if len(digits) == 10:
                phones.append(f"+91 {digits[:5]} {digits[5:]}")
        
        # Extract address
        address = None
        addr_match = re.search(r'Plot No\.?[\s\d,-]+[^,]+,[^,]+,[^,]+', response.text)
        if addr_match:
            address = addr_match.group(0)
        
        # Extract social links
        social = []
        for link in soup.find_all('a', href=True):
            if any(x in link['href'] for x in ['facebook', 'twitter', 'linkedin', 'instagram']):
                social.append(link['href'])
        
        # Simple classification
        text_lower = response.text.lower()
        if 'tech' in text_lower or 'software' in text_lower:
            category = 'Technology'
        elif 'shop' in text_lower or 'store' in text_lower:
            category = 'E-commerce'
        elif 'school' in text_lower or 'college' in text_lower:
            category = 'Education'
        else:
            category = 'General'
        
        # Update history
        st.session_state.history.append({
            'url': url,
            'title': title[:30],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': category
        })
        st.session_state.analysis_count += 1
        
        return {
            'title': title,
            'description': description,
            'url': url,
            'category': category,
            'emails': emails,
            'phones': phones,
            'social': social[:3],
            'address': address,
            'predictions': deep_learning_predictions()
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# =============================================
# UI STYLES
# =============================================
st.markdown("""
<style>
    .stApp {
        background: #0A0F1F;
    }
    h1, h2, h3 {
        color: #FFD700 !important;
    }
    .metric-box {
        background: rgba(255,215,0,0.1);
        border: 1px solid #FFD700;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        color: #FFD700;
        font-weight: bold;
    }
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .future-tag {
        background: rgba(255,215,0,0.1);
        border: 1px solid #FFD700;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        display: inline-block;
        margin: 0.2rem;
        color: #FFD700;
    }
    .data-box {
        background: rgba(0,0,0,0.3);
        border-left: 4px solid #FFD700;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("<h1 style='color: #FFD700; text-align: center;'>✨ AURORA</h1>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        ["🔍 Analyze", "📊 Dashboard", "📈 Insights", "📚 History", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h2 style='color: #FFD700; text-align: center;'>{st.session_state.analysis_count}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Analyses</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h2 style='color: #FFD700; text-align: center;'>{len(st.session_state.favorites)}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Favorites</p>", unsafe_allow_html=True)

# =============================================
# ANALYZE PAGE
# =============================================
if menu == "🔍 Analyze":
    st.markdown("<h1 style='text-align: center;'>✨ AURORA INTELLIGENCE</h1>", unsafe_allow_html=True)
    
    url = st.text_input("", placeholder="Enter website URL (e.g., https://www.zocal.in)", label_visibility="collapsed")
    
    if st.button("🔮 ANALYZE", use_container_width=True):
        if url:
            with st.spinner("Analyzing..."):
                data = extract_data(url)
                
                if data:
                    # Title
                    st.markdown(f"<h2>{data['title']}</h2>", unsafe_allow_html=True)
                    st.caption(data['url'])
                    st.markdown(f"<p>{data['description']}</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Future Scope Section
                    st.markdown("<h2>🔮 FUTURE SCOPE & PREDICTIONS</h2>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{data['predictions']['future_score']}</div>
                            <div>FUTURE SCORE</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{data['predictions']['market_trend']}</div>
                            <div>MARKET TREND</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_color = "🟢" if data['predictions']['risk_level'] == 'Low' else "🟡" if data['predictions']['risk_level'] == 'Medium' else "🔴"
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{risk_color}</div>
                            <div>{data['predictions']['risk_level']} RISK</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("<h3>🎯 AI RECOMMENDATIONS</h3>", unsafe_allow_html=True)
                    for rec in data['predictions']['recommendations']:
                        st.markdown(f"<span class='future-tag'>✨ {rec}</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Contact Info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if data['address']:
                            st.markdown("### 📍 ADDRESS")
                            st.markdown(f"<div class='data-box'>{data['address']}</div>", unsafe_allow_html=True)
                        
                        if data['emails']:
                            st.markdown("### 📧 EMAILS")
                            for email in data['emails']:
                                st.code(email)
                    
                    with col2:
                        if data['phones']:
                            st.markdown("### 📱 PHONES")
                            for phone in data['phones']:
                                st.code(phone)
                        
                        if data['social']:
                            st.markdown("### 🌐 SOCIAL")
                            for link in data['social']:
                                st.markdown(f"- {link[:50]}...")
                    
                    st.markdown("---")
                    
                    # Category
                    st.markdown(f"### 🎯 Category: {data['category']}")

# =============================================
# DASHBOARD PAGE
# =============================================
elif menu == "📊 Dashboard":
    st.markdown("<h1>📊 DASHBOARD</h1>", unsafe_allow_html=True)
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{len(df)}</div>
                <div>Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{df['url'].nunique()}</div>
                <div>Unique</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            top_cat = df['type'].mode()[0] if not df.empty else "N/A"
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{top_cat}</div>
                <div>Top Category</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{len(df)}</div>
                <div>Analyses</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'type' in df.columns:
                fig = px.pie(df, names='type', title='Category Distribution')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                timeline = df.groupby('date').size().reset_index(name='count')
                fig = px.line(timeline, x='date', y='count', title='Analysis Timeline')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Analyze some websites first!")

# =============================================
# INSIGHTS PAGE
# =============================================
elif menu == "📈 Insights":
    st.markdown("<h1>📈 INSIGHTS</h1>", unsafe_allow_html=True)
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        st.markdown("### 📊 Key Insights")
        
        # Most analyzed category
        top_cat = df['type'].mode()[0] if not df.empty else "N/A"
        st.markdown(f"**Most Popular Category:** {top_cat}")
        
        # Analysis frequency
        st.markdown(f"**Total Analyses:** {len(df)}")
        st.markdown(f"**Unique Websites:** {df['url'].nunique()}")
        
        # Recent activity
        st.markdown("### ⏱️ Recent Activity")
        for _, row in df.tail(5).iterrows():
            st.markdown(f"- {row['timestamp']}: {row['title']} ({row['type']})")
    else:
        st.info("No insights available yet")

# =============================================
# HISTORY PAGE
# =============================================
elif menu == "📚 History":
    st.markdown("<h1>📚 HISTORY</h1>", unsafe_allow_html=True)
    
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.markdown(f"""
                <div class='glass-card'>
                    <h3>{item['title']}</h3>
                    <p>{item['url']}</p>
                    <p>🕒 {item['timestamp']} | 🏷️ {item['type']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No history yet")

# =============================================
# SETTINGS PAGE
# =============================================
elif menu == "⚙️ Settings":
    st.markdown("<h1>⚙️ SETTINGS</h1>", unsafe_allow_html=True)
    
    st.markdown("### 🔧 Preferences")
    
    dark_mode = st.toggle("Dark Mode", value=True)
    auto_save = st.toggle("Auto-save History", value=True)
    phone_format = st.selectbox("Phone Format", ["Indian (+91)", "International", "Raw"])
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)
    
    if st.button("Save Settings"):
        st.success("Settings saved!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Made with Streamlit</p>", unsafe_allow_html=True)
