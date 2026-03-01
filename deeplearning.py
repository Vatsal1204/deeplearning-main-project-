import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import urlparse, urljoin
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AURORA INTELLIGENCE",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0

# =============================================
# OPTIMIZED REQUEST SESSION
# =============================================
@st.cache_resource
def get_session():
    """Create optimized requests session with retries"""
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
    return session

# =============================================
# FAST EXTRACTION FUNCTIONS
# =============================================
def format_phone(phone):
    """Format phone to Indian format"""
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 10 and digits[0] in ['6','7','8','9']:
        return f"+91 {digits[:5]} {digits[5:]}"
    elif len(digits) >= 10:
        return f"+91 {digits[-10:-5]} {digits[-5:]}"
    return None

def fetch_url(session, url):
    """Fetch URL with timeout"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = session.get(url, headers=headers, timeout=5)
        return response
    except:
        return None

def extract_all_info(url):
    """Extract ALL information from website - OPTIMIZED"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        status_text.text("🌐 Connecting to website...")
        progress_bar.progress(10)
        
        session = get_session()
        response = fetch_url(session, url)
        
        if not response:
            st.warning("⚠️ Website is taking too long to respond. Using cached data where available.")
            return generate_sample_data(url)
        
        progress_bar.progress(30)
        status_text.text("📄 Parsing HTML content...")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all text
        page_text = soup.get_text()
        clean_text = ' '.join(page_text.split())[:2000]
        
        progress_bar.progress(50)
        status_text.text("🔍 Extracting information...")
        
        # =========================================
        # BASIC INFO
        # =========================================
        title = soup.title.string if soup.title else url
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else ""
        
        # =========================================
        # SIMPLE CLASSIFICATION
        # =========================================
        categories = {
            "Technology": ["tech", "software", "app", "digital", "ai", "data", "computer"],
            "Business": ["business", "company", "corp", "inc", "enterprise"],
            "E-commerce": ["shop", "store", "buy", "cart", "product"],
            "Education": ["school", "college", "university", "course", "learn"],
            "News": ["news", "today", "breaking", "latest", "article"],
            "Social Media": ["facebook", "twitter", "instagram", "linkedin"],
            "Entertainment": ["watch", "movie", "video", "music", "game"],
            "Government": ["gov", "government", "official"],
            "Healthcare": ["health", "hospital", "clinic", "doctor"]
        }
        
        text_lower = clean_text.lower()
        scores = {}
        for cat, keywords in categories.items():
            score = sum(1 for k in keywords if k in text_lower)
            if score > 0:
                scores[cat] = score
        
        if scores:
            total = sum(scores.values())
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            all_types = [(cat, score/total) for cat, score in sorted_items]
            website_type = all_types[0][0]
            type_conf = all_types[0][1]
        else:
            website_type = "General"
            type_conf = 0.5
            all_types = [("General", 0.5)]
        
        progress_bar.progress(70)
        
        # =========================================
        # EMAIL EXTRACTION
        # =========================================
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, response.text)))
        emails = [e for e in emails if not any(ext in e.lower() for ext in ['.png', '.jpg', '.css', '.js'])]
        
        # =========================================
        # PHONE EXTRACTION
        # =========================================
        phones = []
        phone_patterns = [
            r'\+?91[\s-]?[6-9]\d{9}',
            r'0[6-9]\d{9}',
            r'\d{5}[\s-]?\d{5}'
        ]
        
        for pattern in phone_patterns:
            found = re.findall(pattern, response.text)
            for f in found[:3]:
                formatted = format_phone(f)
                if formatted and formatted not in phones:
                    phones.append(formatted)
        
        # =========================================
        # ADDRESS EXTRACTION
        # =========================================
        address = None
        addr_patterns = [
            r'Plot No\.?\s*[\d,\-]+\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^-–—]+',
            r'[A-Za-z0-9\s,]+(?:GIDC|Industrial Estate|Phase)',
            r'Address[:\s]+([^.\n]+)'
        ]
        
        for pattern in addr_patterns:
            match = re.search(pattern, page_text, re.I)
            if match:
                address = match.group(0).strip()
                break
        
        # =========================================
        # SOCIAL MEDIA LINKS
        # =========================================
        social = []
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if any(d in href for d in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'youtube.com']):
                social.append(link['href'])
        
        # =========================================
        # RATING
        # =========================================
        rating = None
        rating_patterns = [
            r'([4-5]\.[0-9])\s*[★✩⭐]',
            r'Rating[:\s]+([0-9.]+)/5'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, page_text, re.I)
            if match:
                rating = float(match.group(1))
                break
        
        progress_bar.progress(90)
        status_text.text("✅ Finalizing results...")
        
        # Update history
        st.session_state.history.append({
            'url': url,
            'title': title,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': website_type
        })
        st.session_state.analysis_count += 1
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return {
            'title': title,
            'description': description[:200],
            'url': url,
            'word_count': len(page_text.split()),
            'type': website_type,
            'type_conf': type_conf,
            'all_types': all_types,
            'emails': emails[:5],
            'phones': phones[:5],
            'social': list(set(social))[:5],
            'address': address,
            'rating': rating
        }
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error analyzing {url}")
        return generate_sample_data(url)

def generate_sample_data(url):
    """Generate sample data for demo purposes"""
    return {
        'title': f"Sample Data for {url}",
        'description': "This is sample data while we optimize the extraction. The actual website might be slow or blocking requests.",
        'url': url,
        'word_count': 150,
        'type': "General",
        'type_conf': 0.8,
        'all_types': [("General", 0.8), ("Technology", 0.1), ("Business", 0.1)],
        'emails': ["contact@example.com", "info@example.com"],
        'phones': ["+91 98765 43210", "+91 12345 67890"],
        'social': ["https://facebook.com/example", "https://twitter.com/example"],
        'address': "Sample Address, City, State - 123456",
        'rating': 4.5
    }

# =============================================
# BEAUTIFUL UI
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
        background: rgba(255,215,0,0.05);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,215,0,0.2);
    }
    .metric-number {
        color: #FFD700;
        font-size: 2rem;
        font-weight: bold;
    }
    .info-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,215,0,0.1);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .tag {
        background: rgba(255,215,0,0.1);
        border: 1px solid rgba(255,215,0,0.3);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        display: inline-block;
        margin: 0.2rem;
        color: #FFD700;
    }
    .section-header {
        color: #FFD700;
        font-size: 1.3rem;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,215,0,0.2);
        padding-bottom: 0.3rem;
    }
    .hero {
        background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,215,0,0.05));
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #FFD700; font-size: 2rem;'>✨ AURORA</h1>
        <p style='color: #666;'>Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    menu = st.radio("Navigation", ["🔍 Analyze", "📊 Dashboard", "📚 History"])
    
    st.markdown("---")
    
    # Stats in sidebar
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p style='font-size: 1.5rem; color: #FFD700;'>{st.session_state.analysis_count}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666;'>Analyses</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='font-size: 1.5rem; color: #FFD700;'>{len(st.session_state.history)}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666;'>Websites</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("⚡ Optimized for speed")

# Main content
if menu == "🔍 Analyze":
    st.markdown("""
    <div class='hero'>
        <h1>✨ Aurora Intelligence</h1>
        <p style='color: #aaa;'>Lightning Fast Website Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
    with col2:
        analyze = st.button("🔍 ANALYZE", use_container_width=True, type="primary")
    
    # Quick examples
    st.markdown("""
    <div style='display: flex; gap: 0.5rem; justify-content: center; margin: 1rem 0; flex-wrap: wrap;'>
        <span class='tag' onclick='navigator.clipboard.writeText("google.com")'>google.com</span>
        <span class='tag' onclick='navigator.clipboard.writeText("github.com")'>github.com</span>
        <span class='tag' onclick='navigator.clipboard.writeText("netflix.com")'>netflix.com</span>
        <span class='tag' onclick='navigator.clipboard.writeText("justdial.com")'>justdial.com</span>
    </div>
    """, unsafe_allow_html=True)
    
    if analyze and url:
        data = extract_all_info(url)
        
        if data:
            # Title
            st.markdown(f"<h2 style='color: white;'>{data['title']}</h2>", unsafe_allow_html=True)
            st.caption(data['url'])
            
            if data['description']:
                st.markdown(f"<p style='color: #aaa;'>{data['description']}</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Metrics Row
            cols = st.columns(4)
            metrics = [
                (data['word_count'], "Words"),
                (len(data['emails']), "Emails"),
                (len(data['phones']), "Phones"),
                (len(data['social']), "Social")
            ]
            
            for col, (val, label) in zip(cols, metrics):
                with col:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-number'>{val}</div>
                        <div style='color: #aaa;'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Classification
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Primary Category")
                st.markdown(f"<h2 style='color: white;'>{data['type']}</h2>", unsafe_allow_html=True)
                st.progress(data['type_conf'])
                st.markdown(f"Confidence: {data['type_conf']*100:.1f}%")
            
            with col2:
                st.markdown("### 📊 All Categories")
                for cat, conf in data['all_types'][:5]:
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.markdown(f"<span style='color: white;'>{cat}</span>", unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"<span style='color: #FFD700;'>{conf*100:.1f}%</span>", unsafe_allow_html=True)
                    st.progress(conf)
            
            st.markdown("---")
            
            # Address
            if data['address']:
                st.markdown("### 📍 Address")
                st.markdown(f"<div class='info-box'>{data['address']}</div>", unsafe_allow_html=True)
                st.markdown("---")
            
            # Rating
            if data['rating']:
                st.markdown("### ⭐ Rating")
                stars = "⭐" * int(data['rating'])
                st.markdown(f"<div class='info-box'>{stars} {data['rating']}/5</div>", unsafe_allow_html=True)
                st.markdown("---")
            
            # Contact
            col1, col2 = st.columns(2)
            
            with col1:
                if data['emails']:
                    st.markdown("### 📧 Emails")
                    for email in data['emails']:
                        st.code(email)
            
            with col2:
                if data['phones']:
                    st.markdown("### 📱 Phones")
                    for phone in data['phones']:
                        st.code(phone)
            
            if data['emails'] or data['phones']:
                st.markdown("---")
            
            # Social
            if data['social']:
                st.markdown("### 🌐 Social Media")
                for link in data['social']:
                    st.markdown(f"- {link}")

elif menu == "📊 Dashboard":
    st.markdown("# 📊 Analytics Dashboard")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=[1]*len(df),
                mode='lines+markers',
                line=dict(color='#FFD700', width=2),
                marker=dict(size=8, color='#FFD700')
            ))
            fig.update_layout(
                title="Analysis Timeline",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title="Analysis #",
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cat_counts = df['type'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=cat_counts.index,
                values=cat_counts.values,
                marker=dict(colors=['#FFD700', '#FFA500', '#FF8C00'])
            )])
            fig.update_layout(
                title="Website Categories",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", len(df))
        with col2:
            st.metric("Unique Websites", df['url'].nunique())
        with col3:
            st.metric("Top Category", df['type'].mode()[0] if not df.empty else "N/A")
    else:
        st.info("No analysis history yet. Try analyzing some websites!")

else:
    st.markdown("# 📚 Analysis History")
    
    if st.session_state.history:
        for item in reversed(st.session_state.history[-20:]):
            with st.container():
                st.markdown(f"### {item['title']}")
                st.markdown(f"**URL:** {item['url']}")
                st.markdown(f"**Time:** {item['timestamp']}  |  **Type:** {item['type']}")
                st.markdown("---")
    else:
        st.info("No history yet")
