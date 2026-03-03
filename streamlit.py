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
# REAL EXTRACTION FUNCTIONS
# =============================================
def format_phone(phone):
    """Format phone to Indian format"""
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 10 and digits[0] in ['6','7','8','9']:
        return f"+91 {digits[:5]} {digits[5:]}"
    elif len(digits) == 11 and digits.startswith('0'):
        return f"+91 {digits[1:6]} {digits[6:]}"
    elif len(digits) == 12 and digits.startswith('91'):
        return f"+91 {digits[2:7]} {digits[7:]}"
    elif len(digits) > 10:
        return f"+91 {digits[-10:-5]} {digits[-5:]}"
    return None

def extract_real_data(url):
    """Extract REAL data from website"""
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🌐 Connecting to website...")
        progress_bar.progress(20)
        
        response = requests.get(url, headers=headers, timeout=10)
        
        status_text.text("📄 Parsing HTML...")
        progress_bar.progress(40)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all text
        page_text = soup.get_text()
        clean_text = ' '.join(page_text.split())[:3000]
        
        status_text.text("🔍 Extracting information...")
        progress_bar.progress(60)
        
        # =========================================
        # BASIC INFO
        # =========================================
        title = soup.title.string if soup.title else url
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else ""
        
        # =========================================
        # EMAIL EXTRACTION - REAL
        # =========================================
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, response.text)))
        emails = [e for e in emails if not any(ext in e.lower() for ext in ['.png', '.jpg', '.css', '.js', '.svg'])]
        
        # =========================================
        # PHONE EXTRACTION - REAL
        # =========================================
        phone_patterns = [
            r'\+?91[\s-]?[6-9]\d{9}',
            r'0[6-9]\d{9}',
            r'\d{5}[\s-]?\d{5}',
            r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',
            r'\d{10}'
        ]
        
        phones = []
        for pattern in phone_patterns:
            found = re.findall(pattern, response.text)
            for f in found:
                formatted = format_phone(f)
                if formatted and formatted not in phones:
                    phones.append(formatted)
        
        # =========================================
        # ADDRESS EXTRACTION - REAL
        # =========================================
        address = None
        addr_patterns = [
            r'Plot No\.?\s*[\d,\-]+\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^-–—]+',
            r'[A-Za-z0-9\s,]+(?:GIDC|Industrial Estate|Phase)[^,]+(?:Jamnagar|Gujarat|Mumbai|Delhi|Bangalore)',
            r'Address[:\s]+([^.\n]+(?:\.[^.\n]+)*)',
            r'Located at[:\s]+([^.\n]+)'
        ]
        
        for pattern in addr_patterns:
            match = re.search(pattern, page_text, re.I)
            if match:
                address = match.group(0).strip()
                break
        
        # =========================================
        # SOCIAL MEDIA LINKS - REAL
        # =========================================
        social = []
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if any(d in href for d in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'youtube.com']):
                full_url = urljoin(url, link['href'])
                social.append(full_url)
        
        # =========================================
        # RATING EXTRACTION - REAL
        # =========================================
        rating = None
        rating_patterns = [
            r'([4-5]\.[0-9])\s*[★✩⭐]',
            r'Rating[:\s]+([0-9.]+)/5',
            r'([0-9.]+)\s*out of\s*5'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, page_text, re.I)
            if match:
                try:
                    rating = float(match.group(1))
                except:
                    pass
                break
        
        # =========================================
        # CLASSIFICATION
        # =========================================
        categories = {
            "Technology": ["tech", "software", "app", "digital", "ai", "data", "computer", "cloud"],
            "Business": ["business", "company", "corp", "inc", "enterprise", "ltd", "private"],
            "E-commerce": ["shop", "store", "buy", "cart", "product", "price", "order", "checkout"],
            "Education": ["school", "college", "university", "course", "learn", "education", "student"],
            "News": ["news", "today", "breaking", "latest", "headline", "article", "press"],
            "Social Media": ["facebook", "twitter", "instagram", "linkedin", "share", "post"],
            "Entertainment": ["watch", "movie", "video", "music", "game", "play", "stream"],
            "Government": ["gov", "government", "official", "ministry", "public"],
            "Healthcare": ["health", "hospital", "clinic", "doctor", "medical", "care"]
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
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Update history
        st.session_state.history.append({
            'url': url,
            'title': title[:50],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': website_type
        })
        st.session_state.analysis_count += 1
        
        return {
            'title': title,
            'description': description[:300],
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
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Website took too long to respond. Try another site.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to website. Check the URL.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# =============================================
# UI
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
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='color: #FFD700;'>✨ AURORA</h1>", unsafe_allow_html=True)
    menu = st.radio("Navigation", ["🔍 Analyze", "📊 Dashboard", "📚 History"])
    st.markdown("---")
    st.markdown(f"**Analyses:** {st.session_state.analysis_count}")
    st.markdown(f"**Websites:** {len(st.session_state.history)}")

# Main content
if menu == "🔍 Analyze":
    st.markdown("""
    <div class='hero'>
        <h1>✨ Aurora Intelligence</h1>
        <p style='color: #aaa;'>Real Website Data Extraction</p>
    </div>
    """, unsafe_allow_html=True)
    
    url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
    
    if st.button("🔍 ANALYZE", use_container_width=True):
        if url:
            data = extract_real_data(url)
            
            if data:
                st.markdown(f"<h2 style='color: white;'>{data['title']}</h2>", unsafe_allow_html=True)
                st.caption(data['url'])
                
                if data['description']:
                    st.markdown(f"<p style='color: #aaa;'>{data['description']}</p>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Metrics
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
                    st.markdown("### 🎯 Category")
                    st.markdown(f"## {data['type']}")
                    st.progress(data['type_conf'])
                with col2:
                    st.markdown("### 📊 All Categories")
                    for cat, conf in data['all_types']:
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(cat)
                        with cols[1]:
                            st.markdown(f"{conf*100:.1f}%")
                        st.progress(conf)
                
                st.markdown("---")
                
                # Address
                if data['address']:
                    st.markdown("### 📍 Address")
                    st.markdown(f"<div class='info-box'>{data['address']}</div>", unsafe_allow_html=True)
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
                
                # Social
                if data['social']:
                    st.markdown("### 🌐 Social")
                    for link in data['social']:
                        st.markdown(f"- {link}")

elif menu == "📊 Dashboard":
    st.markdown("# 📊 Dashboard")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.info("No history yet")

else:
    st.markdown("# 📚 History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"**{item['title']}** - {item['type']}")
            st.caption(item['url'])
            st.markdown("---")
    else:
        st.info("No history yet")
