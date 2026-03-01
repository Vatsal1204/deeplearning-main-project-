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
# FAST EXTRACTION FUNCTIONS (NO DEEP LEARNING)
# =============================================
def format_phone(phone):
    """Format phone to Indian format"""
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 10 and digits[0] in ['6','7','8','9']:
        return f"+91 {digits[:5]} {digits[5:]}"
    elif len(digits) >= 10:
        return f"+91 {digits[-10:-5]} {digits[-5:]}"
    return None

def extract_all_info(url):
    """Extract ALL information from website - FAST VERSION"""
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all text
        page_text = soup.get_text()
        clean_text = ' '.join(page_text.split())[:2000]
        
        # =========================================
        # BASIC INFO
        # =========================================
        title = soup.title.string if soup.title else "No title found"
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else ""
        
        # Meta keywords
        meta_keys = soup.find('meta', attrs={'name': 'keywords'})
        keywords = meta_keys['content'] if meta_keys else ""
        
        # =========================================
        # SIMPLE ORGANIZATION EXTRACTION
        # =========================================
        orgs = []
        words = clean_text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                if any(term in word.lower() for term in ['inc', 'corp', 'ltd', 'co', 'tech', 'solutions', 'media', 'group']):
                    orgs.append(word)
        
        # =========================================
        # SIMPLE CLASSIFICATION
        # =========================================
        categories = {
            "Technology": ["tech", "software", "app", "digital", "ai", "data", "computer", "cloud"],
            "Business": ["business", "company", "corp", "inc", "enterprise", "ltd", "private"],
            "E-commerce": ["shop", "store", "buy", "cart", "product", "price", "checkout", "order"],
            "Education": ["school", "college", "university", "course", "learn", "education", "student"],
            "News": ["news", "today", "breaking", "latest", "headline", "article", "press"],
            "Social Media": ["facebook", "twitter", "instagram", "linkedin", "share", "post", "social"],
            "Entertainment": ["watch", "movie", "video", "music", "game", "play", "stream"],
            "Government": ["gov", "government", "official", "ministry", "public", "department"],
            "Healthcare": ["health", "hospital", "clinic", "doctor", "medical", "care", "patient"]
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
        
        # =========================================
        # EMAIL EXTRACTION
        # =========================================
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, response.text)))
        emails = [e for e in emails if not any(ext in e.lower() for ext in ['.png', '.jpg', '.css', '.js', '.svg'])]
        
        # =========================================
        # PHONE EXTRACTION
        # =========================================
        phones = []
        phone_patterns = [
            r'\+?91[\s-]?[6-9]\d{9}',
            r'0[6-9]\d{9}',
            r'\d{5}[\s-]?\d{5}',
            r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            found = re.findall(pattern, response.text)
            for f in found:
                formatted = format_phone(f)
                if formatted and formatted not in phones:
                    phones.append(formatted)
        
        # =========================================
        # ADDRESS EXTRACTION
        # =========================================
        address = None
        addr_patterns = [
            r'Plot No\.?\s*[\d,\-]+\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^-–—]+',
            r'[A-Za-z0-9\s,]+(?:GIDC|Industrial Estate|Phase)[^,]+(?:Jamnagar|Gujarat)',
            r'Address[:\s]+([^.\n]+(?:\.[^.\n]+)*)',
            r'Located at[:\s]+([^.\n]+)'
        ]
        
        for pattern in addr_patterns:
            match = re.search(pattern, page_text, re.I)
            if match:
                address = match.group(0).strip()
                break
        
        # =========================================
        # SOCIAL MEDIA LINKS
        # =========================================
        social = {
            'facebook': [],
            'twitter': [],
            'linkedin': [],
            'instagram': [],
            'youtube': []
        }
        
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if 'facebook.com' in href:
                social['facebook'].append(link['href'])
            elif 'twitter.com' in href or 'x.com' in href:
                social['twitter'].append(link['href'])
            elif 'linkedin.com' in href:
                social['linkedin'].append(link['href'])
            elif 'instagram.com' in href:
                social['instagram'].append(link['href'])
            elif 'youtube.com' in href:
                social['youtube'].append(link['href'])
        
        # =========================================
        # BUSINESS HOURS
        # =========================================
        hours = None
        hours_patterns = [
            r'(?:Open|Hours|Timing)[:\s]+([^.\n]+)',
            r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM)\s*[–-]\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM))'
        ]
        
        for pattern in hours_patterns:
            match = re.search(pattern, page_text, re.I)
            if match:
                hours = match.group(0).strip()
                break
        
        # =========================================
        # RATING
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
                rating = float(match.group(1))
                break
        
        # =========================================
        # REVIEWS COUNT
        # =========================================
        reviews = None
        reviews_pattern = r'([0-9,]+)\s*(?:reviews?|Ratings?)'
        match = re.search(reviews_pattern, page_text, re.I)
        if match:
            reviews = match.group(1)
        
        # =========================================
        # WEBSITE TECHNOLOGY
        # =========================================
        tech_stack = []
        tech_patterns = {
            'WordPress': ['wp-content', 'wordpress'],
            'Shopify': ['shopify'],
            'WooCommerce': ['woocommerce'],
            'React': ['react', 'reactjs'],
            'Angular': ['angular', 'ng-'],
            'Vue.js': ['vue'],
            'Bootstrap': ['bootstrap'],
            'jQuery': ['jquery'],
            'PHP': ['.php'],
            'Python': ['django', 'flask'],
            'Node.js': ['node', 'express']
        }
        
        html_content = response.text.lower()
        for tech, patterns in tech_patterns.items():
            if any(p in html_content for p in patterns):
                tech_stack.append(tech)
        
        # Update history
        st.session_state.history.append({
            'url': url,
            'title': title,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': website_type
        })
        st.session_state.analysis_count += 1
        
        return {
            'basic': {
                'title': title,
                'description': description,
                'keywords': keywords,
                'url': url,
                'word_count': len(page_text.split())
            },
            'entities': {
                'orgs': list(set(orgs))[:10]
            },
            'classification': {
                'website_type': website_type,
                'confidence': type_conf,
                'all_types': all_types
            },
            'contact': {
                'emails': emails[:8],
                'phones': phones[:8]
            },
            'business': {
                'address': address,
                'hours': hours,
                'rating': rating,
                'reviews': reviews
            },
            'social': {k: list(set(v))[:3] for k, v in social.items() if v},
            'technology': tech_stack[:10]
        }
        
    except Exception as e:
        st.error(f"Error analyzing {url}: {str(e)}")
        return None

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
        padding: 0.2rem 0.8rem;
        display: inline-block;
        margin: 0.2rem;
        color: #FFD700;
        font-size: 0.9rem;
    }
    .section-header {
        color: #FFD700;
        font-size: 1.3rem;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,215,0,0.2);
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# ✨ **AURORA**")
    st.markdown("---")
    st.success("⚡ FAST MODE ACTIVE")
    menu = st.radio("Navigation", ["🔍 Analyze", "📊 Dashboard", "📚 History"])
    st.markdown("---")
    st.markdown(f"**Analyses:** {st.session_state.analysis_count}")
    st.markdown(f"**Websites:** {len(st.session_state.history)}")

# Main content
if menu == "🔍 Analyze":
    st.markdown("# ✨ Aurora Intelligence")
    st.markdown("### Lightning Fast Website Analysis")
    
    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
    with col2:
        analyze = st.button("✨ ANALYZE", use_container_width=True)
    
    if analyze and url:
        with st.spinner("Analyzing website..."):
            data = extract_all_info(url)
            
            if data:
                # Title
                st.markdown(f"## {data['basic']['title']}")
                st.caption(data['basic']['url'])
                
                if data['basic']['description']:
                    st.markdown(f"*{data['basic']['description']}*")
                
                st.markdown("---")
                
                # Metrics Row
                cols = st.columns(5)
                metrics = [
                    (data['basic']['word_count'], "Words"),
                    (len(data['entities']['orgs']), "Organizations"),
                    (len(data['contact']['emails']), "Emails"),
                    (len(data['contact']['phones']), "Phones"),
                    (len(data['technology']), "Technologies")
                ]
                
                for col, (val, label) in zip(cols, metrics):
                    with col:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-number'>{val}</div>
                            <div>{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Classification
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Primary Category")
                    st.markdown(f"## {data['classification']['website_type']}")
                    st.progress(data['classification']['confidence'])
                    st.markdown(f"Confidence: {data['classification']['confidence']*100:.1f}%")
                
                with col2:
                    st.markdown("### 📊 All Categories")
                    for cat, conf in data['classification']['all_types'][:5]:
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(cat)
                        with c2:
                            st.markdown(f"{conf*100:.1f}%")
                        st.progress(conf)
                
                st.markdown("---")
                
                # Organizations
                if data['entities']['orgs']:
                    st.markdown("### 🏢 Organizations")
                    cols = st.columns(4)
                    for i, org in enumerate(data['entities']['orgs']):
                        with cols[i % 4]:
                            st.markdown(f"<span class='tag'>{org}</span>", unsafe_allow_html=True)
                    st.markdown("---")
                
                # Business Information
                if data['business']['address'] or data['business']['hours'] or data['business']['rating']:
                    st.markdown("### 🏢 Business Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if data['business']['address']:
                            st.markdown("**Address:**")
                            st.markdown(f"<div class='info-box'>{data['business']['address']}</div>", unsafe_allow_html=True)
                        
                        if data['business']['hours']:
                            st.markdown("**Hours:**")
                            st.markdown(f"<div class='info-box'>{data['business']['hours']}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        if data['business']['rating']:
                            st.markdown("**Rating:**")
                            stars = "⭐" * int(data['business']['rating'])
                            st.markdown(f"<div class='info-box'>{stars} {data['business']['rating']}/5</div>", unsafe_allow_html=True)
                        
                        if data['business']['reviews']:
                            st.markdown("**Reviews:**")
                            st.markdown(f"<div class='info-box'>{data['business']['reviews']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # Contact Information
                if data['contact']['emails'] or data['contact']['phones']:
                    st.markdown("### 📞 Contact")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if data['contact']['emails']:
                            st.markdown("**Emails:**")
                            for email in data['contact']['emails']:
                                st.code(email)
                    
                    with col2:
                        if data['contact']['phones']:
                            st.markdown("**Phones:**")
                            for phone in data['contact']['phones']:
                                st.code(phone)
                    st.markdown("---")
                
                # Social Media
                if data['social']:
                    st.markdown("### 🌐 Social Media")
                    tabs = st.tabs(list(data['social'].keys()))
                    for i, (platform, links) in enumerate(data['social'].items()):
                        with tabs[i]:
                            for link in links:
                                st.markdown(f"- {link}")
                    st.markdown("---")
                
                # Technology Stack
                if data['technology']:
                    st.markdown("### 🛠️ Technology Stack")
                    cols = st.columns(4)
                    for i, tech in enumerate(data['technology']):
                        with cols[i % 4]:
                            st.markdown(f"<span class='tag'>{tech}</span>", unsafe_allow_html=True)

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
                font=dict(color='white')
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
        st.info("No analysis history yet")

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
