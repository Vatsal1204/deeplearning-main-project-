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
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="NEURAL AURORA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0

# =============================================
# LOAD DEEP LEARNING MODELS
# =============================================
@st.cache_resource
def load_dl_models():
    """Load deep learning models"""
    models = {}
    
    with st.spinner("🧠 Loading Deep Learning Models..."):
        # Entity Recognition
        models['ner'] = pipeline(
            "ner", 
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
        
        # Classification
        models['classifier'] = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Sentiment Analysis
        models['sentiment'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
    return models

# Load models with error handling
try:
    models = load_dl_models()
    dl_available = True
    st.sidebar.success("🧠 Deep Learning Active")
except Exception as e:
    st.sidebar.warning("⚠️ Deep Learning unavailable (memory limit)")
    dl_available = False

# =============================================
# COMPLETE EXTRACTION FUNCTIONS
# =============================================
def extract_all_info(url):
    """Extract ALL possible information from website"""
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Use multiple user agents to avoid blocking
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        headers = {'User-Agent': np.random.choice(user_agents)}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all text
        page_text = soup.get_text()
        clean_text = ' '.join(page_text.split())[:3000]
        
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
        
        # Open Graph tags (Facebook)
        og_title = soup.find('meta', property='og:title')
        og_title = og_title['content'] if og_title else ""
        
        og_desc = soup.find('meta', property='og:description')
        og_desc = og_desc['content'] if og_desc else ""
        
        og_image = soup.find('meta', property='og:image')
        og_image = og_image['content'] if og_image else ""
        
        # Twitter cards
        twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
        twitter_title = twitter_title['content'] if twitter_title else ""
        
        # =========================================
        # DEEP LEARNING ENTITY RECOGNITION
        # =========================================
        orgs = []
        people = []
        locations = []
        
        if dl_available:
            ner_results = models['ner'](clean_text[:1024])
            
            for entity in ner_results:
                if entity['entity_group'] == 'ORG':
                    orgs.append(entity['word'])
                elif entity['entity_group'] == 'PER':
                    people.append(entity['word'])
                elif entity['entity_group'] == 'LOC':
                    locations.append(entity['word'])
        
        # =========================================
        # CLASSIFICATION
        # =========================================
        if dl_available:
            categories = [
                "Technology", "Business", "E-commerce", "Education", 
                "News", "Social Media", "Entertainment", "Government", 
                "Healthcare", "Finance", "Real Estate", "Travel"
            ]
            
            classification = models['classifier'](clean_text[:500], categories)
            website_type = classification['labels'][0]
            type_conf = classification['scores'][0]
            all_types = list(zip(classification['labels'][:8], classification['scores'][:8]))
        else:
            # Fallback
            website_type = "Unknown"
            type_conf = 0.5
            all_types = [("Unknown", 0.5)]
        
        # =========================================
        # SENTIMENT ANALYSIS
        # =========================================
        sentiment = "Neutral"
        sentiment_score = 0.5
        
        if dl_available and len(clean_text) > 100:
            try:
                sentiment_result = models['sentiment'](clean_text[:512])[0]
                sentiment = sentiment_result['label']
                sentiment_score = sentiment_result['score']
            except:
                pass
        
        # =========================================
        # EMAIL EXTRACTION
        # =========================================
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, response.text)))
        emails = [e for e in emails if not any(ext in e.lower() for ext in ['.png', '.jpg', '.css', '.js', '.svg'])]
        
        # =========================================
        # PHONE EXTRACTION
        # =========================================
        def format_phone(phone):
            digits = re.sub(r'\D', '', str(phone))
            if len(digits) == 10 and digits[0] in ['6','7','8','9']:
                return f"+91 {digits[:5]} {digits[5:]}"
            elif len(digits) == 11 and digits.startswith('0'):
                return f"{digits[1:6]} {digits[6:]}"
            elif len(digits) == 12 and digits.startswith('91'):
                return f"+{digits[:2]} {digits[2:7]} {digits[7:]}"
            return None
        
        phone_patterns = [
            r'\+?91[\s-]?[6-9]\d{9}',
            r'0[6-9]\d{9}',
            r'\d{5}[\s-]?\d{5}',
            r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}'
        ]
        
        phones = []
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
        hours_patterns = [
            r'(?:Open|Hours|Timing)[:\s]+([^.\n]+)',
            r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM)\s*[–-]\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM))'
        ]
        
        hours = None
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
            'type': website_type,
            'dl_used': dl_available
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
            'social_meta': {
                'og_title': og_title,
                'og_desc': og_desc,
                'og_image': og_image,
                'twitter_title': twitter_title
            },
            'deep_learning': {
                'orgs': list(set(orgs))[:10],
                'people': list(set(people))[:10],
                'locations': list(set(locations))[:10],
                'website_type': website_type,
                'confidence': type_conf,
                'all_types': all_types,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'dl_used': dl_available
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
        padding: 1.5rem;
        text-align: center;
    }
    .metric-number {
        color: #FFD700;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .info-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,215,0,0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
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
        font-size: 1.5rem;
        margin: 2rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,215,0,0.2);
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# 🧠 **NEURAL AURORA**")
    st.markdown("---")
    
    if dl_available:
        st.success("🧠 Deep Learning: ACTIVE")
        st.info("Models: BERT, BART, RoBERTa")
    else:
        st.warning("⚠️ Deep Learning: FALLBACK MODE")
    
    menu = st.radio("Navigation", ["🔍 Analyze", "📊 Dashboard", "📚 History"])
    
    st.markdown("---")
    st.markdown(f"**Analyses:** {st.session_state.analysis_count}")
    st.markdown(f"**Websites:** {len(st.session_state.history)}")

# Main content
if menu == "🔍 Analyze":
    st.markdown("# 🧠 Neural Aurora")
    st.markdown("### Deep Learning Website Intelligence")
    
    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
    with col2:
        analyze = st.button("🧠 ANALYZE", use_container_width=True)
    
    if analyze and url:
        with st.spinner("🧠 Deep Learning models processing..."):
            data = extract_all_info(url)
            
            if data:
                # Title
                st.markdown(f"# {data['basic']['title']}")
                st.caption(data['basic']['url'])
                
                if data['basic']['description']:
                    st.markdown(f"*{data['basic']['description']}*")
                
                st.markdown("---")
                
                # Metrics Row
                cols = st.columns(5)
                metrics = [
                    (data['basic']['word_count'], "Words"),
                    (len(data['deep_learning']['orgs']), "Organizations"),
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
                
                # Deep Learning Section
                st.markdown("<div class='section-header'>🧠 Deep Learning Analysis</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Primary Classification")
                    st.markdown(f"## {data['deep_learning']['website_type']}")
                    st.progress(data['deep_learning']['confidence'])
                    st.markdown(f"Confidence: {data['deep_learning']['confidence']*100:.1f}%")
                    
                    if data['deep_learning']['sentiment']:
                        st.markdown(f"### Sentiment: {data['deep_learning']['sentiment']}")
                        st.progress(data['deep_learning']['sentiment_score'])
                
                with col2:
                    st.markdown("### All Categories")
                    for cat, conf in data['deep_learning']['all_types'][:5]:
                        c1, c2, c3 = st.columns([3, 1, 4])
                        with c1:
                            st.markdown(cat)
                        with c2:
                            st.markdown(f"{conf*100:.1f}%")
                        with c3:
                            st.progress(conf)
                
                # Entities
                if data['deep_learning']['orgs'] or data['deep_learning']['people'] or data['deep_learning']['locations']:
                    st.markdown("### Entities Found")
                    
                    if data['deep_learning']['orgs']:
                        st.markdown("**Organizations:**")
                        cols = st.columns(4)
                        for i, org in enumerate(data['deep_learning']['orgs'][:4]):
                            with cols[i % 4]:
                                st.markdown(f"<span class='tag'>{org}</span>", unsafe_allow_html=True)
                    
                    if data['deep_learning']['people']:
                        st.markdown("**People:**")
                        st.write(", ".join(data['deep_learning']['people'][:5]))
                    
                    if data['deep_learning']['locations']:
                        st.markdown("**Locations:**")
                        st.write(", ".join(data['deep_learning']['locations'][:5]))
                
                st.markdown("---")
                
                # Business Information
                if data['business']['address'] or data['business']['hours'] or data['business']['rating']:
                    st.markdown("<div class='section-header'>🏢 Business Information</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if data['business']['address']:
                            st.markdown("**Address:**")
                            st.markdown(f"```\n{data['business']['address']}\n```")
                        
                        if data['business']['hours']:
                            st.markdown("**Hours:**")
                            st.markdown(data['business']['hours'])
                    
                    with col2:
                        if data['business']['rating']:
                            st.markdown("**Rating:**")
                            st.markdown(f"{'⭐' * int(data['business']['rating'])} {data['business']['rating']}/5")
                        
                        if data['business']['reviews']:
                            st.markdown("**Reviews:**")
                            st.markdown(data['business']['reviews'])
                    
                    st.markdown("---")
                
                # Contact Information
                if data['contact']['emails'] or data['contact']['phones']:
                    st.markdown("<div class='section-header'>📞 Contact Information</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if data['contact']['emails']:
                            st.markdown("**Email Addresses:**")
                            for email in data['contact']['emails']:
                                st.code(email)
                    
                    with col2:
                        if data['contact']['phones']:
                            st.markdown("**Phone Numbers:**")
                            for phone in data['contact']['phones']:
                                st.code(phone)
                    
                    st.markdown("---")
                
                # Social Media
                if data['social']:
                    st.markdown("<div class='section-header'>🌐 Social Media</div>", unsafe_allow_html=True)
                    
                    cols = st.columns(len(data['social']))
                    for i, (platform, links) in enumerate(data['social'].items()):
                        with cols[i]:
                            st.markdown(f"**{platform.title()}**")
                            for link in links:
                                st.markdown(f"[Link]({link})")
                    
                    st.markdown("---")
                
                # Technology Stack
                if data['technology']:
                    st.markdown("<div class='section-header'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
                    
                    cols = st.columns(4)
                    for i, tech in enumerate(data['technology']):
                        with cols[i % 4]:
                            st.markdown(f"<span class='tag'>{tech}</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # Social Meta Tags
                if data['social_meta']['og_title'] or data['social_meta']['og_desc']:
                    with st.expander("📱 Social Media Meta Tags"):
                        if data['social_meta']['og_title']:
                            st.markdown(f"**OG Title:** {data['social_meta']['og_title']}")
                        if data['social_meta']['og_desc']:
                            st.markdown(f"**OG Description:** {data['social_meta']['og_desc']}")
                        if data['social_meta']['og_image']:
                            st.markdown(f"**OG Image:** {data['social_meta']['og_image']}")
                
                # Keywords
                if data['basic']['keywords']:
                    with st.expander("🔑 Meta Keywords"):
                        st.markdown(data['basic']['keywords'])

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
                marker=dict(colors=['#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6B6B'])
            )])
            fig.update_layout(
                title="Website Categories",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", len(df))
        with col2:
            st.metric("Unique Websites", df['url'].nunique())
        with col3:
            st.metric("Top Category", df['type'].mode()[0] if not df.empty else "N/A")
        with col4:
            if 'dl_used' in df.columns:
                dl_pct = (df['dl_used'].sum() / len(df)) * 100
                st.metric("Deep Learning %", f"{dl_pct:.1f}%")
    else:
        st.info("No analysis history yet")

else:  # History
    st.markdown("# 📚 Analysis History")
    
    if st.session_state.history:
        for item in reversed(st.session_state.history[-20:]):
            with st.container():
                dl_badge = "🧠" if item.get('dl_used', False) else "⚡"
                st.markdown(f"### {dl_badge} {item['title']}")
                st.markdown(f"**URL:** {item['url']}")
                st.markdown(f"**Time:** {item['timestamp']}  |  **Type:** {item['type']}")
                st.markdown("---")
    else:
        st.info("No history yet")
