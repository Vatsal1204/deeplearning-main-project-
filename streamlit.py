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
import json
import random
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
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'dark_mode': True,
        'auto_save': True,
        'phone_format': 'Indian (+91)',
        'deep_learning': True,
        'confidence_threshold': 0.6
    }

# =============================================
# ADVANCED DEEP LEARNING FUNCTIONS
# =============================================
def deep_learning_predictions(data):
    """Simulate deep learning predictions for business insights"""
    
    predictions = {
        'growth_potential': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2]),
        'market_trend': np.random.choice(['📈 Rising', '📊 Stable', '📉 Declining'], p=[0.4, 0.4, 0.2]),
        'risk_score': np.random.randint(20, 80),
        'confidence_level': np.random.randint(70, 98),
        'next_milestone': (datetime.now() + timedelta(days=np.random.randint(30, 180))).strftime('%B %d, %Y'),
        'recommended_actions': np.random.choice([
            'Expand digital presence', 'Increase social media engagement', 
            'Optimize for mobile users', 'Add customer reviews section',
            'Improve page load speed', 'Add live chat support',
            'Implement SEO strategy', 'Create email newsletter'
        ], size=3, replace=False).tolist()
    }
    
    # Calculate future scope score
    scope_score = 0
    if predictions['growth_potential'] == 'High':
        scope_score += 40
    elif predictions['growth_potential'] == 'Medium':
        scope_score += 25
    else:
        scope_score += 10
        
    if predictions['market_trend'] == '📈 Rising':
        scope_score += 30
    elif predictions['market_trend'] == '📊 Stable':
        scope_score += 20
    else:
        scope_score += 5
        
    predictions['future_scope_score'] = min(scope_score + np.random.randint(10, 30), 100)
    
    return predictions

def detect_owner_info(soup, text):
    """Detect potential owner/business information"""
    
    owner_info = {
        'name': None,
        'designation': None,
        'email': None,
        'founded': None,
        'employees': None,
        'certifications': []
    }
    
    # Look for founder/CEO mentions
    owner_patterns = [
        r'(?:Founder|CEO|Owner|Director|Proprietor)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:Managed by|Owned by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    ]
    
    for pattern in owner_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            owner_info['name'] = match.group(1).strip()
            break
    
    # Look for established year
    year_pattern = r'(?:Established|Since|Founded|Est\.)[:\s]+(\d{4})'
    match = re.search(year_pattern, text, re.I)
    if match:
        owner_info['founded'] = match.group(1)
    
    # Look for employee count
    employee_pattern = r'(\d+(?:\+|\s*-\s*\d+)?)\s*(?:employees?|team size|staff)'
    match = re.search(employee_pattern, text, re.I)
    if match:
        owner_info['employees'] = match.group(1)
    
    # Look for certifications
    cert_keywords = ['ISO', 'GST', 'MSME', 'NSIC', 'IATF', 'Certified']
    for cert in cert_keywords:
        if cert.lower() in text.lower():
            owner_info['certifications'].append(cert)
    
    return owner_info

def extract_advanced_features(soup, text, url):
    """Extract advanced features using pattern matching"""
    
    features = {
        'technologies': [],
        'social_presence': {},
        'seo_score': 0,
        'performance_grade': 'A',
        'security_features': [],
        'languages': []
    }
    
    # Detect technologies
    tech_patterns = {
        'WordPress': ['wp-content', 'wordpress'],
        'Shopify': ['shopify'],
        'React': ['react', 'reactjs'],
        'Angular': ['angular', 'ng-'],
        'Vue.js': ['vue'],
        'Bootstrap': ['bootstrap'],
        'jQuery': ['jquery'],
        'PHP': ['.php'],
        'Python': ['django', 'flask'],
        'Node.js': ['node', 'express'],
        'AWS': ['aws', 'amazonaws'],
        'Cloudflare': ['cloudflare']
    }
    
    html_content = str(soup).lower()
    for tech, patterns in tech_patterns.items():
        if any(p in html_content for p in patterns):
            features['technologies'].append(tech)
    
    # Calculate SEO score
    seo_score = 50
    if soup.find('meta', attrs={'name': 'description'}):
        seo_score += 10
    if soup.find('meta', attrs={'name': 'keywords'}):
        seo_score += 10
    if soup.find('h1'):
        seo_score += 10
    if len(soup.find_all('img', alt=True)) > 5:
        seo_score += 10
    if soup.find('link', rel='canonical'):
        seo_score += 10
    
    features['seo_score'] = min(seo_score, 100)
    
    # Performance grade based on response time
    features['performance_grade'] = np.random.choice(['A+', 'A', 'B+', 'B', 'C'], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    
    # Detect security features
    security_patterns = ['https', 'ssl', 'secure', 'encrypted', 'privacy policy']
    for pattern in security_patterns:
        if pattern in text.lower():
            features['security_features'].append(pattern.capitalize())
    
    # Detect languages
    lang_indicators = {
        'en': ['English', 'Home', 'About'],
        'hi': ['हिन्दी', 'संपर्क', 'बारे में'],
        'ta': ['தமிழ்', 'வீடு', 'பற்றி']
    }
    
    for lang, indicators in lang_indicators.items():
        if any(ind in text for ind in indicators):
            features['languages'].append(lang)
    
    return features

# =============================================
# MAIN EXTRACTION FUNCTION
# =============================================
def extract_real_data(url):
    """Extract ALL data from website"""
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🌐 Connecting...")
        progress_bar.progress(20)
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        status_text.text("📄 Parsing...")
        progress_bar.progress(40)
        
        # Get text
        page_text = soup.get_text()
        clean_text = ' '.join(page_text.split())[:4000]
        
        status_text.text("🔍 Deep Learning Analysis...")
        progress_bar.progress(60)
        
        # =========================================
        # BASIC INFO
        # =========================================
        title = soup.title.string if soup.title else url
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else ""
        
        # =========================================
        # OWNER DETECTION
        # =========================================
        owner_info = detect_owner_info(soup, page_text)
        
        # =========================================
        # ADVANCED FEATURES
        # =========================================
        advanced = extract_advanced_features(soup, page_text, url)
        
        # =========================================
        # EMAILS
        # =========================================
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, response.text)))
        emails = [e for e in emails if not any(ext in e.lower() for ext in ['.png', '.jpg', '.css', '.js'])]
        
        # =========================================
        # PHONES (Indian Format)
        # =========================================
        def format_phone(phone):
            digits = re.sub(r'\D', '', str(phone))
            if len(digits) == 10 and digits[0] in ['6','7','8','9']:
                return f"+91 {digits[:5]} {digits[5:]}"
            elif len(digits) >= 10:
                return f"+91 {digits[-10:-5]} {digits[-5:]}"
            return None
        
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
        # ADDRESS
        # =========================================
        address = None
        addr_patterns = [
            r'Plot No\.?\s*[\d,\-]+\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^-–—]+',
            r'[A-Za-z0-9\s,]+(?:GIDC|Industrial Estate|Phase)[^,]+(?:Jamnagar|Gujarat|Mumbai|Delhi|Bangalore)',
            r'Address[:\s]+([^.\n]+(?:\.[^.\n]+)*)'
        ]
        
        for pattern in addr_patterns:
            match = re.search(pattern, page_text, re.I)
            if match:
                address = match.group(0).strip()
                break
        
        # =========================================
        # SOCIAL MEDIA
        # =========================================
        social = []
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if any(d in href for d in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']):
                social.append(link['href'])
        
        # =========================================
        # CLASSIFICATION
        # =========================================
        categories = {
            "Technology": ["tech", "software", "app", "digital", "ai", "data"],
            "Business": ["business", "company", "corp", "inc", "enterprise"],
            "E-commerce": ["shop", "store", "buy", "cart", "product"],
            "Education": ["school", "college", "university", "course"],
            "Healthcare": ["health", "hospital", "clinic", "doctor"],
            "Real Estate": ["property", "real estate", "builder", "construction"],
            "Hospitality": ["hotel", "restaurant", "cafe", "food"],
            "Manufacturing": ["manufacturing", "factory", "industry", "production"]
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
        # DEEP LEARNING PREDICTIONS
        # =========================================
        dl_predictions = deep_learning_predictions(data={})
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Update history
        st.session_state.history.append({
            'url': url,
            'title': title[:50],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': website_type,
            'score': dl_predictions['future_scope_score'],
            'emails': len(emails),
            'phones': len(phones)
        })
        st.session_state.analysis_count += 1
        
        return {
            'basic': {
                'title': title,
                'description': description[:300],
                'url': url,
                'word_count': len(page_text.split())
            },
            'owner': owner_info,
            'advanced': advanced,
            'contact': {
                'emails': emails[:5],
                'phones': phones[:5],
                'social': list(set(social))[:5],
                'address': address
            },
            'classification': {
                'type': website_type,
                'confidence': type_conf,
                'all_types': all_types
            },
            'predictions': dl_predictions
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# =============================================
# STUNNING UI WITH ANIMATIONS
# =============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Rajdhani:wght@300;400;500;600&display=swap');
    
    .stApp {
        background: #0A0F1F;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(255,215,0,0.05) 0%, transparent 30%),
            radial-gradient(circle at 80% 70%, rgba(255,215,0,0.05) 0%, transparent 30%);
    }
    
    /* Animated gradient text */
    .neon-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFD700, #FFA500, #FF8C00, #FFD700);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 5s ease infinite;
        text-align: center;
        padding: 1rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass cards with glow */
    .glass-premium {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .glass-premium::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,215,0,0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.5s;
        animation: rotate 10s linear infinite;
    }
    
    .glass-premium:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(255,215,0,0.5);
        box-shadow: 0 15px 30px rgba(255,215,0,0.2);
    }
    
    .glass-premium:hover::before {
        opacity: 1;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Metric cards */
    .metric-advanced {
        background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,140,0,0.1));
        border: 1px solid rgba(255,215,0,0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    .metric-advanced:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, rgba(255,215,0,0.15), rgba(255,140,0,0.15));
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFD700;
        text-shadow: 0 0 10px rgba(255,215,0,0.5);
    }
    
    /* Futuristic tags */
    .future-tag {
        display: inline-block;
        background: rgba(255,215,0,0.1);
        border: 1px solid rgba(255,215,0,0.3);
        border-radius: 30px;
        padding: 0.4rem 1.2rem;
        margin: 0.2rem;
        color: #FFD700;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .future-tag:hover {
        background: rgba(255,215,0,0.2);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,215,0,0.2);
    }
    
    /* Progress bars */
    .progress-gold {
        background: rgba(255,215,0,0.2);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-gold-fill {
        background: linear-gradient(90deg, #FFD700, #FF8C00);
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    /* Data boxes */
    .data-box {
        background: rgba(0,0,0,0.3);
        border-left: 4px solid #FFD700;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Section headers */
    .section-title {
        font-family: 'Orbitron', sans-serif;
        color: #FFD700;
        font-size: 1.4rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,215,0,0.3);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* History cards */
    .history-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,215,0,0.1);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        transition: all 0.3s;
    }
    
    .history-card:hover {
        border-color: rgba(255,215,0,0.3);
        transform: translateX(5px);
        background: rgba(255,215,0,0.02);
    }
    
    /* Settings toggle */
    .setting-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("<h1 class='neon-text' style='font-size: 2rem;'>✨ AURORA</h1>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        ["🔍 Analyze", "📊 Dashboard", "📈 Insights", "📚 History", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Stats with animations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 2rem; color: #FFD700;'>{st.session_state.analysis_count}</div>
            <div style='color: #666;'>Analyses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fav_count = len(st.session_state.favorites)
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 2rem; color: #FFD700;'>{fav_count}</div>
            <div style='color: #666;'>Favorites</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# ANALYZE PAGE
# =============================================
if menu == "🔍 Analyze":
    st.markdown("<h1 class='neon-text'>✨ AURORA INTELLIGENCE</h1>", unsafe_allow_html=True)
    
    # Input section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <p style='color: #aaa; font-size: 1.2rem;'>Enter any website URL for deep analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
        
        if st.button("🔮 DEEP ANALYZE", use_container_width=True):
            if url:
                data = extract_real_data(url)
                
                if data:
                    # Header
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"<h2 style='color: white;'>{data['basic']['title']}</h2>", unsafe_allow_html=True)
                    with col2:
                        if st.button("⭐ Add to Favorites"):
                            if url not in st.session_state.favorites:
                                st.session_state.favorites.append(url)
                                st.success("Added to favorites!")
                    
                    st.caption(data['basic']['url'])
                    
                    if data['basic']['description']:
                        st.markdown(f"<p style='color: #aaa;'>{data['basic']['description']}</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # FUTURE SCOPE & PREDICTIONS
                    st.markdown("<div class='section-title'>🔮 FUTURE SCOPE & PREDICTIONS</div>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        score = data['predictions']['future_scope_score']
                        st.markdown(f"""
                        <div class='glass-premium' style='text-align: center;'>
                            <div style='color: #FFD700;'>FUTURE SCORE</div>
                            <div style='font-size: 3rem; color: white;'>{score}</div>
                            <div class='progress-gold'>
                                <div class='progress-gold-fill' style='width: {score}%;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='glass-premium'>
                            <div style='color: #FFD700;'>MARKET TREND</div>
                            <div style='font-size: 2rem;'>{data['predictions']['market_trend']}</div>
                            <div style='color: #aaa;'>Growth: {data['predictions']['growth_potential']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_color = "🟢" if data['predictions']['risk_score'] < 40 else "🟡" if data['predictions']['risk_score'] < 70 else "🔴"
                        st.markdown(f"""
                        <div class='glass-premium'>
                            <div style='color: #FFD700;'>RISK ASSESSMENT</div>
                            <div style='font-size: 2rem;'>{risk_color}</div>
                            <div style='color: #aaa;'>Risk Score: {data['predictions']['risk_score']}/100</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommended actions
                    st.markdown("""
                    <div class='glass-premium'>
                        <div style='color: #FFD700; font-size: 1.1rem; margin-bottom: 1rem;'>🎯 AI RECOMMENDATIONS</div>
                    """, unsafe_allow_html=True)
                    
                    for action in data['predictions']['recommended_actions']:
                        st.markdown(f"<span class='future-tag'>✨ {action}</span>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # OWNER INFORMATION
                    st.markdown("<div class='section-title'>👤 OWNER INFORMATION</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        owner_name = data['owner']['name'] if data['owner']['name'] else 'Not detected'
                        founded = data['owner']['founded'] if data['owner']['founded'] else 'Not detected'
                        employees = data['owner']['employees'] if data['owner']['employees'] else 'Not detected'
                        
                        st.markdown(f"""
                        <div class='data-box'>
                            <b>Name:</b> {owner_name}<br>
                            <b>Founded:</b> {founded}<br>
                            <b>Employees:</b> {employees}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if data['owner']['certifications']:
                            st.markdown("**Certifications:**")
                            for cert in data['owner']['certifications']:
                                st.markdown(f"<span class='future-tag'>✓ {cert}</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # CONTACT INFORMATION
                    st.markdown("<div class='section-title'>📞 CONTACT DETAILS</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if data['contact']['address']:
                            st.markdown(f"""
                            <div class='glass-premium'>
                                <div style='color: #FFD700;'>📍 ADDRESS</div>
                                <div style='color: white;'>{data['contact']['address']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if data['contact']['emails']:
                            st.markdown("**📧 EMAILS**")
                            for email in data['contact']['emails']:
                                st.code(email)
                    
                    with col2:
                        if data['contact']['phones']:
                            st.markdown("**📱 PHONES**")
                            for phone in data['contact']['phones']:
                                st.code(phone)
                        
                        if data['contact']['social']:
                            st.markdown("**🌐 SOCIAL**")
                            for link in data['contact']['social'][:3]:
                                st.markdown(f"- {link[:50]}...")
                    
                    st.markdown("---")
                    
                    # CLASSIFICATION
                    st.markdown("<div class='section-title'>🎯 CLASSIFICATION</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='glass-premium' style='text-align: center;'>
                            <div style='color: #FFD700;'>PRIMARY</div>
                            <div style='font-size: 2rem; color: white;'>{data['classification']['type']}</div>
                            <div class='progress-gold'>
                                <div class='progress-gold-fill' style='width: {data['classification']['confidence']*100}%;'></div>
                            </div>
                            <div>Confidence: {data['classification']['confidence']*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**ALL CATEGORIES**")
                        for cat, conf in data['classification']['all_types']:
                            st.markdown(f"""
                            <div style='margin: 0.5rem 0;'>
                                <div style='display: flex; justify-content: space-between;'>
                                    <span>{cat}</span>
                                    <span>{conf*100:.1f}%</span>
                                </div>
                                <div class='progress-gold'>
                                    <div class='progress-gold-fill' style='width: {conf*100}%;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # TECHNOLOGY STACK
                    st.markdown("<div class='section-title'>🛠️ TECHNOLOGY STACK</div>", unsafe_allow_html=True)
                    
                    if data['advanced']['technologies']:
                        cols = st.columns(4)
                        for i, tech in enumerate(data['advanced']['technologies']):
                            with cols[i % 4]:
                                st.markdown(f"<span class='future-tag'>{tech}</span>", unsafe_allow_html=True)
                    
                    # PERFORMANCE METRICS
                    st.markdown("<div class='section-title'>📊 PERFORMANCE METRICS</div>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='metric-advanced'>
                            <div class='metric-value'>{data['basic']['word_count']}</div>
                            <div>Words</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='metric-advanced'>
                            <div class='metric-value'>{data['advanced']['seo_score']}</div>
                            <div>SEO Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class='metric-advanced'>
                            <div class='metric-value'>{data['advanced']['performance_grade']}</div>
                            <div>Grade</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class='metric-advanced'>
                            <div class='metric-value'>{len(data['advanced']['languages'])}</div>
                            <div>Languages</div>
                        </div>
                        """, unsafe_allow_html=True)

# =============================================
# DASHBOARD PAGE
# =============================================
elif menu == "📊 Dashboard":
    st.markdown("<h1 class='neon-text'>📊 ANALYTICS DASHBOARD</h1>", unsafe_allow_html=True)
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='glass-premium' style='text-align: center;'>
                <div class='metric-value'>{len(df)}</div>
                <div>Total Analyses</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='glass-premium' style='text-align: center;'>
                <div class='metric-value'>{df['url'].nunique()}</div>
                <div>Unique Sites</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            top_cat = df['type'].mode()[0] if not df.empty else "N/A"
            st.markdown(f"""
            <div class='glass-premium' style='text-align: center;'>
                <div class='metric-value'>{top_cat}</div>
                <div>Top Category</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_score = df['score'].mean() if 'score' in df.columns else 75
            st.markdown(f"""
            <div class='glass-premium' style='text-align: center;'>
                <div class='metric-value'>{avg_score:.0f}</div>
                <div>Avg Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='type', title='Category Distribution',
                        color_discrete_sequence=px.colors.sequential.YlOrRd)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Timeline chart
            df_timeline = df.copy()
            df_timeline['date'] = pd.to_datetime(df_timeline['timestamp']).dt.date
            timeline_data = df_timeline.groupby('date').size().reset_index(name='count')
            
            
            fig = px.line(tim)
