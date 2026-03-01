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
    page_title="AURORA INTELLIGENCE",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# LOAD AI MODELS
# =============================================
@st.cache_resource(show_spinner=False)
def load_ai_models():
    """Load AI models"""
    models = {}
    
    with st.spinner("✨ Initializing AI..."):
        progress = st.progress(0)
        
        models['ner'] = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        progress.progress(50)
        
        models['classifier'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        progress.progress(100)
        
        time.sleep(0.5)
        progress.empty()
    
    return models

models = load_ai_models()

# =============================================
# ENHANCED EXTRACTION FUNCTIONS
# =============================================
def format_phone_number(phone):
    """Format phone number to Indian format"""
    # Remove all non-digits
    digits = re.sub(r'\D', '', str(phone))
    
    # Indian mobile numbers: 10 digits starting with 6-9
    if len(digits) == 10 and digits[0] in ['6','7','8','9']:
        return f"+91 {digits[:5]} {digits[5:]}"
    
    # Landline with STD code
    elif len(digits) in [11, 12] and digits.startswith('0'):
        return f"{digits[:5]} {digits[5:]}"
    
    # If it's a valid-looking number but not standard format
    elif len(digits) >= 10:
        return f"+91 {digits[-10:-5]} {digits[-5:]}"
    
    return None

def extract_complete_address(soup):
    """Extract complete address from page"""
    
    # Common address patterns in Indian websites
    address_patterns = [
        r'Plot No\.?\s*[\d,\-]+\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^-–—]+',
        r'[A-Za-z0-9\s,]+(?:GIDC|Industrial Estate|Phase)[^,]+(?:Jamnagar|Gujarat)[^,]*',
        r'Address[:\s]+([^.\n]+(?:\.[^.\n]+)*)'
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, soup.get_text(), re.I)
        if match:
            return match.group(0).strip()
    
    return None

def analyze_website(url):
    """Extract data from ANY website"""
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Use a more realistic browser header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get text content
        page_text = soup.get_text()
        clean_text = ' '.join(page_text.split())[:2000]
        
        # Basic info
        title = soup.title.string if soup.title else url
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else ""
        
        # AI Entity Recognition
        ner_results = models['ner'](clean_text[:1024])
        
        orgs = []
        people = []
        locs = []
        for e in ner_results:
            if e['entity_group'] == 'ORG':
                orgs.append(e['word'])
            elif e['entity_group'] == 'PER':
                people.append(e['word'])
            elif e['entity_group'] == 'LOC':
                locs.append(e['word'])
        
        # Classify website
        categories = ["Business", "Technology", "E-commerce", "Education", "News", 
                     "Social Media", "Entertainment", "Government", "Healthcare"]
        
        classification = models['classifier'](clean_text[:500], categories)
        
        # Extract emails
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, response.text)))
        emails = [e for e in emails if not e.endswith(('.png', '.jpg', '.css', '.js'))][:5]
        
        # Extract and format phone numbers
        phone_pattern = r'[+]?[\d\s\-\(\)]{8,20}'
        raw_phones = list(set(re.findall(phone_pattern, response.text)))
        
        phones = []
        for p in raw_phones:
            formatted = format_phone_number(p)
            if formatted and formatted not in phones:
                phones.append(formatted)
        
        # Extract complete address
        address = extract_complete_address(soup)
        if not address:
            # Try to find address in common containers
            address_elem = soup.find('span', string=re.compile(r'Plot|Address|Location', re.I))
            if address_elem:
                parent = address_elem.find_parent(['div', 'section', 'li'])
                if parent:
                    address = parent.get_text().strip()
        
        # Social links
        social = []
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if any(d in href for d in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']):
                social.append(link['href'])
        
        return {
            'title': title,
            'description': description[:200],
            'url': url,
            'word_count': len(page_text.split()),
            'orgs': list(set(orgs))[:8],
            'people': list(set(people))[:8],
            'locs': list(set(locs))[:8],
            'type': classification['labels'][0],
            'type_conf': classification['scores'][0],
            'all_types': list(zip(classification['labels'][:5], classification['scores'][:5])),
            'emails': emails,
            'phones': phones[:5],
            'social': list(set(social))[:5],
            'address': address if address else "Address not found"
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# =============================================
# STUNNING LUXURY DESIGN
# =============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp {
        background: #0A0F1F;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(255, 215, 0, 0.03) 0%, transparent 30%),
            radial-gradient(circle at 90% 50%, rgba(255, 215, 0, 0.03) 0%, transparent 40%);
    }
    
    .orb {
        position: fixed;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(255,215,0,0.15), transparent 70%);
        filter: blur(40px);
        z-index: 0;
        animation: float 20s infinite ease-in-out;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(30px, -30px) scale(1.1); }
        66% { transform: translate(-20px, 20px) scale(0.9); }
    }
    
    .hero {
        background: linear-gradient(135deg, rgba(255,215,0,0.1) 0%, rgba(255,215,0,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 32px;
        padding: 3rem 2rem;
        margin: 1rem 0 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,215,0,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
        text-align: center;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,215,0,0.1);
        border-radius: 24px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        height: 100%;
        margin: 0.5rem 0;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255,215,0,0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,215,0,0.1);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFD700;
        line-height: 1.2;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    
    .entity-tag {
        display: inline-block;
        background: rgba(255,215,0,0.1);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        margin: 0.2rem;
        color: #FFD700;
        font-size: 0.85rem;
    }
    
    .contact-item {
        background: rgba(255,215,0,0.03);
        border: 1px solid rgba(255,215,0,0.1);
        border-radius: 12px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .contact-icon {
        font-size: 1.2rem;
        background: rgba(255,215,0,0.1);
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
    }
    
    .gold-input {
        background: rgba(255,255,255,0.05);
        border: 2px solid rgba(255,215,0,0.2);
        border-radius: 50px;
        padding: 0.8rem 1.5rem;
        color: white;
        font-size: 1rem;
        width: 100%;
    }
    
    .gold-input:focus {
        border-color: #FFD700;
        outline: none;
    }
    
    .gold-button {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #0A0F1F;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .gold-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(255,215,0,0.3);
    }
    
    .gold-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,215,0,0.3), transparent);
        margin: 2rem 0;
    }
    
    .confidence-bar {
        width: 100%;
        height: 4px;
        background: rgba(255,255,255,0.1);
        border-radius: 2px;
        margin: 0.3rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        border-radius: 2px;
        transition: width 0.5s ease;
    }
    
    .address-box {
        background: rgba(255,215,0,0.05);
        border-left: 4px solid #FFD700;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        font-family: monospace;
        line-height: 1.6;
        color: white;
    }
    
    .loader {
        width: 40px;
        height: 40px;
        border: 3px solid rgba(255,215,0,0.1);
        border-top-color: #FFD700;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>

<div class='orb' style='top: -100px; left: -100px;'></div>
<div class='orb' style='bottom: -100px; right: -100px;'></div>
""", unsafe_allow_html=True)

# =============================================
# MAIN UI
# =============================================

# Hero Section
st.markdown("""
<div class='hero'>
    <div class='hero-content'>
        <div style='margin-bottom: 1rem;'>
            <span style='background: rgba(255,215,0,0.1); padding: 0.3rem 1.2rem; border-radius: 50px; color: #FFD700;'>✨ AI-POWERED</span>
        </div>
        <h1 class='hero-title'>AURORA INTELLIGENCE</h1>
        <p style='color: rgba(255,255,255,0.7);'>Analyze any website in the world with AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button("✨ ANALYZE", use_container_width=True)

# Examples
st.markdown("""
<div style='display: flex; justify-content: center; gap: 1rem; margin: 1rem 0; flex-wrap: wrap;'>
    <span class='entity-tag'>google.com</span>
    <span class='entity-tag'>github.com</span>
    <span class='entity-tag'>netflix.com</span>
    <span class='entity-tag'>harvard.edu</span>
    <span class='entity-tag'>wikipedia.org</span>
</div>
""", unsafe_allow_html=True)

# Analysis
if analyze and url:
    with st.spinner(""):
        st.markdown("<div class='loader'></div>", unsafe_allow_html=True)
        data = analyze_website(url)
        
        if data:
            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
            
            # Title
            st.markdown(f"<h1 style='font-family: Playfair Display; color: white;'>{data['title']}</h1>", unsafe_allow_html=True)
            st.caption(data['url'])
            
            if data['description']:
                st.markdown(f"<p style='color: rgba(255,255,255,0.7);'>{data['description']}</p>", unsafe_allow_html=True)
            
            st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-item'>
                    <div class='metric-value'>{data['word_count']}</div>
                    <div class='metric-label'>Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-item'>
                    <div class='metric-value'>{len(data['orgs'])}</div>
                    <div class='metric-label'>Organizations</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-item'>
                    <div class='metric-value'>{len(data['emails'])}</div>
                    <div class='metric-label'>Emails</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-item'>
                    <div class='metric-value'>{len(data['phones']) + len(data['social'])}</div>
                    <div class='metric-label'>Contacts</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
            
            # AI Classification
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='glass-card'>
                    <h3 style='color: #FFD700; margin-bottom: 1rem;'>🎯 AI Classification</h3>
                    <h2 style='color: white;'>{data['type']}</h2>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {data['type_conf'] * 100}%;'></div>
                    </div>
                    <p style='color: rgba(255,255,255,0.6);'>Confidence: {data['type_conf'] * 100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Build categories HTML without f-strings inside
                html = '<div class="glass-card"><h3 style="color: #FFD700; margin-bottom: 1rem;">📊 All Categories</h3>'
                for cat, conf in data['all_types']:
                    html += f"""
                    <div style='margin: 0.5rem 0;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <span style='color: white;'>{cat}</span>
                            <span style='color: #FFD700;'>{conf*100:.1f}%</span>
                        </div>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {conf * 100}%;'></div>
                        </div>
                    </div>
                    """
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)
            
            st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
            
            # Entities
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if data['orgs']:
                    html = '<div class="glass-card"><h3 style="color: #FFD700; margin-bottom: 1rem;">🏢 Organizations</h3>'
                    for org in data['orgs']:
                        html += f'<span class="entity-tag">{org}</span>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
            
            with col2:
                if data['people']:
                    html = '<div class="glass-card"><h3 style="color: #FFD700; margin-bottom: 1rem;">👤 People</h3>'
                    for person in data['people']:
                        html += f'<span class="entity-tag">{person}</span>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
            
            with col3:
                if data['locs']:
                    html = '<div class="glass-card"><h3 style="color: #FFD700; margin-bottom: 1rem;">📍 Locations</h3>'
                    for loc in data['locs']:
                        html += f'<span class="entity-tag">{loc}</span>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
            
            st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
            
            # Address Section
            if data['address'] and data['address'] != "Address not found":
                st.markdown(f"""
                <div class='glass-card'>
                    <h3 style='color: #FFD700; margin-bottom: 1rem;'>📍 Complete Address</h3>
                    <div class='address-box'>
                        {data['address']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
            
            # Contact Info
            col1, col2 = st.columns(2)
            
            with col1:
                if data['emails'] or data['phones']:
                    html = '<div class="glass-card"><h3 style="color: #FFD700; margin-bottom: 1rem;">📞 Contact</h3>'
                    
                    for email in data['emails']:
                        html += f"""
                        <div class='contact-item'>
                            <div class='contact-icon'>📧</div>
                            <div style='color: white;'>{email}</div>
                        </div>
                        """
                    
                    for phone in data['phones']:
                        html += f"""
                        <div class='contact-item'>
                            <div class='contact-icon'>📱</div>
                            <div style='color: white;'>{phone}</div>
                        </div>
                        """
                    
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
            
            with col2:
                if data['social']:
                    html = '<div class="glass-card"><h3 style="color: #FFD700; margin-bottom: 1rem;">🌐 Social</h3>'
                    
                    for link in data['social']:
                        display_link = link[:50] + '...' if len(link) > 50 else link
                        html += f"""
                        <div class='contact-item'>
                            <div class='contact-icon'>🔗</div>
                            <a href='{link}' target='_blank' style='color: #FFD700; text-decoration: none;'>{display_link}</a>
                        </div>
                        """
                    
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin: 3rem 0 1rem 0;'>
    <div style='color: rgba(255,215,0,0.3);'>✨ AURORA INTELLIGENCE ✨</div>
    <div style='color: rgba(255,255,255,0.2); font-size: 0.8rem;'>Powered by BERT • Works on any website</div>
</div>
""", unsafe_allow_html=True)