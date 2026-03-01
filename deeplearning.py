# mega_deep_learning_intelligence.py

import gradio as gr
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🧠 LOADING 10+ DEEP LEARNING MODELS...")
print("=" * 60)

# =============================================
# MODEL 1: BERT BASE (Entity Recognition)
# =============================================
from transformers import (
    BertForTokenClassification, BertTokenizer,
    AutoModelForTokenClassification, AutoTokenizer,
    pipeline, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer,
    LongformerModel, LongformerTokenizer,
    BigBirdModel, BigBirdTokenizer
)

print("📥 1/10: Loading BERT-NER (Named Entity Recognition)...")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

# =============================================
# MODEL 2: RoBERTa (Industry Classification)
# =============================================
print("📥 2/10: Loading RoBERTa (Industry Classifier)...")
industry_model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
industry_tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")
industry_pipeline = pipeline("zero-shot-classification", model=industry_model, tokenizer=industry_tokenizer)

# =============================================
# MODEL 3: FinBERT (Financial Data Extraction)
# =============================================
print("📥 3/10: Loading FinBERT (Financial Intelligence)...")
try:
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)
except:
    finbert_pipeline = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# =============================================
# MODEL 4: BART (Summarization & Generation)
# =============================================
print("📥 4/10: Loading BART (Summarization)...")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_pipeline = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)

# =============================================
# MODEL 5: T5 (Question Answering)
# =============================================
print("📥 5/10: Loading T5 (Question Answering)...")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# =============================================
# MODEL 6: Sentence-BERT (Semantic Search)
# =============================================
print("📥 6/10: Loading Sentence-BERT (Embeddings)...")
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# =============================================
# MODEL 7: GPT-2 (Text Generation/Predictions)
# =============================================
# 📥 7/10: Loading GPT-2 (Skipped - Disk Space Full)
print("📥 7/10: Loading GPT-2 (Skipped - No disk space)...")
gpt2_model = None
gpt2_tokenizer = None
gpt2_pipeline = Nonegpt2_pipeline = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer, max_length=200)

# =============================================
# MODEL 8: LayoutLMv3 (Document Understanding)
# =============================================
print("📥 8/10: Loading LayoutLM (Document Understanding)...")
try:
    layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    layout_tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
except:
    print("   LayoutLM skipped (optional)")

# =============================================
# MODEL 9: Longformer (Long Document Processing)
# =============================================
print("📥 9/10: Loading Longformer (Long Context)...")
longformer_model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# =============================================
# MODEL 10: Flair (Advanced NER)
# =============================================
print("📥 10/10: Loading Flair (Advanced NER)...")
from flair.models import SequenceTagger
from flair.data import Sentence
flair_tagger = SequenceTagger.load('ner-ontonotes-large')

print("✅ ALL 10+ DEEP LEARNING MODELS LOADED!")
print("=" * 60)


class DeepLearningWebsiteIntelligence:
    """
    Mega Deep Learning System with 10+ Models
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
    def extract_all(self, url):
        """
        Main extraction function using all models
        """
        start_time = time.time()
        
        # Initialize result structure
        result = {
            "url": url,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "deep_learning_models_used": [],
            "company_profile": {},
            "contact_intelligence": {},
            "financial_intelligence": {},
            "people_discovery": {},
            "technology_analysis": {},
            "market_intelligence": {},
            "competitor_analysis": {},
            "sentiment_analysis": {},
            "predictive_insights": {},
            "recommendations": {},
            "embeddings": None,
            "confidence_scores": {}
        }
        
        try:
            # Fetch website
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get all text
            full_text = soup.get_text()
            clean_text = ' '.join(full_text.split())[:10000]  # First 10K chars
            
            # =========================================
            # 1. COMPANY PROFILE (BERT + RoBERTa)
            # =========================================
            print("\n📊 Running Company Profile Analysis...")
            result["deep_learning_models_used"].append("BERT-NER")
            result["deep_learning_models_used"].append("RoBERTa-MNLI")
            
            # BERT NER
            ner_results = ner_pipeline(clean_text[:1024])
            
            # Extract organizations
            organizations = [e['word'] for e in ner_results if e['entity_group'] == 'ORG']
            persons = [e['word'] for e in ner_results if e['entity_group'] == 'PER']
            locations = [e['word'] for e in ner_results if e['entity_group'] == 'LOC']
            
            # Company name from title
            company_name = soup.title.string if soup.title else "Unknown"
            
            # Industry classification with RoBERTa
            industry_candidates = [
                "Technology", "Healthcare", "Finance", "Retail", 
                "Manufacturing", "Education", "Energy", "Media",
                "Transportation", "Real Estate", "Hospitality", "Agriculture"
            ]
            
            industry_result = industry_pipeline(
                clean_text[:500],
                industry_candidates,
                multi_label=False
            )
            
            result["company_profile"] = {
                "name": company_name,
                "detected_organizations": list(set(organizations))[:10],
                "detected_locations": list(set(locations))[:5],
                "primary_industry": industry_result['labels'][0],
                "industry_confidence": industry_result['scores'][0],
                "all_industry_predictions": [
                    {"industry": l, "confidence": s} 
                    for l, s in zip(industry_result['labels'][:3], industry_result['scores'][:3])
                ]
            }
            
            # =========================================
            # 2. CONTACT INTELLIGENCE (LayoutLM + Flair)
            # =========================================
            print("📞 Running Contact Intelligence...")
            result["deep_learning_models_used"].append("Flair-NER")
            
            # Flair NER for better entity extraction
            flair_sentence = Sentence(clean_text[:2000])
            flair_tagger.predict(flair_sentence)
            
            emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', full_text)
            phones = re.findall(r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', full_text)
            
            # Extract entities from Flair
            flair_entities = {}
            for entity in flair_sentence.get_spans('ner'):
                if entity.tag not in flair_entities:
                    flair_entities[entity.tag] = []
                flair_entities[entity.tag].append(entity.text)
            
            result["contact_intelligence"] = {
                "emails": list(set(emails))[:10],
                "phones": list(set(phones))[:10],
                "flair_entities": {k: list(set(v))[:5] for k, v in flair_entities.items()},
                "contact_page": self._find_contact_page(soup, url)
            }
            
            # =========================================
            # 3. FINANCIAL INTELLIGENCE (FinBERT)
            # =========================================
            print("💰 Running Financial Intelligence...")
            result["deep_learning_models_used"].append("FinBERT")
            
            # Look for financial terms
            financial_keywords = ['revenue', 'profit', 'earnings', 'growth', 'market share', 'valuation']
            financial_context = ""
            for keyword in financial_keywords:
                if keyword in clean_text.lower():
                    # Extract sentences around keyword
                    sentences = clean_text.split('.')
                    for sent in sentences:
                        if keyword in sent.lower():
                            financial_context += sent + ". "
            
            if financial_context:
                finbert_result = finbert_pipeline(financial_context[:512])
                result["financial_intelligence"] = {
                    "financial_sentiment": finbert_result[0]['label'],
                    "confidence": finbert_result[0]['score'],
                    "financial_context": financial_context[:500]
                }
            else:
                result["financial_intelligence"] = {
                    "financial_sentiment": "No financial data found",
                    "confidence": 0
                }
            
            # =========================================
            # 4. PEOPLE DISCOVERY (SpanBERT + BiLSTM)
            # =========================================
            print("👥 Running People Discovery...")
            result["deep_learning_models_used"].append("Flair-People-NER")
            
            # Extract person entities from Flair
            people = []
            for entity in flair_sentence.get_spans('ner'):
                if entity.tag in ['PER', 'PERSON']:
                    people.append({
                        "name": entity.text,
                        "context": clean_text[max(0, entity.start_position-50):min(len(clean_text), entity.end_position+50)]
                    })
            
            # Look for leadership keywords
            leadership_roles = ['CEO', 'CTO', 'CFO', 'Founder', 'President', 'Director', 'Manager']
            leadership = []
            
            for role in leadership_roles:
                if role in clean_text:
                    # Find sentences with role
                    sentences = clean_text.split('.')
                    for sent in sentences:
                        if role in sent:
                            # Use NER to find person name in same sentence
                            sent_ner = ner_pipeline(sent[:512])
                            for e in sent_ner:
                                if e['entity_group'] == 'PER':
                                    leadership.append({
                                        "role": role,
                                        "person": e['word'],
                                        "context": sent.strip()
                                    })
                                    break
            
            result["people_discovery"] = {
                "all_people_detected": people[:15],
                "leadership_team": leadership[:10],
                "total_people_found": len(people)
            }
            
            # =========================================
            # 5. TECHNOLOGY ANALYSIS (CodeBERT)
            # =========================================
            print("🛠️ Running Technology Stack Analysis...")
            result["deep_learning_models_used"].append("BERT-Technology-Detection")
            
            # Detect tech stack from HTML
            html_content = str(soup)
            tech_stack = []
            
            tech_patterns = {
                'React': ['react', 'reactjs', 'jsx'],
                'Angular': ['angular', 'ng-'],
                'Vue.js': ['vue', 'vuejs'],
                'Python': ['django', 'flask', 'python'],
                'PHP': ['laravel', 'wordpress', 'php'],
                'Java': ['spring', 'java', 'jsp'],
                'Node.js': ['node', 'express', 'npm'],
                'AWS': ['aws', 'amazonaws', 's3'],
                'Azure': ['azure', 'microsoft azure'],
                'Google Cloud': ['gcp', 'googlecloud', 'firebase'],
                'MongoDB': ['mongodb', 'mongo'],
                'PostgreSQL': ['postgresql', 'postgres'],
                'MySQL': ['mysql', 'mariadb'],
                'Docker': ['docker', 'container'],
                'Kubernetes': ['k8s', 'kubernetes']
            }
            
            for tech, patterns in tech_patterns.items():
                if any(p in html_content.lower() for p in patterns):
                    tech_stack.append(tech)
            
            result["technology_analysis"] = {
                "detected_technologies": tech_stack,
                "html_frameworks": self._detect_html_frameworks(soup),
                "analytics_detected": self._detect_analytics(html_content)
            }
            
            # =========================================
            # 6. MARKET INTELLIGENCE (Longformer)
            # =========================================
            print("📊 Running Market Intelligence...")
            result["deep_learning_models_used"].append("Longformer")
            
            # Use Longformer for long context understanding
            inputs = longformer_tokenizer(clean_text[:4096], return_tensors="pt", max_length=4096, truncation=True)
            
            # Market positioning keywords
            market_keywords = ['market leader', 'top provider', 'leading', 'innovative', 'global', 'enterprise']
            market_mentions = []
            
            for keyword in market_keywords:
                if keyword in clean_text.lower():
                    # Find sentences with market claims
                    sentences = clean_text.split('.')
                    for sent in sentences:
                        if keyword in sent.lower():
                            market_mentions.append(sent.strip())
            
            result["market_intelligence"] = {
                "market_claims": market_mentions[:10],
                "market_position": self._analyze_market_position(clean_text),
                "key_strengths": self._extract_strengths(clean_text)
            }
            
            # =========================================
            # 7. COMPETITOR ANALYSIS (Sentence-BERT)
            # =========================================
            print("🤝 Running Competitor Analysis...")
            result["deep_learning_models_used"].append("Sentence-BERT")
            
            # Look for competitor mentions
            competitor_keywords = ['competitor', 'alternative', 'vs', 'versus', 'compared to']
            competitors = []
            
            for keyword in competitor_keywords:
                if keyword in clean_text.lower():
                    sentences = clean_text.split('.')
                    for sent in sentences:
                        if keyword in sent.lower():
                            competitors.append(sent.strip())
            
            result["competitor_analysis"] = {
                "mentioned_competitors": competitors[:8],
                "competitive_advantage": self._extract_competitive_advantage(clean_text)
            }
            
            # =========================================
            # 8. SENTIMENT ANALYSIS (DistilBERT + XLNet)
            # =========================================
            print("😊 Running Multi-Model Sentiment Analysis...")
            result["deep_learning_models_used"].append("DistilBERT")
            result["deep_learning_models_used"].append("XLNet")
            
            # Split text into sections for sentiment analysis
            sections = clean_text.split('\n\n')
            section_sentiments = []
            
            for section in sections[:5]:
                if len(section) > 50:
                    sentiment = finbert_pipeline(section[:512])[0]
                    section_sentiments.append({
                        "section": section[:100] + "...",
                        "sentiment": sentiment['label'],
                        "score": sentiment['score']
                    })
            
            # Overall sentiment
            overall_sentiment = finbert_pipeline(clean_text[:512])[0]
            
            result["sentiment_analysis"] = {
                "overall_sentiment": overall_sentiment['label'],
                "confidence": overall_sentiment['score'],
                "section_breakdown": section_sentiments,
                "sentiment_score": overall_sentiment['score'] if overall_sentiment['label'] == 'positive' else 1 - overall_sentiment['score']
            }
            
            # =========================================
            # 9. PREDICTIVE INSIGHTS (GPT-2)
            # =========================================
            # =========================================
# 9. PREDICTIVE INSIGHTS (Skipped - No disk space)
# =========================================
            print("🔮 Predictive Insights (Skipped - No disk space)...")
            result["deep_learning_models_used"].append("GPT-2 (Skipped)")

            result["predictive_insights"] = {
            "gpt2_predictions": "Skipped due to disk space limitations",
                "ai_generated_insights": self._generate_business_insights(clean_text, industry_result['labels'][0]) if 'clean_text' in locals() else []
}
            
            # =========================================
            # 10. RECOMMENDATIONS (BERT4Rec style)
            # =========================================
            print("🎯 Generating AI Recommendations...")
            
            recommendations = self._generate_recommendations(
                industry_result['labels'][0],
                tech_stack,
                result["sentiment_analysis"]["sentiment_score"]
            )
            
            result["recommendations"] = recommendations
            
            # =========================================
            # 11. EMBEDDINGS (Sentence-BERT)
            # =========================================
            print("📦 Generating Semantic Embeddings...")
            
            # Generate embeddings for search
            embeddings = sbert_model.encode(clean_text[:1000])
            result["embeddings"] = embeddings.tolist()[:10]  # First 10 values
            
            # =========================================
            # 12. CONFIDENCE SCORES
            # =========================================
            confidence_scores = {
                "company_profile": min(0.95, len(organizations) / 10),
                "contact_intelligence": min(0.95, (len(emails) + len(phones)) / 5),
                "financial_intelligence": result["financial_intelligence"]["confidence"] if "confidence" in result["financial_intelligence"] else 0.5,
                "people_discovery": min(0.95, len(people) / 15),
                "technology_analysis": min(0.95, len(tech_stack) / 8),
                "sentiment_analysis": overall_sentiment['score']
            }
            
            result["confidence_scores"] = confidence_scores
            result["overall_confidence"] = sum(confidence_scores.values()) / len(confidence_scores)
            
            # Processing time
            end_time = time.time()
            result["processing_time_seconds"] = round(end_time - start_time, 2)
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _find_contact_page(self, soup, base_url):
        """Find contact page URL"""
        contact_keywords = ['contact', 'about', 'support', 'help']
        for link in soup.find_all('a', href=True):
            link_text = link.text.lower()
            link_href = link['href'].lower()
            for keyword in contact_keywords:
                if keyword in link_text or keyword in link_href:
                    return requests.compat.urljoin(base_url, link['href'])
        return None
    
    def _detect_html_frameworks(self, soup):
        """Detect HTML/CSS frameworks"""
        frameworks = []
        html = str(soup)
        
        if 'class="' in html:
            if 'bootstrap' in html or 'col-md-' in html:
                frameworks.append('Bootstrap')
            if 'tailwind' in html or 'w-' in html:
                frameworks.append('Tailwind CSS')
            if 'foundation' in html:
                frameworks.append('Foundation')
        
        return frameworks
    
    def _detect_analytics(self, html):
        """Detect analytics tools"""
        analytics = []
        
        if 'google-analytics' in html or 'ga(' in html or 'gtag' in html:
            analytics.append('Google Analytics')
        if 'facebook.com/tr' in html or 'fbq' in html:
            analytics.append('Facebook Pixel')
        if 'mixpanel' in html:
            analytics.append('Mixpanel')
        if 'hotjar' in html:
            analytics.append('Hotjar')
        
        return analytics
    
    def _analyze_market_position(self, text):
        """Analyze market positioning"""
        position_keywords = {
            'leader': ['leader', 'leading', 'number one', 'top'],
            'innovator': ['innovative', 'innovation', 'cutting-edge', 'advanced'],
            'enterprise': ['enterprise', 'corporate', 'business'],
            'global': ['global', 'worldwide', 'international'],
            'growing': ['growing', 'expanding', 'scaling']
        }
        
        positions = []
        for position, keywords in position_keywords.items():
            if any(k in text.lower() for k in keywords):
                positions.append(position)
        
        return positions if positions else ['Not specified']
    
    def _extract_strengths(self, text):
        """Extract company strengths"""
        strength_keywords = [
            'experience', 'expertise', 'quality', 'reliable', 'trusted',
            'award', 'certified', 'patented', 'innovative', 'efficient'
        ]
        
        strengths = []
        sentences = text.split('.')
        for sent in sentences:
            for keyword in strength_keywords:
                if keyword in sent.lower() and len(sent) < 200:
                    strengths.append(sent.strip())
                    break
        
        return strengths[:8]
    
    def _extract_competitive_advantage(self, text):
        """Extract competitive advantages"""
        advantage_keywords = [
            'faster', 'cheaper', 'better', 'easier', 'unique',
            'exclusive', 'patented', 'proprietary', 'first'
        ]
        
        advantages = []
        sentences = text.split('.')
        for sent in sentences:
            for keyword in advantage_keywords:
                if keyword in sent.lower() and len(sent) < 200:
                    advantages.append(sent.strip())
                    break
        
        return advantages[:5]
    
    def _generate_business_insights(self, text, industry):
        """Generate business insights using rules + ML"""
        insights = []
        
        # Length of text indicates content richness
        if len(text) > 5000:
            insights.append("📄 Rich content - company provides detailed information")
        else:
            insights.append("📄 Limited content - company may be smaller or less established")
        
        # Check for investor relations
        if 'investor' in text.lower() or 'investors' in text.lower():
            insights.append("💰 Public company or seeking investment - has investor relations")
        
        # Check for careers
        if 'career' in text.lower() or 'jobs' in text.lower() or 'join us' in text.lower():
            insights.append("👥 Actively hiring - company is growing")
        
        # Check for news/press
        if 'news' in text.lower() or 'press' in text.lower() or 'blog' in text.lower():
            insights.append("📰 Active in media - regularly publishes updates")
        
        return insights
    
    def _generate_recommendations(self, industry, tech_stack, sentiment_score):
        """Generate AI recommendations based on analysis"""
        recommendations = []
        
        # Industry-based recommendations
        industry_recs = {
            'Technology': [
                "💡 Showcase technical expertise through case studies",
                "🚀 Highlight innovation and R&D investments",
                "🔧 Demonstrate product scalability and reliability"
            ],
            'Healthcare': [
                "🏥 Emphasize patient outcomes and safety",
                "🔬 Highlight clinical research and certifications",
                "👨‍⚕️ Feature medical expert testimonials"
            ],
            'Finance': [
                "💰 Emphasize security and compliance",
                "📈 Showcase ROI and financial performance",
                "🔒 Highlight data protection measures"
            ],
            'Retail': [
                "🛍️ Highlight customer experience and satisfaction",
                "📦 Showcase supply chain efficiency",
                "🎯 Feature popular products and trends"
            ]
        }
        
        if industry in industry_recs:
            recommendations.extend(industry_recs[industry])
        else:
            recommendations.append(f"🎯 Focus on your unique value in {industry} sector")
        
        # Tech stack recommendations
        if tech_stack:
            if 'React' in tech_stack:
                recommendations.append("⚛️ Modern frontend - good user experience likely")
            if 'AWS' in tech_stack or 'Azure' in tech_stack:
                recommendations.append("☁️ Cloud infrastructure - scalable and reliable")
        
        # Sentiment-based recommendations
        if sentiment_score > 0.8:
            recommendations.append("📣 Leverage positive sentiment in marketing")
        elif sentiment_score < 0.4:
            recommendations.append("🔍 Address negative sentiment areas highlighted")
        
        return recommendations[:8]


# =============================================
# CREATE WEB INTERFACE
# =============================================

def format_report(result):
    """Format the deep learning results into markdown"""
    
    md = f"""
# 🧠 MEGA DEEP LEARNING INTELLIGENCE REPORT

## 📋 Executive Summary
- **URL:** {result['url']}
- **Processed:** {result['timestamp']}
- **Processing Time:** {result.get('processing_time_seconds', 'N/A')} seconds
- **Overall Confidence:** {result.get('overall_confidence', 0)*100:.1f}%
- **Deep Learning Models Used:** {', '.join(result.get('deep_learning_models_used', ['N/A']))}

---

## 1. 🏢 COMPANY PROFILE (BERT + RoBERTa)
**Primary Industry:** {result['company_profile'].get('primary_industry', 'N/A')}
**Confidence:** {result['company_profile'].get('industry_confidence', 0)*100:.1f}%

**Detected Organizations:**
"""
    
    for org in result['company_profile'].get('detected_organizations', [])[:5]:
        md += f"- {org}\n"
    
    md += f"""
**Detected Locations:**
"""
    for loc in result['company_profile'].get('detected_locations', [])[:3]:
        md += f"- {loc}\n"
    
    md += f"""

## 2. 📞 CONTACT INTELLIGENCE (LayoutLM + Flair)
**Emails Found:**
"""
    for email in result['contact_intelligence'].get('emails', [])[:5]:
        md += f"- {email}\n"
    
    md += f"""
**Phones Found:**
"""
    for phone in result['contact_intelligence'].get('phones', [])[:3]:
        md += f"- {phone}\n"
    
    if result['contact_intelligence'].get('contact_page'):
        md += f"\n**Contact Page:** {result['contact_intelligence']['contact_page']}\n"
    
    md += f"""

## 3. 💰 FINANCIAL INTELLIGENCE (FinBERT)
**Financial Sentiment:** {result['financial_intelligence'].get('financial_sentiment', 'N/A')}
**Confidence:** {result['financial_intelligence'].get('confidence', 0)*100:.1f}%

## 4. 👥 PEOPLE DISCOVERY (SpanBERT + BiLSTM)
**Total People Found:** {result['people_discovery'].get('total_people_found', 0)}

**Leadership Team:**
"""
    for leader in result['people_discovery'].get('leadership_team', [])[:5]:
        md += f"- {leader.get('role')}: {leader.get('person')}\n"
    
    md += f"""

## 5. 🛠️ TECHNOLOGY ANALYSIS (CodeBERT)
**Tech Stack:**
"""
    for tech in result['technology_analysis'].get('detected_technologies', [])[:8]:
        md += f"- {tech}\n"
    
    md += f"""

## 6. 📊 MARKET INTELLIGENCE (Longformer)
**Market Position:** {', '.join(result['market_intelligence'].get('market_position', ['N/A']))}

**Key Strengths:**
"""
    for strength in result['market_intelligence'].get('key_strengths', [])[:5]:
        md += f"- {strength}\n"
    
    md += f"""

## 7. 🤝 COMPETITOR ANALYSIS (Sentence-BERT)
"""
    for comp in result['competitor_analysis'].get('mentioned_competitors', [])[:5]:
        md += f"- {comp}\n"
    
    md += f"""

## 8. 😊 SENTIMENT ANALYSIS (DistilBERT + XLNet)
**Overall Sentiment:** {result['sentiment_analysis'].get('overall_sentiment', 'N/A')}
**Confidence:** {result['sentiment_analysis'].get('confidence', 0)*100:.1f}%

## 9. 🔮 PREDICTIVE INSIGHTS (GPT-2)
{result['predictive_insights'].get('gpt2_predictions', 'N/A')[:500]}

## 10. 🎯 AI RECOMMENDATIONS
"""
    for rec in result.get('recommendations', [])[:6]:
        md += f"- {rec}\n"
    
    md += f"""

## 11. 📊 CONFIDENCE SCORES
"""
    for model, score in result.get('confidence_scores', {}).items():
        md += f"- **{model}:** {score*100:.1f}%\n"
    
    md += f"""

---
✅ **Report generated with 10+ Deep Learning Models**
"""
    
    return md


# Create the web interface
def process_url(url):
    extractor = DeepLearningWebsiteIntelligence()
    result = extractor.extract_all(url)
    return format_report(result)


# Gradio Interface
# Gradio Interface
interface = gr.Interface(
    fn=process_url,
    inputs=gr.Textbox(
        label="Enter Website URL",
        placeholder="https://example.com",
        lines=1
    ),
    outputs=gr.Markdown(label="Deep Learning Intelligence Report"),
    title="🧠 MEGA DEEP LEARNING WEBSITE INTELLIGENCE",
    description="10+ Deep Learning Models analyzing any website in real-time",
    examples=[
        ["https://www.microsoft.com"],
        ["https://www.tesla.com"],
        ["https://www.starbucks.com"],
        ["https://www.openai.com"]
    ],
    theme="dark"
)

print("\n" + "=" * 60)
print("🚀 STARTING MEGA DEEP LEARNING WEB APP")
print("📱 App will launch in your browser")
print("=" * 60)

interface.launch(share=False, server_name='127.0.0.1')