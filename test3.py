# step4_intelligence_extractor.py
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse

def extract_website_intelligence(url):
    """
    Extract emails, phones, and business info from any website
    """
    print(f"\n🔍 Analyzing: {url}")
    print("-" * 60)
    
    result = {
        "url": url,
        "emails": [],
        "phones": [],
        "social_links": [],
        "business_name": None,
        "description": None,
        "pages_found": []
    }
    
    try:
        # Fetch the website
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch: Status {response.status_code}")
            return result
        
        # Parse main page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get business name from title
        if soup.title:
            result["business_name"] = soup.title.string.strip()
            print(f"📌 Business: {result['business_name']}")
        
        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            result["description"] = meta_desc.get('content', '')[:150]
            print(f"📝 Description: {result['description']}...")
        
        # EXTRACT EMAILS from main page
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        found_emails = re.findall(email_pattern, response.text)
        result["emails"] = list(set([e for e in found_emails if not e.endswith('.png') and not e.endswith('.jpg')]))
        
        if result["emails"]:
            print(f"📧 Emails found: {', '.join(result['emails'][:3])}")
        
        # EXTRACT PHONES from main page
        phone_patterns = [
            r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',  # US format
            r'\+44\s?\d{2,4}\s?\d{3,4}\s?\d{3,4}',  # UK format
            r'\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'  # Generic
        ]
        
        all_phones = []
        for pattern in phone_patterns:
            found = re.findall(pattern, response.text)
            all_phones.extend(found)
        
        result["phones"] = list(set(all_phones))[:5]
        if result["phones"]:
            print(f"📞 Phones found: {', '.join(result['phones'][:2])}")
        
        # FIND social media links
        social_domains = ['twitter.com', 'linkedin.com', 'facebook.com', 'instagram.com', 'youtube.com']
        for link in soup.find_all('a', href=True):
            href = link['href']
            for domain in social_domains:
                if domain in href:
                    full_url = urljoin(url, href)
                    result["social_links"].append(full_url)
                    print(f"🔗 Social: {domain}")
                    break
        
        # Try to find and scrape contact page
        contact_keywords = ['contact', 'contact-us', 'about', 'about-us', 'support']
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        
        for keyword in contact_keywords:
            contact_url = f"{base_url}/{keyword}"
            try:
                contact_response = requests.get(contact_url, headers=headers, timeout=5)
                if contact_response.status_code == 200:
                    print(f"📄 Found contact page: {keyword}")
                    result["pages_found"].append(contact_url)
                    
                    # Extract emails from contact page
                    contact_emails = re.findall(email_pattern, contact_response.text)
                    new_emails = [e for e in contact_emails if e not in result["emails"]]
                    result["emails"].extend(new_emails[:3])
                    
                    # Extract phones from contact page
                    for pattern in phone_patterns:
                        contact_phones = re.findall(pattern, contact_response.text)
                        result["phones"].extend(contact_phones[:2])
                    
                    break
            except:
                continue
        
        # Clean up duplicates
        result["emails"] = list(set(result["emails"]))[:10]
        result["phones"] = list(set(result["phones"]))[:5]
        result["social_links"] = list(set(result["social_links"]))[:5]
        
        print("\n✅ EXTRACTION COMPLETE!")
        print(f"📊 Summary: {len(result['emails'])} emails, {len(result['phones'])} phones, {len(result['social_links'])} social links")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return result

# =============================================
# TEST IT ON REAL WEBSITES
# =============================================

# Test websites
test_sites = [
    "https://www.microsoft.com",
    "https://www.tesla.com", 
    "https://www.starbucks.com"
]

print("=" * 60)
print("🚀 WEBSITE INTELLIGENCE EXTRACTOR")
print("=" * 60)

for site in test_sites:
    result = extract_website_intelligence(site)
    print("-" * 60)
    time.sleep(2)  # Be nice to servers

print("\n✅ Step 4 complete! Your extractor is working!")