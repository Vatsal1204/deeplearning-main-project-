# step3_real_websites.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# REAL websites to test
real_websites = [
    {"name": "Microsoft", "url": "https://www.microsoft.com", "industry": "Technology"},
    {"name": "Tesla", "url": "https://www.tesla.com", "industry": "Automotive"},
    {"name": "Starbucks", "url": "https://www.starbucks.com", "industry": "Food & Beverage"},
]

print("🔍 Testing on REAL websites...")
print("-" * 60)

for site in real_websites:
    print(f"\n🌐 Testing: {site['name']} - {site['url']}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(site['url'], headers=headers, timeout=10)
        
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.title.string if soup.title else "No title"
            print(f"📌 Title: {title[:80]}...")
            
            # Count links
            links = soup.find_all('a', href=True)
            print(f"🔗 Links found: {len(links)}")
            
            # Check for contact page
            contact_keywords = ['contact', 'about', 'support']
            found = False
            for link in links:
                link_text = link.text.lower()
                link_href = link['href'].lower()
                for keyword in contact_keywords:
                    if keyword in link_text or keyword in link_href:
                        print(f"📞 Found '{keyword}' page: {link['href']}")
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print("📞 No contact page found in first 10 links")
        
        # Be nice to servers
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "-" * 60)
print("✅ Step 3 complete!")