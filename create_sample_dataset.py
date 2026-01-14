"""
Sample dataset generator for testing purposes.

This script creates a sample CSV dataset with benign and malicious URLs
for testing and demonstration. The URLs are synthetic and safe to use.
"""

import pandas as pd
import random


def generate_sample_dataset(output_file='sample_dataset.csv', num_samples=1000):
    """
    Generate a sample dataset with benign and malicious URLs.
    
    Args:
        output_file (str): Output CSV file path
        num_samples (int): Number of samples to generate
    """
    benign_urls = [
        "https://www.example.com",
        "https://www.google.com/search?q=python",
        "https://github.com/user/repo",
        "https://stackoverflow.com/questions/12345",
        "https://www.wikipedia.org/wiki/Machine_learning",
        "https://www.youtube.com/watch?v=example",
        "https://www.amazon.com/product/12345",
        "https://www.microsoft.com/en-us",
        "https://www.apple.com/products",
        "https://www.netflix.com/browse",
        "https://www.reddit.com/r/programming",
        "https://www.twitter.com/user",
        "https://www.linkedin.com/in/profile",
        "https://www.instagram.com/user",
        "https://www.facebook.com/page",
        "https://docs.python.org/3/",
        "https://pypi.org/project/package/",
        "https://www.medium.com/article",
        "https://www.news.com/article/123",
        "https://www.blog.com/post-title"
    ]
    
    malicious_patterns = [
        "http://{}.com/login/verify",
        "https://{}.net/secure/update",
        "http://{}.org/account/confirm",
        "https://{}.info/banking/login",
        "http://{}.tk/click/here",
        "https://{}.ml/download/now",
        "http://{}.ga/win/prize",
        "https://{}.cf/urgent/alert",
        "http://{}.gq/suspended/account",
        "https://{}.xyz/unusual/activity"
    ]
    
    suspicious_domains = [
        "suspicious-site", "malware-domain", "phishing-link",
        "fake-bank", "scam-page", "malicious-host",
        "trojan-site", "virus-link", "fraud-page"
    ]
    
    data = []
    
    # Generate benign URLs
    num_benign = num_samples // 2
    for _ in range(num_benign):
        base_url = random.choice(benign_urls)
        # Add some variation
        if random.random() < 0.3:
            base_url += f"/path{random.randint(1, 100)}"
        if random.random() < 0.2:
            base_url += f"?param={random.randint(1, 100)}"
        data.append({
            'url': base_url,
            'label': 'benign'
        })
    
    # Generate malicious URLs
    num_malicious = num_samples - num_benign
    for _ in range(num_malicious):
        pattern = random.choice(malicious_patterns)
        domain = random.choice(suspicious_domains)
        url = pattern.format(domain)
        
        # Add some variation
        if random.random() < 0.4:
            url += f"?id={random.randint(1000, 9999)}"
        if random.random() < 0.3:
            url += f"&token={random.randint(100000, 999999)}"
        
        data.append({
            'url': url,
            'label': 'malicious'
        })
    
    # Shuffle data
    random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample dataset created: {output_file}")
    print(f"Total samples: {len(df)}")
    print(f"Benign: {len(df[df['label'] == 'benign'])}")
    print(f"Malicious: {len(df[df['label'] == 'malicious'])}")


if __name__ == "__main__":
    import sys
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'sample_dataset.csv'
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    print("Generating sample dataset...")
    generate_sample_dataset(output_file, num_samples)
    print("\nYou can now use this dataset to train models:")
    print(f"  python main.py {output_file}")
