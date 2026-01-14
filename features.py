import re
import math
import pandas as pd
from urllib.parse import urlparse, parse_qs
from collections import Counter


def calculate_entropy(text):
    if not text:
        return 0.0
    
    text = str(text)
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * math.log2(p) for p in prob if p > 0])
    return entropy


def has_ip_address(url):
    ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ipv6_pattern = r'\[?([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\]?'
    
    if re.search(ipv4_pattern, url) or re.search(ipv6_pattern, url):
        return 1
    return 0


def count_subdomains(url):
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc or parsed.path.split('/')[0]
        
        if not hostname:
            return 0
        
        hostname = hostname.split(':')[0]
        parts = hostname.split('.')
        parts = [p for p in parts if p and p.lower() != 'www']
        
        if len(parts) >= 2:
            return max(0, len(parts) - 2)
        return 0
    except:
        return 0


def has_suspicious_tokens(url):
    suspicious_tokens = [
        'login', 'secure', 'verify', 'account', 'update', 'confirm',
        'banking', 'paypal', 'ebay', 'amazon', 'click', 'download',
        'free', 'win', 'prize', 'urgent', 'suspended', 'expired',
        'warning', 'alert', 'unusual', 'activity', 'locked', 'blocked'
    ]
    
    url_lower = url.lower()
    for token in suspicious_tokens:
        if token in url_lower:
            return 1
    return 0


def extract_features(url):
    if not url or pd.isna(url):
        url = ""
    
    url = str(url)
    
    url_length = len(url)
    
    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    num_special_chars = len(re.findall(r'[^a-zA-Z0-9]', url))
    
    num_dots = url.count('.')
    num_dashes = url.count('-')
    num_slashes = url.count('/')
    num_at_signs = url.count('@')
    num_question_marks = url.count('?')
    num_equals = url.count('=')
    num_ampersands = url.count('&')
    num_percent = url.count('%')
    
    has_https = 1 if url.lower().startswith('https://') else 0
    has_ip = has_ip_address(url)
    num_subdomains = count_subdomains(url)
    
    url_entropy = calculate_entropy(url)
    has_suspicious = has_suspicious_tokens(url)
    
    try:
        parsed = urlparse(url)
        path_length = len(parsed.path)
        query_length = len(parsed.query)
        hostname_length = len(parsed.netloc or "")
    except:
        path_length = 0
        query_length = 0
        hostname_length = 0
    
    digit_ratio = num_digits / url_length if url_length > 0 else 0
    special_char_ratio = num_special_chars / url_length if url_length > 0 else 0
    
    features = {
        'url_length': url_length,
        'num_digits': num_digits,
        'num_letters': num_letters,
        'num_special_chars': num_special_chars,
        'num_dots': num_dots,
        'num_dashes': num_dashes,
        'num_slashes': num_slashes,
        'num_at_signs': num_at_signs,
        'num_question_marks': num_question_marks,
        'num_equals': num_equals,
        'num_ampersands': num_ampersands,
        'num_percent': num_percent,
        'has_https': has_https,
        'has_ip_address': has_ip,
        'num_subdomains': num_subdomains,
        'url_entropy': url_entropy,
        'has_suspicious_tokens': has_suspicious,
        'path_length': path_length,
        'query_length': query_length,
        'hostname_length': hostname_length,
        'digit_ratio': digit_ratio,
        'special_char_ratio': special_char_ratio
    }
    
    return features


def extract_features_batch(urls):
    feature_list = []
    for url in urls:
        features = extract_features(url)
        feature_list.append(features)
    
    return pd.DataFrame(feature_list)


if __name__ == "__main__":
    test_url = "https://example.com/path/to/page?param=value"
    features = extract_features(test_url)
    print("Example features extracted:")
    for key, value in features.items():
        print(f"  {key}: {value}")
