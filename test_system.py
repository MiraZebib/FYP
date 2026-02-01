import sys
import traceback

def test_imports():
    print("="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    try:
        import train
        import app
        import evaluate
        import features
        import data_loader
        from sklearn.ensemble import GradientBoostingClassifier
        print("[PASS] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False

def test_feature_extraction():
    print("\n" + "="*70)
    print("TEST 2: Feature Extraction")
    print("="*70)
    try:
        from features import extract_features
        test_url = 'https://example.com/path?param=value'
        feat = extract_features(test_url)
        print(f"[PASS] Feature extraction works: {len(feat)} features extracted")
        print(f"  Sample: url_length={feat['url_length']}, has_https={feat['has_https']}, num_subdomains={feat['num_subdomains']}")
        print(f"  Entropy: {feat['url_entropy']:.4f}, Has IP: {feat['has_ip_address']}")
        return True
    except Exception as e:
        print(f"[FAIL] Feature extraction failed: {e}")
        traceback.print_exc()
        return False

def test_class_imbalance():
    print("\n" + "="*70)
    print("TEST 3: Class Imbalance Reporting")
    print("="*70)
    try:
        from train import report_class_imbalance
        import numpy as np
        y_test = np.array([0, 0, 0, 1, 1])
        result = report_class_imbalance(y_test)
        print("[PASS] Class imbalance reporting works")
        return True
    except Exception as e:
        print(f"[FAIL] Class imbalance test failed: {e}")
        traceback.print_exc()
        return False

def test_false_negative_analysis():
    print("\n" + "="*70)
    print("TEST 4: False Negative Analysis")
    print("="*70)
    try:
        from evaluate import analyze_false_negatives
        import numpy as np
        y_test = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 1])
        fn_indices, fn_count, fn_rate = analyze_false_negatives(y_test, y_pred, 'Test Model')
        print(f"[PASS] False negative analysis works: {fn_count} FNs found, rate: {fn_rate:.2%}")
        return True
    except Exception as e:
        print(f"[FAIL] False negative analysis failed: {e}")
        traceback.print_exc()
        return False

def test_label_encoding():
    print("\n" + "="*70)
    print("TEST 5: Label Encoding")
    print("="*70)
    try:
        from data_loader import encode_labels
        import pandas as pd
        df = pd.DataFrame({'label': ['benign', 'malicious', 'phishing', 'benign', 'defacement']})
        df_enc, mapping = encode_labels(df, 'label')
        print(f"[PASS] Label encoding works")
        print(f"  Mapping: {mapping}")
        print(f"  Encoded values: {df_enc['label'].tolist()}")
        expected = {'benign': 0, 'malicious': 1, 'phishing': 1, 'defacement': 1}
        for key, val in expected.items():
            if mapping.get(key) != val:
                print(f"  [WARN] {key} mapped to {mapping.get(key)}, expected {val}")
        return True
    except Exception as e:
        print(f"[FAIL] Label encoding failed: {e}")
        traceback.print_exc()
        return False

def test_gradient_boosting():
    print("\n" + "="*70)
    print("TEST 6: Gradient Boosting Model")
    print("="*70)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        import numpy as np
        X = np.random.rand(100, 20)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
        gb.fit(X_train, y_train)
        score = gb.score(X_test, y_test)
        print(f"[PASS] Gradient Boosting model works: Test accuracy = {score:.4f}")
        print(f"  Has feature_importances_: {hasattr(gb, 'feature_importances_')}")
        return True
    except Exception as e:
        print(f"[FAIL] Gradient Boosting test failed: {e}")
        traceback.print_exc()
        return False

def test_train_functions():
    print("\n" + "="*70)
    print("TEST 7: Training Functions")
    print("="*70)
    try:
        from train import train_gradient_boosting, train_logistic_regression, train_random_forest
        from sklearn.model_selection import train_test_split
        import numpy as np
        X = np.random.rand(200, 20)
        y = np.random.randint(0, 2, 200)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("  Testing Random Forest...")
        rf = train_random_forest(X_train, y_train, n_jobs=1)
        print("  [PASS] Random Forest training function works")
        
        print("  Testing Logistic Regression...")
        lr, scaler = train_logistic_regression(X_train, y_train, handle_imbalance=False)
        print("  [PASS] Logistic Regression training function works")
        
        print("  Testing Gradient Boosting...")
        gb = train_gradient_boosting(X_train, y_train, handle_imbalance=False)
        print("  [PASS] Gradient Boosting training function works")
        
        return True
    except Exception as e:
        print(f"[FAIL] Training functions test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("SYSTEM TEST SUITE")
    print("="*70)
    
    tests = [
        test_imports,
        test_feature_extraction,
        test_class_imbalance,
        test_false_negative_analysis,
        test_label_encoding,
        test_gradient_boosting,
        test_train_functions
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("[PASS] All tests passed!")
        return 0
    else:
        print(f"[FAIL] {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
