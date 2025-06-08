#!/usr/bin/env python3
"""
Debug script to test HuggingFace Inference API directly
Run this to see what's happening with your token and models
"""

import requests
import json
import time

# Your working token
TOKEN = "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"

def test_model(model_id, token=None):
    """Test a specific model"""
    print(f"\n🧪 Testing model: {model_id}")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Simple test payload
    payload = {
        "inputs": "Generate a SQL query to select all users:",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.3,
            "return_full_text": False
        }
    }

    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        print(f"📡 URL: {url}")
        print(f"🔑 Using token: {'Yes' if token else 'No'}")

        response = requests.post(url, headers=headers, json=payload, timeout=15)

        print(f"📊 Status: {response.status_code}")
        print(f"📝 Response length: {len(response.text)}")

        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ Success! Response type: {type(result)}")
                if isinstance(result, list) and len(result) > 0:
                    print(f"📄 Generated text: {result[0].get('generated_text', 'No text found')[:100]}...")
                else:
                    print(f"📄 Raw response: {str(result)[:200]}...")
                return True
            except json.JSONDecodeError:
                print(f"📄 Plain text response: {response.text[:200]}...")
                return True
        else:
            print(f"❌ Error: {response.text[:300]}")
            return False

    except Exception as e:
        print(f"💥 Exception: {e}")
        return False

def test_available_models():
    """Test which models are actually available"""

    # Models to test (from most likely to work to least likely)
    models_to_test = [
        # Free public models
        "gpt2",
        "distilgpt2",
        "gpt2-medium",
        "microsoft/DialoGPT-medium",
        "google/flan-t5-small",
        "google/flan-t5-base",

        # Code models (may require pro)
        "codellama/CodeLlama-7b-Instruct-hf",
        "Salesforce/codegen-350M-mono",
        "microsoft/CodeGPT-small-py",

        # Chat models (may require pro)
        "microsoft/DialoGPT-large",
        "facebook/blenderbot-400M-distill",
    ]

    print("🚀 Testing HuggingFace Inference API with your token...")
    print(f"🔑 Token: {TOKEN[:20]}...")

    working_models = []
    failed_models = []

    for model in models_to_test:
        success = test_model(model, TOKEN)
        if success:
            working_models.append(model)
        else:
            failed_models.append(model)

        # Small delay to avoid rate limiting
        time.sleep(1)

    print(f"\n📊 RESULTS SUMMARY:")
    print(f"✅ Working models ({len(working_models)}):")
    for model in working_models:
        print(f"   - {model}")

    print(f"\n❌ Failed models ({len(failed_models)}):")
    for model in failed_models:
        print(f"   - {model}")

    if working_models:
        print(f"\n🎉 Great! You have {len(working_models)} working models.")
        print(f"💡 Recommended model to use: {working_models[0]}")
    else:
        print(f"\n😞 No models are working. This could mean:")
        print(f"   1. HuggingFace API is having issues")
        print(f"   2. Your token permissions need adjustment")
        print(f"   3. All models are temporarily unavailable")

    return working_models

def test_token_permissions():
    """Test what your token can access"""
    print(f"\n🔍 Testing token permissions...")

    headers = {"Authorization": f"Bearer {TOKEN}"}

    # Test 1: User info
    try:
        response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=10)
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ User info: {user_info.get('name', 'Unknown')}")
            print(f"📋 Account type: {user_info.get('type', 'unknown')}")
            if 'plan' in user_info:
                print(f"💳 Plan: {user_info['plan']}")
        else:
            print(f"❌ Cannot get user info: {response.status_code}")
    except Exception as e:
        print(f"💥 Error getting user info: {e}")

    # Test 2: List models (to check API access)
    try:
        response = requests.get("https://huggingface.co/api/models?limit=5", headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"✅ Can access models API")
        else:
            print(f"❌ Cannot access models API: {response.status_code}")
    except Exception as e:
        print(f"💥 Error accessing models API: {e}")

if __name__ == "__main__":
    print("🤖 HuggingFace Inference API Debugging Tool")
    print("=" * 50)

    # Test token permissions first
    test_token_permissions()

    # Test available models
    working_models = test_available_models()

    print(f"\n🔧 NEXT STEPS:")
    if working_models:
        print(f"1. Update your AI_CHAT_CONFIG to use: {working_models[0]}")
        print(f"2. Set requires_token: True for all models")
        print(f"3. The issue is that even 'public' models now need authentication")
    else:
        print(f"1. Check HuggingFace status: https://status.huggingface.co")
        print(f"2. Try regenerating your token with 'Inference API' permissions")
        print(f"3. Consider using the offline mode fallback")

    print(f"\n💡 TIP: Even if models fail, your app can use offline mode!")