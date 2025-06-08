#!/usr/bin/env python3
"""
LM Studio Integration Test Script
Test the AI functionality with LM Studio before full integration
"""

import requests
import json
import time

class LMStudioTester:
    def __init__(self, base_url="http://localhost:1234/v1"):
        self.base_url = base_url
        self.chat_url = f"{base_url}/chat/completions"
        self.models_url = f"{base_url}/models"

    def test_connection(self):
        """Test basic connection to LM Studio"""
        print("🔌 Testing LM Studio connection...")
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                print("✅ LM Studio is running and accessible")
                return True
            else:
                print(f"❌ LM Studio responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to LM Studio at {self.base_url}")
            print("   Make sure LM Studio is running on the correct port")
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def list_models(self):
        """List available models in LM Studio"""
        print("\n📋 Listing available models...")
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    print("✅ Available models:")
                    for model in models_data["data"]:
                        model_id = model.get("id", "unknown")
                        print(f"   - {model_id}")
                    return [model.get("id") for model in models_data["data"]]
                else:
                    print("⚠️ No models loaded in LM Studio")
                    print("   Please load a model in LM Studio before testing")
                    return []
            else:
                print(f"❌ Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []

    def test_chat_completion(self, model_id):
        """Test chat completion with a specific model"""
        print(f"\n💬 Testing chat completion with model: {model_id}")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful SQL assistant. Generate only valid SQL queries."},
                {"role": "user", "content": "Write a SQL query to select all records from a table named 'users'"}
            ],
            "temperature": 0.1,
            "max_tokens": 100,
            "stream": False
        }

        try:
            start_time = time.time()
            response = requests.post(self.chat_url, json=payload, timeout=60)
            end_time = time.time()
            response_time = end_time - start_time

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    finish_reason = result["choices"][0].get("finish_reason", "unknown")

                    print("✅ Chat completion successful!")
                    print(f"   Response time: {response_time:.2f} seconds")
                    print(f"   Model response: {content[:100]}...")
                    print(f"   Usage: {usage}")
                    print(f"   Finish reason: {finish_reason}")
                    return True
                else:
                    print(f"❌ Invalid response format: {result}")
                    return False
            elif response.status_code == 422:
                print(f"❌ Model '{model_id}' not found or not loaded")
                print("   Try loading the model in LM Studio first")
                return False
            else:
                print(f"❌ Chat completion failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
        except requests.exceptions.Timeout:
            print("❌ Request timed out (model might be slow or processing)")
            return False
        except Exception as e:
            print(f"❌ Error in chat completion: {e}")
            return False

    def test_sql_generation(self, model_id):
        """Test SQL query generation specifically"""
        print(f"\n🔍 Testing SQL generation with model: {model_id}")

        test_questions = [
            "Show me all users who registered in the last 30 days",
            "Count the number of orders by status",
            "Find users with the highest total order value"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n   Test {i}: {question}")

            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": """You are a SQL expert. Generate only valid SQLite SQL queries. 
                    
Available tables:
- users (id INTEGER, name TEXT, email TEXT, created_at DATETIME, status TEXT)
- orders (id INTEGER, user_id INTEGER, total REAL, status TEXT, created_at DATETIME)
- order_items (id INTEGER, order_id INTEGER, product_name TEXT, quantity INTEGER, price REAL)

Return only the SQL query, no explanations."""},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.1,
                "max_tokens": 150,
                "stream": False
            }

        try:
            response = requests.post(self.chat_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    sql_response = result["choices"][0]["message"]["content"].strip()
                    print(f"   ✅ Generated: {sql_response}")

                    # Check if response looks like SQL
                    if any(keyword in sql_response.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                        print("   ✅ Response contains SQL keywords")
                    else:
                        print("   ⚠️ Response doesn't look like SQL")
                else:
                    print("   ❌ Invalid response format")
            else:
                print(f"   ❌ Request failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def run_full_test(self):
    """Run complete test suite"""
    print("🚀 Starting LM Studio Integration Test")
    print("=" * 50)

    # Test 1: Connection
    if not self.test_connection():
        print("\n❌ Connection test failed. Please start LM Studio and try again.")
        return False

    # Test 2: List models
    models = self.list_models()
    if not models:
        print("\n❌ No models available. Please load a model in LM Studio.")
        return False

    # Test 3: Test first available model
    test_model = models[0]
    print(f"\n🎯 Using model for testing: {test_model}")

    if not self.test_chat_completion(test_model):
        print(f"\n❌ Chat completion failed for model: {test_model}")
        return False

    # Test 4: SQL generation
    self.test_sql_generation(test_model)

    print("\n" + "=" * 50)
    print("✅ LM Studio integration test completed successfully!")
    print("\nRecommendations:")
    print("1. The integration should work with your web application")
    print("2. Make sure to use the correct model IDs in your API calls")
    print("3. Consider the response time when setting timeouts")
    print("4. For best SQL results, use models optimized for code generation")

    return True

def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test LM Studio integration')
    parser.add_argument('--url', default='http://localhost:1234/v1',
                        help='LM Studio base URL (default: http://localhost:1234/v1)')
    parser.add_argument('--model', help='Specific model to test (optional)')

    args = parser.parse_args()

    tester = LMStudioTester(args.url)

    if args.model:
        # Test specific model
        print(f"🎯 Testing specific model: {args.model}")
        print("=" * 50)

        if tester.test_connection():
            tester.test_chat_completion(args.model)
            tester.test_sql_generation(args.model)
    else:
        # Run full test suite
        tester.run_full_test()

if __name__ == "__main__":
    main()