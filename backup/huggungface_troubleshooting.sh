#!/bin/bash

# Debugging Hugging Face API Issues - Updated for Inference Providers

# Your token is valid! Let's debug the API calls

echo "=== Testing HF whoami API (should work now) ==="
curl -H "Authorization: Bearer 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" \
  "https://huggingface.co/api/whoami"

echo -e "\n\n=== Testing Legacy Inference API (GPT-2) ==="
# Test the old inference API (might still work for some models)
curl -v -X POST \
  -H "Authorization: Bearer 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" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello, how are you?", "parameters": {"max_new_tokens": 50}}' \
  "https://api-inference.huggingface.co/models/gpt2"

echo -e "\n\n=== Testing Inference Providers - Nebius (DeepSeek) ==="
# Test the new Inference Providers system with Nebius
curl 'https://router.huggingface.co/nebius/v1/chat/completions' \
  -H 'Authorization: Bearer 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' \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "deepseek-ai/DeepSeek-R1-0528",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "max_tokens": 50,
    "stream": false
  }'

echo -e "\n\n=== Testing Inference Providers - Sambanova ==="
# Test with Sambanova provider
curl 'https://router.huggingface.co/sambanova/v1/chat/completions' \
  -H 'Authorization: Bearer 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' \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Llama-3.3-70B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "max_tokens": 100,
    "stream": false
  }'

echo -e "\n\n=== Testing Inference Providers - Together AI ==="
# Test with Together AI provider
curl 'https://router.huggingface.co/together/v1/chat/completions' \
  -H 'Authorization: Bearer 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' \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "messages": [
      {
        "role": "user",
        "content": "Explain AI in one sentence"
      }
    ],
    "max_tokens": 100,
    "stream": false
  }'

echo -e "\n\n=== Testing without authorization (public models) ==="
# Test GPT-2 without authorization (it's a public model)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello, how are you?", "parameters": {"max_new_tokens": 50}}' \
  "https://api-inference.huggingface.co/models/gpt2"

echo -e "\n\n=== Testing our app's token validation ==="
# Test our application's token validation
curl -X 'POST' 'http://localhost:5001/api/ai-chat/validate-token' \
  -H 'Content-Type: application/json' \
  -d '{"hf_token": "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"}'

echo -e "\n\n=== Testing our app's model test (legacy API) ==="
# Test our application's model test endpoint
curl -X 'POST' 'http://localhost:5001/api/ai-chat/test-model' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "gpt2",
    "hf_token": "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"
  }'


echo -e "\n\n=== Testing our AI chat (original endpoint) ==="
# Test our AI chat endpoint
curl -X 'POST' 'http://localhost:5001/api/ai-chat/query' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer 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' \
  -d '{
    "question": "Generate a SELECT query to show all users",
    "model_id": "deepseek-ai/DeepSeek-R1-0528",
    "mode": "sql",
    "execute_sql": false,
    "hf_token": "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"
  }'

echo -e "\n\n=== Testing Python script equivalent ==="
# Test the same call your Python script makes
python3 -c "
import os
from huggingface_hub import InferenceClient

os.environ['HF_TOKEN'] = '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'

client = InferenceClient(
    provider='nebius',
    api_key=os.environ['HF_TOKEN'],
)

completion = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-R1-0528',
    messages=[
        {
            'role': 'user',
            'content': 'What is the capital of France?'
        }
    ],
)

print('Python script result:')
print(completion.choices[0].message.content)
"

echo -e "\n\n=== Debugging guide ==="
echo "Understanding the results:"
echo ""
echo "✅ WORKING - Inference Providers (Modern):"
echo "  - Nebius, Sambanova, Together AI endpoints"
echo "  - Format: router.huggingface.co/{provider}/v1/chat/completions"
echo "  - These are the RECOMMENDED approach"
echo ""
echo "❌ DEPRECATED - Legacy Inference API:"
echo "  - api-inference.huggingface.co endpoints"
echo "  - Limited model availability"
echo "  - Being phased out"
echo ""
echo "🔧 For your application:"
echo "1. Update your app to use Inference Providers instead of legacy API"
echo "2. Use the router.huggingface.co endpoints"
echo "3. Switch to chat completions format instead of 'inputs'"
echo ""
echo "📚 Available providers and models:"
echo "- Nebius: DeepSeek models, FLUX image generation"
echo "- Sambanova: Llama models"
echo "- Together AI: Various open source models"
echo "- Replicate: Image/video generation models"
echo ""
echo "🐛 Common issues:"
echo "- 'Not Found' on legacy API = Use Inference Providers instead"
echo "- 'Not allowed' errors = Check provider/model compatibility"
echo "- 503 errors = Model loading, wait and retry"
echo ""
echo "💡 Migration tips:"
echo "1. Replace api-inference.huggingface.co with router.huggingface.co"
echo "2. Change from 'inputs' to 'messages' format"
echo "3. Specify provider in URL path"
echo "4. Use chat completions structure"