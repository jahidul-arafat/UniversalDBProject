#!/bin/bash
# Complete AI API CLI Commands for HR Database Testing
# Make sure your web_db_explorer.py is running on http://localhost:5001
# And LM Studio is running on http://localhost:1234 with a model loaded

BASE_URL="http://localhost:5001"
LM_STUDIO_URL="http://localhost:1234/v1"

echo "🚀 HR Database AI API Testing Commands"
echo "======================================"

# 1. Test LM Studio Connection
echo "1. Testing LM Studio Connection..."
curl -X POST "${BASE_URL}/api/ai-chat/validate-connection" \
  -H "Content-Type: application/json" \
  -d '{
    "lm_studio_url": "'${LM_STUDIO_URL}'"
  }' | jq '.'

echo -e "\n"

# 2. Get Available Models
echo "2. Getting Available AI Models..."
curl -X GET "${BASE_URL}/api/ai-chat/models" | jq '.'

echo -e "\n"

# 3. Test Specific Model
echo "3. Testing DeepSeek Coder Model..."
curl -X POST "${BASE_URL}/api/ai-chat/test-model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "deepseek-coder-v2-lite-instruct",
    "lm_studio_url": "'${LM_STUDIO_URL}'"
  }' | jq '.'

echo -e "\n"

# 4. Get Database Schema for AI
echo "4. Getting Database Schema..."
curl -X GET "${BASE_URL}/api/ai-chat/schema" | jq '.'

echo -e "\n"

# 5. HR Database Query Examples
echo "5. HR Database AI Queries..."

# Employee queries
echo "5.1 Employee Information Query..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me all employees with their department names and job titles",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# Salary analysis
echo "5.2 Salary Analysis Query..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Find the average salary by department and show which departments pay above the company average",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# Manager hierarchy
echo "5.3 Manager Hierarchy Query..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me the management hierarchy - which employees are managers and how many people report to each manager",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# Location-based analysis
echo "5.4 Location Analysis Query..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many employees work in each city and country? Include the department information.",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# Job analysis
echo "5.5 Job Analysis Query..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me employees whose current salary is below the minimum salary for their job title",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# Dependent analysis
echo "5.6 Dependent Analysis Query..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which employees have dependents and what types of relationships (spouse, child, etc.)?",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# Complex analytical query
echo "5.7 Complex HR Analytics Query..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Create a comprehensive HR report showing: department headcount, average salary, salary range, and number of managers per department",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# 6. Schema Explanation Tests
echo "6. AI Schema Explanations..."

echo "6.1 Overall Schema Overview..."
curl -X POST "${BASE_URL}/api/ai-chat/explain-schema" \
  -H "Content-Type: application/json" \
  -d '{
    "focus": "overview",
    "model_id": "llama-3-groq-70b-tool-use"
  }' | jq '.'

echo -e "\n"

echo "6.2 Table Relationships..."
curl -X POST "${BASE_URL}/api/ai-chat/explain-schema" \
  -H "Content-Type: application/json" \
  -d '{
    "focus": "relationships",
    "model_id": "llama-3-groq-70b-tool-use"
  }' | jq '.'

echo -e "\n"

echo "6.3 Individual Tables..."
curl -X POST "${BASE_URL}/api/ai-chat/explain-schema" \
  -H "Content-Type: application/json" \
  -d '{
    "focus": "tables",
    "model_id": "llama-3-groq-70b-tool-use"
  }' | jq '.'

echo -e "\n"

echo "6.4 Performance Analysis..."
curl -X POST "${BASE_URL}/api/ai-chat/explain-schema" \
  -H "Content-Type: application/json" \
  -d '{
    "focus": "performance",
    "model_id": "llama-3-groq-70b-tool-use"
  }' | jq '.'

echo -e "\n"

# 7. Query Suggestions
echo "7. AI Query Suggestions..."

echo "7.1 Analytical Queries..."
curl -X POST "${BASE_URL}/api/ai-chat/suggest-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "analysis",
    "model_id": "deepseek-coder-v2-lite-instruct"
  }' | jq '.'

echo -e "\n"

echo "7.2 Reporting Queries..."
curl -X POST "${BASE_URL}/api/ai-chat/suggest-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "reporting",
    "model_id": "deepseek-coder-v2-lite-instruct"
  }' | jq '.'

echo -e "\n"

echo "7.3 Maintenance Queries..."
curl -X POST "${BASE_URL}/api/ai-chat/suggest-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "maintenance",
    "model_id": "deepseek-coder-v2-lite-instruct"
  }' | jq '.'

echo -e "\n"

# 8. Conversation Tests
echo "8. Multi-turn Conversation Test..."

echo "8.1 First message in conversation..."
curl -X POST "${BASE_URL}/api/ai-chat/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the employee table structure",
    "history": [],
    "model_id": "deepseek-coder-v2-lite-instruct"
  }' | jq '.'

echo -e "\n"

echo "8.2 Follow-up message with context..."
curl -X POST "${BASE_URL}/api/ai-chat/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Now show me how to find employees with the highest salaries",
    "history": [
      {
        "user": "Tell me about the employee table structure",
        "assistant": "The employees table contains employee information including personal details, job assignment, salary, and relationships to other tables."
      }
    ],
    "model_id": "deepseek-coder-v2-lite-instruct"
  }' | jq '.'

echo -e "\n"

# 9. HR-Specific Business Questions
echo "9. HR Business Intelligence Queries..."

echo "9.1 Workforce Distribution..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the distribution of employees across regions and countries?",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "analysis",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

echo "9.2 Salary Equity Analysis..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze salary equity - are there any employees being paid outside the defined salary ranges for their jobs?",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "analysis",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

echo "9.3 Department Efficiency..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which departments have the best manager-to-employee ratios and what are the span of control metrics?",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "analysis",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

echo "9.4 Benefits Analysis..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What percentage of employees have dependents and how does this vary by department and job level?",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "analysis",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# 10. Advanced Analytics
echo "10. Advanced HR Analytics..."

echo "10.1 Tenure Analysis..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Calculate employee tenure and identify patterns - which departments have the longest and shortest tenure employees?",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

echo "10.2 Organizational Structure..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Create a hierarchical view showing the complete organizational structure from top-level managers down to individual contributors",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

echo "10.3 Compensation Benchmarking..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare actual salaries vs job salary ranges and identify employees who might be due for salary adjustments",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": false
  }' | jq '.'

echo -e "\n"

# 11. Debug and Diagnostics
echo "11. Debug LM Studio Integration..."
curl -X POST "${BASE_URL}/api/ai-chat/debug-lm-studio" \
  -H "Content-Type: application/json" \
  -d '{
    "lm_studio_url": "'${LM_STUDIO_URL}'",
    "model_id": "deepseek-coder-v2-lite-instruct"
  }' | jq '.'

echo -e "\n"

# 12. Test with SQL Execution (use carefully!)
echo "12. Test with Automatic SQL Execution (Simple Query)..."
curl -X POST "${BASE_URL}/api/ai-chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Count how many employees we have in total",
    "model_id": "deepseek-coder-v2-lite-instruct",
    "mode": "sql",
    "execute_sql": true
  }' | jq '.'

echo -e "\n"

echo "✅ All AI API tests completed!"
echo "📊 Review the responses to verify your LM Studio integration is working correctly."
echo "🔧 If any tests fail, check:"
echo "   1. LM Studio is running on localhost:1234"
echo "   2. A model is loaded in LM Studio"
echo "   3. Your database is connected in the web application"
echo "   4. The web_db_explorer.py is running on localhost:5001"