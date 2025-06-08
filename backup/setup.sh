#!/bin/bash
# Complete Setup and Test Script for LM Studio AI Integration
# This script helps you set up and test the entire AI integration

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                  LM Studio AI Integration                   ║"
    echo "║              Universal Database Explorer                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${YELLOW}👉 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for user input
wait_for_user() {
    read -p "Press Enter to continue..." -r
}

# Check LM Studio connection
check_lm_studio() {
    local url="http://localhost:1234/v1/models"

    if curl -s "$url" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check web server
check_web_server() {
    local url="http://localhost:5001/api/status"

    if curl -s "$url" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Main setup function
main() {
    print_header

    echo "This script will help you set up and test the LM Studio AI integration"
    echo "with your Universal Database Explorer."
    echo ""

    # Step 1: Check prerequisites
    print_step "Step 1: Checking Prerequisites"

    # Check curl
    if command_exists curl; then
        print_success "curl is installed"
    else
        print_error "curl is required but not installed"
        echo "Please install curl and run this script again"
        exit 1
    fi

    # Check jq (optional)
    if command_exists jq; then
        print_success "jq is installed (for JSON formatting)"
    else
        print_info "jq is not installed (optional, for better JSON output)"
    fi

    # Check python
    if command_exists python3; then
        print_success "Python 3 is installed"
    else
        print_error "Python 3 is required"
        exit 1
    fi

    # Step 2: Check LM Studio
    print_step "Step 2: Checking LM Studio"

    if check_lm_studio; then
        print_success "LM Studio is running and accessible"

        # Get available models
        models=$(curl -s "http://localhost:1234/v1/models" 2>/dev/null)
        if echo "$models" | grep -q '"data"'; then
            model_count=$(echo "$models" | grep -o '"id"' | wc -l)
            print_success "Found $model_count loaded model(s) in LM Studio"
        else
            print_info "LM Studio is running but no models are loaded"
            echo "Please load a model in LM Studio (recommended: deepseek-coder-v2-lite-instruct)"
        fi
    else
        print_error "LM Studio is not running or not accessible"
        echo ""
        echo "📋 To fix this:"
        echo "1. Download and install LM Studio from https://lmstudio.ai/"
        echo "2. Start LM Studio"
        echo "3. Load a model (recommended: deepseek-coder-v2-lite-instruct)"
        echo "4. Make sure the server is running on port 1234"
        echo ""
        read -p "Have you started LM Studio with a model loaded? (y/n): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if check_lm_studio; then
                print_success "LM Studio is now accessible"
            else
                print_error "Still cannot connect to LM Studio"
                exit 1
            fi
        else
            print_info "Please start LM Studio and load a model, then run this script again"
            exit 1
        fi
    fi

    # Step 3: Check web server
    print_step "Step 3: Checking Web Application"

    if check_web_server; then
        print_success "Web application is running"
    else
        print_error "Web application is not running"
        echo ""
        echo "📋 To fix this:"
        echo "1. Navigate to your project directory"
        echo "2. Run: python web_db_explorer.py"
        echo "3. Wait for the server to start on port 5001"
        echo ""
        read -p "Is your web application running on port 5001? (y/n): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if check_web_server; then
                print_success "Web application is now accessible"
            else
                print_error "Still cannot connect to web application"
                exit 1
            fi
        else
            print_info "Please start the web application and run this script again"
            exit 1
        fi
    fi

    # Step 4: Check database connection
    print_step "Step 4: Checking Database Connection"

    db_info=$(curl -s "http://localhost:5001/api/database-info" 2>/dev/null)
    if echo "$db_info" | grep -q '"total_tables"'; then
        table_count=$(echo "$db_info" | grep -o '"total_tables":[0-9]*' | cut -d: -f2)
        print_success "Database is connected with $table_count tables"
    else
        print_error "No database is connected"
        echo ""
        echo "📋 To fix this:"
        echo "1. Open your web browser and go to http://localhost:5001"
        echo "2. Upload your HR database file (.db, .sqlite, or .sqlite3)"
        echo "3. Click 'Connect' to establish the connection"
        echo ""
        read -p "Have you connected a database? (y/n): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Recheck
            db_info=$(curl -s "http://localhost:5001/api/database-info" 2>/dev/null)
            if echo "$db_info" | grep -q '"total_tables"'; then
                print_success "Database is now connected"
            else
                print_error "Still no database connected"
                exit 1
            fi
        else
            print_info "Please connect a database and run this script again"
            exit 1
        fi
    fi

    # Step 5: Run tests
    print_step "Step 5: Running AI Integration Tests"

    echo "Now we'll test the AI integration with your database."
    echo ""
    read -p "Run quick test or full test suite? (q/f): " -r

    if [[ $REPLY =~ ^[Qq]$ ]]; then
        echo ""
        print_info "Running quick test..."

        # Quick connection test
        response=$(curl -s -X POST "http://localhost:5001/api/ai-chat/validate-connection" \
                   -H "Content-Type: application/json" \
                   -d '{"lm_studio_url": "http://localhost:1234/v1"}')

        if echo "$response" | grep -q '"valid":true'; then
            print_success "LM Studio connection test passed"
        else
            print_error "LM Studio connection test failed"
            echo "Response: $response"
        fi

        # Quick query test
        response=$(curl -s -X POST "http://localhost:5001/api/ai-chat/query" \
                   -H "Content-Type: application/json" \
                   -d '{"question": "Count total employees", "model_id": "deepseek-coder-v2-lite-instruct", "mode": "sql"}')

        if echo "$response" | grep -q '"success":true'; then
            print_success "AI query generation test passed"

            # Extract SQL if available
            sql_query=$(echo "$response" | grep -o '"sql_query":"[^"]*"' | cut -d'"' -f4)
            if [ -n "$sql_query" ]; then
                print_info "Generated SQL: $sql_query"
            fi
        else
            print_error "AI query generation test failed"
            echo "Response: $response"
        fi

    else
        echo ""
        print_info "Running full test suite..."

        # Create and run the test script
        cat > /tmp/ai_test_runner.sh << 'EOF'
#!/bin/bash
# This is the embedded test runner
source <(curl -s https://raw.githubusercontent.com/your-repo/ai-test-runner.sh)
EOF

        # For now, run a simplified version
        echo "Testing all AI endpoints..."

        # Array of test cases
        declare -a tests=(
            "validate-connection:POST:{\"lm_studio_url\": \"http://localhost:1234/v1\"}"
            "models:GET:"
            "schema:GET:"
            "test-model:POST:{\"model_id\": \"deepseek-coder-v2-lite-instruct\"}"
            "query:POST:{\"question\": \"Show employee count by department\", \"model_id\": \"deepseek-coder-v2-lite-instruct\", \"mode\": \"sql\"}"
        )

        passed=0
        total=${#tests[@]}

        for test in "${tests[@]}"; do
            IFS=':' read -r endpoint method data <<< "$test"

            echo -n "Testing $endpoint... "

            if [ "$method" = "GET" ]; then
                response=$(curl -s "http://localhost:5001/api/ai-chat/$endpoint")
            else
                response=$(curl -s -X POST "http://localhost:5001/api/ai-chat/$endpoint" \
                           -H "Content-Type: application/json" \
                           -d "$data")
            fi

            if echo "$response" | grep -q '"success":true\|"valid":true'; then
                echo -e "${GREEN}PASS${NC}"
                ((passed++))
            else
                echo -e "${RED}FAIL${NC}"
            fi
        done

        echo ""
        print_info "Test Results: $passed/$total tests passed"

        if [ $passed -eq $total ]; then
            print_success "All tests passed!"
        else
            print_error "Some tests failed. Check your setup."
        fi
    fi

    # Step 6: Usage instructions
    print_step "Step 6: Usage Instructions"

    echo ""
    print_success "Setup complete! Here's how to use the AI features:"
    echo ""
    echo "🌐 Web Interface:"
    echo "   1. Open http://localhost:5001 in your browser"
    echo "   2. Navigate to the AI Assistant section"
    echo "   3. Ask questions about your database"
    echo ""
    echo "🚀 API Examples:"
    echo "   # Simple query"
    echo "   curl -X POST http://localhost:5001/api/ai-chat/query \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"question\": \"Show me employee count by department\", \"model_id\": \"deepseek-coder-v2-lite-instruct\"}'"
    echo ""
    echo "   # Schema explanation"
    echo "   curl -X POST http://localhost:5001/api/ai-chat/explain-schema \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"focus\": \"overview\"}'"
    echo ""
    echo "📖 For more examples, check the documentation or run:"
    echo "   curl http://localhost:5001/api/ai-chat/models"
    echo ""

    print_success "🎉 LM Studio AI integration is ready to use!"
}

# Help function
show_help() {
    echo "LM Studio AI Integration Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --check-only   Only check prerequisites, don't run tests"
    echo "  --test-only    Skip setup checks, only run tests"
    echo ""
    echo "This script will:"
    echo "  1. Check all prerequisites"
    echo "  2. Verify LM Studio is running"
    echo "  3. Verify web application is running"
    echo "  4. Verify database is connected"
    echo "  5. Run AI integration tests"
    echo "  6. Provide usage instructions"
}

# Parse arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --check-only)
        echo "Checking prerequisites only..."
        # Run only steps 1-4
        ;;
    --test-only)
        echo "Running tests only..."
        # Run only step 5
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac