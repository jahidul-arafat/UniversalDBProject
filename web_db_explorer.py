#!/usr/bin/env python3
"""
Web Database Explorer - Browser-based interface for Universal Database Explorer
Usage: python web_db_explorer.py
Then open http://localhost:5000 in your browser
"""

import os
import sys
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import base64
from io import BytesIO

from flask_swagger_ui import get_swaggerui_blueprint
import yaml
import json

# AI
import requests
import json
import re
from typing import List, Dict, Any, Optional
import time

# Import the core functionality from db_explorer
try:
    from db_explorer import UniversalDatabaseExplorer, DatabaseSchema, TableInfo
except ImportError:
    print("❌ Error: db_explorer.py not found. Please ensure db_explorer.py is in the same directory.")
    sys.exit(1)

# Web framework imports
try:
    from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash, Response, send_from_directory
    from werkzeug.utils import secure_filename
except ImportError:
    print("❌ Error: Flask not available. Install with: pip install flask")
    sys.exit(1)

# Data visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: matplotlib/seaborn/pandas not available. Data visualization will be limited.")
    VISUALIZATION_AVAILABLE = False


# AI Database Chat Configuration
# AI Database Chat Configuration - Updated for LM Studio
AI_CHAT_CONFIG = {
    "lm_studio": {
        "base_url": "http://localhost:1234/v1",
        "models": [
            {
                "name": "DeepSeek Coder V2 Lite",
                "model_id": "deepseek-coder-v2-lite-instruct",
                "description": "Code-focused model optimized for SQL and database tasks",
                "max_tokens": 4096,
                "temperature": 0.1,
                "specialized_for": "sql"
            },
            {
                "name": "Llama 3 Groq 70B Tool Use",
                "model_id": "llama-3-groq-70b-tool-use",
                "description": "Large model for complex analysis and general chat",
                "max_tokens": 8192,
                "temperature": 0.3,
                "specialized_for": "analysis"
            },
            {
                "name": "Custom Model",
                "model_id": "custom",
                "description": "Use any model loaded in LM Studio",
                "max_tokens": 4096,
                "temperature": 0.2,
                "specialized_for": "general"
            }
        ]
    },
    "default_model": "deepseek-coder-v2-lite-instruct",
    "api_timeout": 30,
    "max_retries": 3
}


# Create Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Global variables
db_explorer = None
current_db_path = None
current_schema = None

# Add this after creating the Flask app instance
def setup_swagger(app):
    """Setup Swagger UI for API documentation"""

    # Swagger UI configuration
    SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI
    API_URL = '/api/swagger.yaml'  # URL for the swagger spec file

    # Create Swagger UI blueprint
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "Universal Database Explorer API",
            'supportedSubmitMethods': ['get', 'post'],
            'docExpansion': 'list',
            'defaultModelsExpandDepth': 2,
            'defaultModelExpandDepth': 2,
            'displayRequestDuration': True,
            'tryItOutEnabled': True,
            'filter': True,
            'layout': 'StandaloneLayout',
            'deepLinking': True
        }
    )

    # Register blueprint
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    return app


class WebDatabaseExplorer:
    """Web interface for database exploration"""

    def __init__(self):
        self.explorer = UniversalDatabaseExplorer()
        self.connected = False
        self.db_path = None
        self.schema = None

    # def connect_database(self, db_path: str) -> Dict[str, Any]:
    #     """Connect to a database and return status"""
    #     try:
    #         if not os.path.exists(db_path):
    #             return {"success": False, "error": f"Database file '{db_path}' not found"}
    #
    #         # Determine database type
    #         db_type = 'sqlite'  # Currently only supporting SQLite
    #
    #         if self.explorer.connect(db_type, {'database': db_path}):
    #             self.connected = True
    #             self.db_path = db_path
    #             self.schema = self.explorer.schema
    #
    #             return {
    #                 "success": True,
    #                 "message": f"Successfully connected to {os.path.basename(db_path)}",
    #                 "database_info": self.get_database_info()
    #             }
    #         else:
    #             return {"success": False, "error": "Failed to connect to database"}
    #
    #     except Exception as e:
    #         return {"success": False, "error": str(e)}

    def connect_database(self, db_path: str) -> Dict[str, Any]:
        """Connect to a database and return status"""
        try:
            if not os.path.exists(db_path):
                return {"success": False, "error": f"Database file '{db_path}' not found"}

            # Determine database type
            db_type = 'sqlite'  # Currently only supporting SQLite

            if self.explorer.connect(db_type, {'database': db_path}):
                # IMPORTANT: Enable foreign key constraints in SQLite
                try:
                    self.explorer.execute_query("PRAGMA foreign_keys = ON")
                    print("✅ Foreign key constraints enabled")
                except Exception as fk_error:
                    print(f"⚠️ Could not enable foreign keys: {fk_error}")

                self.connected = True
                self.db_path = db_path
                self.schema = self.explorer.schema

                return {
                    "success": True,
                    "message": f"Successfully connected to {os.path.basename(db_path)}",
                    "database_info": self.get_database_info()
                }
            else:
                return {"success": False, "error": "Failed to connect to database"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        if not self.connected or not self.schema:
            return {}

        try:
            file_size = os.path.getsize(self.db_path) if self.db_path else 0
            size_mb = file_size / (1024 * 1024)

            return {
                "database_name": os.path.basename(self.db_path) if self.db_path else "Unknown",
                "database_type": self.schema.database_type,
                "file_size_mb": round(size_mb, 2),
                "total_tables": len(self.schema.tables),
                "total_views": len(self.schema.views),
                "total_triggers": len(self.schema.triggers),
                "total_indexes": len(self.schema.indexes),
                "total_rows": sum(table.row_count for table in self.schema.tables),
                "tables": [
                    {
                        "name": table.name,
                        "columns": len(table.columns),
                        "rows": table.row_count,
                        "primary_keys": table.primary_keys,
                        "foreign_keys": len(table.foreign_keys),
                        "indexes": len(table.indexes),
                        "triggers": len(table.triggers)
                    }
                    for table in self.schema.tables
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def get_table_structure(self, table_name: str) -> Dict[str, Any]:
        """Get detailed table structure"""
        if not self.connected:
            return {"error": "No database connected"}

        try:
            structure = self.explorer.get_table_structure(table_name)
            return {"success": True, "structure": structure} if structure else {"error": f"Table '{table_name}' not found"}
        except Exception as e:
            return {"error": str(e)}

    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query"""
        if not self.connected:
            return {"error": "No database connected"}

        try:
            results = self.explorer.execute_query(query)

            if results and 'error' in results[0]:
                return {"success": False, "error": results[0]['error']}
            elif query.upper().strip().startswith('SELECT'):
                return {
                    "success": True,
                    "type": "select",
                    "data": results,
                    "row_count": len(results),
                    "columns": list(results[0].keys()) if results else []
                }
            else:
                affected_rows = results[0].get('affected_rows', 0) if results else 0
                return {
                    "success": True,
                    "type": "modify",
                    "affected_rows": affected_rows,
                    "message": f"Query executed successfully. {affected_rows} rows affected."
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_table_data(self, table_name: str, limit: int = 100) -> Dict[str, Any]:
        """Get table data"""
        if not self.connected:
            return {"error": "No database connected"}

        try:
            data = self.explorer.get_sample_data(table_name, limit)
            return {
                "success": True,
                "data": data,
                "row_count": len(data),
                "columns": list(data[0].keys()) if data else []
            }
        except Exception as e:
            return {"error": str(e)}

    def search_tables(self, keyword: str) -> Dict[str, Any]:
        """Search tables by keyword"""
        if not self.connected:
            return {"error": "No database connected"}

        try:
            matching_tables = self.explorer.search_tables(keyword)
            return {"success": True, "tables": matching_tables}
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        """Close database connection"""
        if self.explorer:
            self.explorer.close()
        self.connected = False
        self.db_path = None
        self.schema = None
web_explorer = WebDatabaseExplorer()


# Updated DatabaseAIAssistant class for LM Studio
class DatabaseAIAssistant:
    """AI Assistant for database interactions using LM Studio local LLM"""

    def __init__(self, web_explorer, lm_studio_url: str = None):
        self.web_explorer = web_explorer
        self.base_url = lm_studio_url or AI_CHAT_CONFIG["lm_studio"]["base_url"]
        self.chat_url = f"{self.base_url}/chat/completions"
        self.schema_cache = None
        self.conversation_history = []

    def get_database_schema_context(self) -> str:
        """Get database schema information for AI context"""
        if not self.web_explorer.connected:
            return "No database connected."

        try:
            # Get table names
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            tables_result = self.web_explorer.execute_query(tables_query)

            schema_info = []
            schema_info.append("=== DATABASE SCHEMA ===")

            if tables_result:
                table_names = []
                if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                    table_names = [row['name'] for row in tables_result['data']]
                elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                    table_names = [row['name'] for row in tables_result if 'name' in row]

                for table_name in table_names:
                    try:
                        # Get table structure
                        structure_query = f"PRAGMA table_info({table_name})"
                        structure_result = self.web_explorer.execute_query(structure_query)

                        # Get row count
                        count_query = f"SELECT COUNT(*) as count FROM `{table_name}`"
                        count_result = self.web_explorer.execute_query(count_query)

                        row_count = 0
                        if count_result:
                            if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                                row_count = count_result['data'][0].get('count', 0)
                            elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                                row_count = count_result[0].get('count', 0)

                        schema_info.append(f"\nTable: {table_name} ({row_count} rows)")

                        if structure_result:
                            columns = []
                            if isinstance(structure_result, dict) and structure_result.get('success') and structure_result.get('data'):
                                columns = structure_result['data']
                            elif isinstance(structure_result, list) and not (len(structure_result) > 0 and structure_result[0].get('error')):
                                columns = structure_result

                            for col in columns:
                                col_name = col.get('name', '')
                                col_type = col.get('type', '')
                                not_null = " NOT NULL" if col.get('notnull', 0) else ""
                                pk = " PRIMARY KEY" if col.get('pk', 0) else ""
                                schema_info.append(f"  - {col_name}: {col_type}{not_null}{pk}")

                        # Get foreign keys
                        fk_query = f"PRAGMA foreign_key_list({table_name})"
                        fk_result = self.web_explorer.execute_query(fk_query)

                        if fk_result:
                            foreign_keys = []
                            if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                                foreign_keys = fk_result['data']
                            elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                                foreign_keys = fk_result

                            for fk in foreign_keys:
                                from_col = fk.get('from', '')
                                to_table = fk.get('table', '')
                                to_col = fk.get('to', '')
                                schema_info.append(f"  FK: {from_col} -> {to_table}({to_col})")

                    except Exception as table_error:
                        schema_info.append(f"  Error analyzing table {table_name}: {str(table_error)}")
                        continue

            return "\n".join(schema_info)

        except Exception as e:
            return f"Error getting schema: {str(e)}"

    def create_ai_prompt(self, user_question: str, model_type: str = "sql") -> str:
        """Create AI prompt with database context"""
        schema = self.get_database_schema_context()

        if model_type == "sql":
            system_message = f"""You are a SQL expert assistant helping users query a SQLite database.

{schema}

INSTRUCTIONS:
1. Generate ONLY valid SQLite SQL queries
2. Use proper table and column names from the schema above
3. For complex questions, break them down into clear SQL
4. Always use proper JOIN syntax when relating tables
5. Include LIMIT clauses for large result sets
6. Return only the SQL query, no explanations unless asked
7. Use backticks around table and column names if they contain spaces or special characters"""

            user_message = f"Generate a SQL query for: {user_question}"

        elif model_type == "analysis":
            system_message = f"""You are a database analyst helping users understand their data.

{schema}

INSTRUCTIONS:
1. Analyze the user's question about the database
2. Suggest appropriate SQL queries or approaches
3. Explain what insights can be gained
4. Recommend visualizations if applicable
5. Be conversational and helpful"""

            user_message = user_question

        else:  # general chat
            system_message = f"""You are a helpful database assistant. You have access to this database schema:

{schema}

Provide helpful responses about the database. If users want data analysis, suggest SQL queries. 
If they want to understand the database structure, explain it clearly."""

            user_message = user_question

        return system_message, user_message

    def call_lm_studio_model(self, system_message: str, user_message: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """Call LM Studio local LLM API"""

        # Find model config
        model_config = None
        for model in AI_CHAT_CONFIG["lm_studio"]["models"]:
            if model["model_id"] == model_id:
                model_config = model
                break

        # Use model-specific parameters or defaults
        max_tokens = model_config.get("max_tokens", 4096) if model_config else kwargs.get("max_tokens", 4096)
        temperature = model_config.get("temperature", 0.2) if model_config else kwargs.get("temperature", 0.2)

        # Create payload for LM Studio chat completions API
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        headers = {"Content-Type": "application/json"}

        print(f"Calling LM Studio API: {self.chat_url}")
        print(f"Model: {model_id}, Temperature: {temperature}, Max tokens: {max_tokens}")

        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                timeout=AI_CHAT_CONFIG["api_timeout"]
            )

            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"Response format: {type(result)}")

                    # Parse LM Studio response format
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            generated_text = choice["message"]["content"].strip()

                            return {
                                "success": True,
                                "response": generated_text,
                                "model": model_id,
                                "usage": result.get("usage", {}),
                                "finish_reason": choice.get("finish_reason", "unknown")
                            }

                    return {
                        "success": False,
                        "error": f"Unexpected response format: {result}",
                        "model": model_id
                    }

                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"JSON decode error: {e}. Response: {response.text[:200]}",
                        "model": model_id
                    }

            elif response.status_code == 404:
                return {
                    "success": False,
                    "error": f"LM Studio server not found. Is LM Studio running on {self.base_url}?",
                    "model": model_id
                }
            elif response.status_code == 422:
                return {
                    "success": False,
                    "error": f"Model '{model_id}' not loaded in LM Studio. Please load the model first.",
                    "model": model_id
                }
            else:
                return {
                    "success": False,
                    "error": f"API call failed with status {response.status_code}: {response.text[:200]}",
                    "model": model_id
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Cannot connect to LM Studio at {self.base_url}. Please ensure LM Studio is running.",
                "model": model_id
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. The model might be processing a complex query.",
                "model": model_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "model": model_id
            }

    def extract_sql_from_response(self, ai_response: str) -> Optional[str]:
        """Extract SQL query from AI response"""
        # Look for SQL patterns
        sql_patterns = [
            r"```sql\s*(.*?)\s*```",
            r"```\s*(SELECT.*?)```",
            r"(SELECT.*?;)",
            r"(INSERT.*?;)",
            r"(UPDATE.*?;)",
            r"(DELETE.*?;)",
            r"(CREATE.*?;)",
            r"(DROP.*?;)"
        ]

        for pattern in sql_patterns:
            matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
            if matches:
                sql = matches[0].strip()
                # Clean up the SQL
                sql = re.sub(r'\s+', ' ', sql)  # Normalize whitespace
                return sql

        # If no pattern matches, look for SQL keywords at the start
        lines = ai_response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP')):
                return line

        return None



def create_visualization(data: List[Dict[str, Any]], chart_type: str,
                         x_column: str, y_column: str, title: str = "Chart") -> str:
    """Create visualization and return base64 encoded image"""
    if not VISUALIZATION_AVAILABLE or not data:
        return None

    try:
        # Create DataFrame
        df = pd.DataFrame(data)

        # Limit data for performance
        if len(df) > 100:
            df = df.head(100)

        # Create figure
        plt.figure(figsize=(12, 8))

        if chart_type == "bar":
            plt.bar(df[x_column].astype(str), pd.to_numeric(df[y_column], errors='coerce'))
            plt.xticks(rotation=45)
        elif chart_type == "line":
            plt.plot(df[x_column], pd.to_numeric(df[y_column], errors='coerce'))
        elif chart_type == "pie":
            plt.pie(pd.to_numeric(df[y_column], errors='coerce'), labels=df[x_column], autopct='%1.1f%%')
        elif chart_type == "histogram":
            plt.hist(pd.to_numeric(df[y_column], errors='coerce'), bins=20)
            plt.xlabel(y_column)
            plt.ylabel('Frequency')
        elif chart_type == "scatter":
            plt.scatter(pd.to_numeric(df[x_column], errors='coerce'),
                        pd.to_numeric(df[y_column], errors='coerce'))
            plt.xlabel(x_column)
            plt.ylabel(y_column)

        plt.title(title)
        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None



# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/connect', methods=['POST'])
def connect_database():
    """Connect to database"""
    if 'database_file' not in request.files:
        return jsonify({"success": False, "error": "No database file provided"})

    file = request.files['database_file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})

    if file:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Connect to database
        result = web_explorer.connect_database(filepath)

        if result["success"]:
            session['connected'] = True
            session['db_path'] = filepath
            session['db_name'] = filename

        return jsonify(result)

@app.route('/api/database-info')
def get_database_info():
    """Get database information"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    return jsonify(web_explorer.get_database_info())

# Add a route to serve static assets if needed
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(static_dir, filename)

@app.route('/api/tables')
def get_tables():
    """Get list of tables"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        tables = [
            {
                "name": table.name,
                "rows": table.row_count,
                "columns": len(table.columns)
            }
            for table in web_explorer.schema.tables
        ]
        return jsonify({"success": True, "tables": tables})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/table/<table_name>/structure')
def get_table_structure(table_name):
    """Get table structure"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print(f"Getting structure for table: {table_name}")  # Debug log

        # Method 1: First check if table exists
        table_exists_query = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
        exists_result = web_explorer.execute_query(table_exists_query.replace('?', f"'{table_name}'"))

        table_exists = False
        if exists_result:
            if isinstance(exists_result, dict) and exists_result.get('success') and exists_result.get('data'):
                table_exists = len(exists_result['data']) > 0
            elif isinstance(exists_result, list) and not (len(exists_result) > 0 and exists_result[0].get('error')):
                table_exists = len(exists_result) > 0

        if not table_exists:
            return jsonify({"error": f"Table '{table_name}' not found"})

        print(f"Table {table_name} exists, getting structure...")  # Debug log

        # Method 2: Get table structure using PRAGMA table_info
        pragma_query = f"PRAGMA table_info({table_name})"  # Remove backticks for PRAGMA
        print(f"Executing PRAGMA query: {pragma_query}")  # Debug log
        columns_result = web_explorer.execute_query(pragma_query)

        print(f"PRAGMA columns result: {columns_result}")  # Debug log

        columns = []
        if columns_result:
            raw_columns = []
            if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                raw_columns = columns_result['data']
            elif isinstance(columns_result, list):
                # Handle case where result is a list directly
                if len(columns_result) > 0 and not columns_result[0].get('error'):
                    raw_columns = columns_result

            print(f"Raw columns data: {raw_columns}")  # Debug log

            # Convert PRAGMA table_info format to our format
            for col in raw_columns:
                print(f"Processing column: {col}")  # Debug log
                column_info = {
                    "name": col.get('name', ''),
                    "type": col.get('type', ''),
                    "nullable": col.get('notnull', 0) == 0,  # notnull=0 means nullable=True
                    "default": col.get('dflt_value'),
                    "primary_key": col.get('pk', 0) == 1
                }
                columns.append(column_info)

        # Fallback: If PRAGMA didn't work, try alternative method
        if not columns:
            print("PRAGMA failed, trying alternative method...")  # Debug log
            # Try getting column info from sqlite_master
            schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            schema_result = web_explorer.execute_query(schema_query)

            if schema_result:
                schema_data = []
                if isinstance(schema_result, dict) and schema_result.get('success') and schema_result.get('data'):
                    schema_data = schema_result['data']
                elif isinstance(schema_result, list) and not (len(schema_result) > 0 and schema_result[0].get('error')):
                    schema_data = schema_result

                if schema_data and len(schema_data) > 0:
                    create_sql = schema_data[0].get('sql', '')
                    print(f"Create SQL: {create_sql}")  # Debug log

                    # Basic parsing of CREATE TABLE statement
                    if create_sql:
                        import re
                        # Extract column definitions from CREATE TABLE statement
                        columns_match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
                        if columns_match:
                            columns_text = columns_match.group(1)
                            column_lines = [line.strip() for line in columns_text.split(',')]

                            for line in column_lines:
                                if line and not line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK')):
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        col_name = parts[0].strip('`"\'')
                                        col_type = parts[1]

                                        column_info = {
                                            "name": col_name,
                                            "type": col_type,
                                            "nullable": 'NOT NULL' not in line.upper(),
                                            "default": None,
                                            "primary_key": 'PRIMARY KEY' in line.upper()
                                        }
                                        columns.append(column_info)

        # Method 3: Get primary keys
        primary_keys = [col['name'] for col in columns if col.get('primary_key')]

        # Method 4: Get foreign keys using PRAGMA foreign_key_list
        fk_query = f"PRAGMA foreign_key_list({table_name})"  # Remove backticks
        print(f"Executing FK query: {fk_query}")  # Debug log
        fk_result = web_explorer.execute_query(fk_query)

        print(f"FK result: {fk_result}")  # Debug log

        foreign_keys = []
        if fk_result:
            raw_fks = []
            if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                raw_fks = fk_result['data']
            elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                raw_fks = fk_result

            for fk in raw_fks:
                foreign_key_info = {
                    "column": fk.get('from', ''),
                    "referenced_table": fk.get('table', ''),
                    "referenced_column": fk.get('to', '')
                }
                foreign_keys.append(foreign_key_info)

        # Method 5: Get indexes using PRAGMA index_list
        idx_query = f"PRAGMA index_list({table_name})"  # Remove backticks
        print(f"Executing index query: {idx_query}")  # Debug log
        idx_result = web_explorer.execute_query(idx_query)

        print(f"Index result: {idx_result}")  # Debug log

        indexes = []
        if idx_result:
            raw_indexes = []
            if isinstance(idx_result, dict) and idx_result.get('success') and idx_result.get('data'):
                raw_indexes = idx_result['data']
            elif isinstance(idx_result, list) and not (len(idx_result) > 0 and idx_result[0].get('error')):
                raw_indexes = idx_result

            for idx in raw_indexes:
                # Get index details
                idx_name = idx.get('name', '')
                if idx_name and not idx_name.startswith('sqlite_'):  # Skip auto-generated indexes
                    idx_info_query = f"PRAGMA index_info(`{idx_name}`)"
                    idx_info_result = web_explorer.execute_query(idx_info_query)

                    index_columns = []
                    if idx_info_result:
                        raw_idx_info = []
                        if isinstance(idx_info_result, dict) and idx_info_result.get('success') and idx_info_result.get('data'):
                            raw_idx_info = idx_info_result['data']
                        elif isinstance(idx_info_result, list) and not (len(idx_info_result) > 0 and idx_info_result[0].get('error')):
                            raw_idx_info = idx_info_result

                        index_columns = [col_info.get('name', '') for col_info in raw_idx_info]

                    index_info = {
                        "name": idx_name,
                        "unique": idx.get('unique', 0) == 1,
                        "columns": index_columns
                    }
                    indexes.append(index_info)

        # Method 6: Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"  # Remove backticks
        print(f"Executing count query: {count_query}")  # Debug log
        count_result = web_explorer.execute_query(count_query)

        print(f"Count result: {count_result}")  # Debug log

        row_count = 0
        if count_result:
            if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                row_count = count_result['data'][0].get('row_count', 0)
            elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                row_count = count_result[0].get('row_count', 0)

        # Method 7: Get triggers
        trigger_query = f"SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name = '{table_name}'"
        trigger_result = web_explorer.execute_query(trigger_query)

        triggers = []
        if trigger_result:
            if isinstance(trigger_result, dict) and trigger_result.get('success') and trigger_result.get('data'):
                triggers = [row['name'] for row in trigger_result['data']]
            elif isinstance(trigger_result, list) and not (len(trigger_result) > 0 and trigger_result[0].get('error')):
                triggers = [row['name'] for row in trigger_result if 'name' in row]

        # Build the complete structure response
        structure = {
            "table_name": table_name,
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "indexes": indexes,
            "triggers": triggers,
            "row_count": row_count,
            "column_count": len(columns)
        }

        print(f"Table structure for {table_name}: {structure}")  # Debug log

        return jsonify({
            "success": True,
            "structure": structure
        })

    except Exception as e:
        print(f"Error getting table structure for {table_name}: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/table/<table_name>/data')
def get_table_data(table_name):
    """Get table data"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        limit = request.args.get('limit', 100, type=int)

        print(f"Getting data for table: {table_name}, limit: {limit}")  # Debug log

        # Use the original method from web_explorer if it exists
        if hasattr(web_explorer, 'get_table_data') and callable(getattr(web_explorer, 'get_table_data')):
            try:
                result = web_explorer.get_table_data(table_name, limit)
                print(f"Original method result: {result}")  # Debug log
                return jsonify(result)
            except Exception as e:
                print(f"Original method failed: {e}")  # Debug log
                # Fall back to custom implementation

        # Fallback: Try to get data directly with SELECT
        data_query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        print(f"Executing fallback data query: {data_query}")  # Debug log

        data_result = web_explorer.execute_query(data_query)
        print(f"Fallback data result: {data_result}")  # Debug log

        # Handle different response formats
        if data_result:
            if isinstance(data_result, dict):
                if data_result.get('success'):
                    data = data_result.get('data', [])
                    columns = data_result.get('columns', [])

                    # If columns not provided, extract from first row
                    if not columns and data and len(data) > 0:
                        columns = list(data[0].keys())

                    return jsonify({
                        "success": True,
                        "data": data,
                        "row_count": len(data),
                        "columns": columns
                    })
                else:
                    return jsonify({
                        "error": data_result.get('error', 'Query failed')
                    })
            elif isinstance(data_result, list):
                if len(data_result) > 0 and data_result[0].get('error'):
                    return jsonify({
                        "error": data_result[0]['error']
                    })
                else:
                    # Extract columns from first row if data exists
                    columns = []
                    if len(data_result) > 0:
                        columns = list(data_result[0].keys())

                    return jsonify({
                        "success": True,
                        "data": data_result,
                        "row_count": len(data_result),
                        "columns": columns
                    })

        # If we get here, the query returned no results
        # Try to check if table exists by querying sqlite_master
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = web_explorer.execute_query(check_query)

        table_exists = False
        if check_result:
            if isinstance(check_result, dict) and check_result.get('success') and check_result.get('data'):
                table_exists = len(check_result['data']) > 0
            elif isinstance(check_result, list) and not (len(check_result) > 0 and check_result[0].get('error')):
                table_exists = len(check_result) > 0

        if not table_exists:
            return jsonify({"error": f"Table '{table_name}' not found"})

        # Table exists but has no data - try to get column info using alternative methods
        columns = []

        # Method 1: Try using a SELECT with LIMIT 0 to get column structure
        column_query = f"SELECT * FROM `{table_name}` LIMIT 0"
        print(f"Executing column detection query: {column_query}")  # Debug log
        column_result = web_explorer.execute_query(column_query)
        print(f"Column detection result: {column_result}")  # Debug log

        if column_result:
            if isinstance(column_result, dict) and column_result.get('success'):
                columns = column_result.get('columns', [])
            elif isinstance(column_result, list):
                # For empty result, try to get columns from first row keys (should be empty but may have structure)
                if len(column_result) > 0 and not column_result[0].get('error'):
                    columns = list(column_result[0].keys())

        # Method 2: If SELECT failed, try getting CREATE TABLE statement and parse it
        if not columns:
            print("SELECT method failed, trying CREATE TABLE method...")  # Debug log
            create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            create_result = web_explorer.execute_query(create_query)
            print(f"CREATE TABLE result: {create_result}")  # Debug log

            if create_result:
                create_sql = ""
                if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                    if len(create_result['data']) > 0:
                        create_sql = create_result['data'][0].get('sql', '')
                elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                    if len(create_result) > 0:
                        create_sql = create_result[0].get('sql', '')

                print(f"CREATE SQL: {create_sql}")  # Debug log

                # Parse column names from CREATE TABLE statement
                if create_sql:
                    import re
                    # Extract content between parentheses
                    match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
                    if match:
                        columns_text = match.group(1)
                        # Split by comma and extract column names
                        column_lines = [line.strip() for line in columns_text.split(',')]
                        for line in column_lines:
                            if line and not line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT')):
                                # Extract column name (first word, remove quotes)
                                parts = line.split()
                                if len(parts) >= 1:
                                    col_name = parts[0].strip('`"\'')
                                    if col_name and col_name.upper() not in ('PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT'):
                                        columns.append(col_name)

        print(f"Final columns for empty table: {columns}")  # Debug log

        return jsonify({
            "success": True,
            "data": [],
            "row_count": 0,
            "columns": columns,
            "message": f"Table '{table_name}' exists but contains no data"
        })

    except Exception as e:
        print(f"Error getting table data for {table_name}: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/execute-query', methods=['POST'])
def execute_query():
    """Execute SQL query"""
    if not web_explorer.connected:
        return jsonify({"success": False, "error": "No database connected"})

    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"success": False, "error": "No query provided"})

    try:
        print(f"Executing query: {query}")  # Debug log

        # Execute the query
        result = web_explorer.execute_query(query)

        print(f"Raw query result: {result}")  # Debug log

        if not result:
            return jsonify({
                "success": False,
                "error": "No response from database"
            })

        # Handle different response formats
        if isinstance(result, dict):
            # Already formatted response from newer methods
            if 'success' in result:
                return jsonify(result)
            else:
                # Legacy format - convert to standard format
                if 'error' in result:
                    return jsonify({
                        "success": False,
                        "error": result.get('error')
                    })
                else:
                    # Assume success if no error
                    return jsonify({
                        "success": True,
                        "data": result,
                        "row_count": 1,
                        "columns": list(result.keys()) if result else []
                    })

        elif isinstance(result, list):
            # Check if it's an error response
            if len(result) > 0 and isinstance(result[0], dict) and result[0].get('error'):
                return jsonify({
                    "success": False,
                    "error": result[0]['error']
                })

            # Determine query type based on the query string
            query_upper = query.upper().strip()

            # Check if it's a SELECT-like query (returns data)
            is_select_query = (
                    query_upper.startswith('SELECT') or
                    query_upper.startswith('PRAGMA') or
                    query_upper.startswith('EXPLAIN') or
                    'RETURNING' in query_upper
            )

            if is_select_query:
                # Return as SELECT result
                columns = []
                if len(result) > 0 and isinstance(result[0], dict):
                    columns = list(result[0].keys())

                return jsonify({
                    "success": True,
                    "type": "select",
                    "data": result,
                    "row_count": len(result),
                    "columns": columns
                })
            else:
                # Return as modification result
                affected_rows = len(result) if result else 0

                # For DDL statements, affected_rows doesn't make sense
                if any(cmd in query_upper for cmd in ['CREATE', 'DROP', 'ALTER']):
                    return jsonify({
                        "success": True,
                        "type": "ddl",
                        "message": "DDL statement executed successfully",
                        "query_type": "schema_modification"
                    })
                else:
                    return jsonify({
                        "success": True,
                        "type": "modify",
                        "affected_rows": affected_rows,
                        "message": f"Query executed successfully. {affected_rows} rows affected."
                    })

        else:
            # Unexpected result format
            return jsonify({
                "success": True,
                "data": result,
                "message": "Query executed successfully",
                "note": "Unexpected result format"
            })

    except Exception as e:
        print(f"Exception in execute_query: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/search-tables')
def search_tables():
    """Search tables by keyword"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    keyword = request.args.get('keyword', '').strip()
    if not keyword:
        return jsonify({"error": "No search keyword provided"})

    try:
        # Method 1: Search using SQLite system tables
        search_query = """
                       SELECT name FROM sqlite_master
                       WHERE type='table'
                         AND name LIKE ?
                       ORDER BY name \
                       """

        print(f"Searching for tables with keyword: {keyword}")  # Debug log

        # Execute search query directly
        result = web_explorer.execute_query(search_query.replace('?', f"'%{keyword}%'"))

        matching_tables = []
        if result:
            if isinstance(result, dict) and result.get('success') and result.get('data'):
                matching_tables = [row['name'] for row in result['data']]
            elif isinstance(result, list) and not (len(result) > 0 and result[0].get('error')):
                matching_tables = [row['name'] for row in result if 'name' in row]

        print(f"Found tables: {matching_tables}")  # Debug log

        # Method 2: Also search in column names if no table name matches
        if not matching_tables:
            print("No table names matched, searching column names...")  # Debug log

            # Get all tables and search their columns
            all_tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables_result = web_explorer.execute_query(all_tables_query)

            all_tables = []
            if tables_result:
                if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                    all_tables = [row['name'] for row in tables_result['data']]
                elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                    all_tables = [row['name'] for row in tables_result if 'name' in row]

            print(f"All tables in database: {all_tables}")  # Debug log

            # Search columns in each table
            for table_name in all_tables:
                try:
                    # Get table structure to search column names
                    pragma_query = f"PRAGMA table_info(`{table_name}`)"
                    columns_result = web_explorer.execute_query(pragma_query)

                    if columns_result:
                        columns = []
                        if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                            columns = columns_result['data']
                        elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                            columns = columns_result

                        # Check if any column contains the keyword
                        for col_info in columns:
                            col_name = col_info.get('name', '')
                            if keyword.lower() in col_name.lower():
                                if table_name not in matching_tables:
                                    matching_tables.append(table_name)
                                break

                except Exception as col_error:
                    print(f"Error searching columns in table {table_name}: {col_error}")
                    continue

        # Method 3: If still no results, try exact table name match
        if not matching_tables:
            print("Trying exact table name search...")  # Debug log
            exact_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name = '{keyword}'"
            exact_result = web_explorer.execute_query(exact_query)

            if exact_result:
                if isinstance(exact_result, dict) and exact_result.get('success') and exact_result.get('data'):
                    matching_tables = [row['name'] for row in exact_result['data']]
                elif isinstance(exact_result, list) and not (len(exact_result) > 0 and exact_result[0].get('error')):
                    matching_tables = [row['name'] for row in exact_result if 'name' in row]

        print(f"Final matching tables: {matching_tables}")  # Debug log

        return jsonify({
            "success": True,
            "tables": matching_tables,
            "search_keyword": keyword,
            "total_matches": len(matching_tables)
        })

    except Exception as e:
        print(f"Error in search_tables: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/visualize', methods=['POST'])
def create_chart():
    """Create data visualization"""
    if not VISUALIZATION_AVAILABLE:
        return jsonify({"error": "Visualization not available. Install matplotlib, seaborn, and pandas."})

    data = request.get_json()
    chart_data = data.get('data', [])
    chart_type = data.get('chart_type', 'bar')
    x_column = data.get('x_column')
    y_column = data.get('y_column')
    title = data.get('title', 'Chart')

    if not chart_data or not x_column or not y_column:
        return jsonify({"error": "Invalid chart parameters"})

    image_data = create_visualization(chart_data, chart_type, x_column, y_column, title)

    if image_data:
        return jsonify({"success": True, "image": image_data})
    else:
        return jsonify({"error": "Failed to create visualization"})

@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    """Export data to CSV"""
    data = request.get_json()
    table_data = data.get('data', [])
    filename = data.get('filename', 'export.csv')

    if not table_data:
        return jsonify({"error": "No data to export"})

    try:
        # Create CSV content
        if table_data:
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=table_data[0].keys())
            writer.writeheader()
            writer.writerows(table_data)

            csv_content = output.getvalue()

            # Save to temp file
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)

            return send_file(temp_path, as_attachment=True, download_name=filename)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/validate-data')
def validate_data():
    """Run data validation"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Starting data validation...")  # Debug log
        validation_results = []

        # Get all tables from database directly
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        print(f"Found tables: {table_names}")  # Debug log

        if not table_names:
            validation_results.append({
                "type": "warning",
                "title": "No Tables Found",
                "message": "No user tables found in the database"
            })
            return jsonify({"success": True, "results": validation_results})

        # Check for empty tables using real-time row counts
        empty_tables = []
        table_stats = {}

        for table_name in table_names:
            try:
                count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
                count_result = web_explorer.execute_query(count_query)

                row_count = 0
                if count_result:
                    if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                        row_count = count_result['data'][0].get('row_count', 0)
                    elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                        row_count = count_result[0].get('row_count', 0)

                table_stats[table_name] = row_count

                if row_count == 0:
                    empty_tables.append(table_name)

                print(f"Table {table_name}: {row_count} rows")  # Debug log

            except Exception as table_error:
                print(f"Error checking table {table_name}: {table_error}")
                continue

        print(f"Empty tables: {empty_tables}")  # Debug log
        print(f"Table stats: {table_stats}")  # Debug log

        # Report empty tables (excluding system tables)
        if empty_tables:
            user_empty_tables = [t for t in empty_tables if not t.startswith('sqlite_')]
            if user_empty_tables:
                validation_results.append({
                    "type": "warning",
                    "title": "Empty Tables",
                    "message": f"Found {len(user_empty_tables)} empty tables: {', '.join(user_empty_tables)}"
                })

        # Check referential integrity for tables with foreign keys
        integrity_issues = []

        for table_name in table_names:
            try:
                # Get foreign key information
                fk_query = f"PRAGMA foreign_key_list({table_name})"
                fk_result = web_explorer.execute_query(fk_query)

                foreign_keys = []
                if fk_result:
                    if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                        foreign_keys = fk_result['data']
                    elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                        foreign_keys = fk_result

                # Check each foreign key constraint
                for fk in foreign_keys:
                    try:
                        from_col = fk.get('from', '')
                        to_table = fk.get('table', '')
                        to_col = fk.get('to', '')

                        if all([from_col, to_table, to_col]):
                            # Check for orphaned records
                            orphan_query = f"""
                                SELECT COUNT(*) as orphaned_count
                                FROM `{table_name}` t1
                                LEFT JOIN `{to_table}` t2 ON t1.`{from_col}` = t2.`{to_col}`
                                WHERE t1.`{from_col}` IS NOT NULL 
                                AND t2.`{to_col}` IS NULL
                            """

                            orphan_result = web_explorer.execute_query(orphan_query)
                            orphaned_count = 0

                            if orphan_result:
                                if isinstance(orphan_result, dict) and orphan_result.get('success') and orphan_result.get('data'):
                                    orphaned_count = orphan_result['data'][0].get('orphaned_count', 0)
                                elif isinstance(orphan_result, list) and not (len(orphan_result) > 0 and orphan_result[0].get('error')):
                                    orphaned_count = orphan_result[0].get('orphaned_count', 0)

                            if orphaned_count > 0:
                                integrity_issues.append({
                                    "table": table_name,
                                    "foreign_key": f"{from_col} -> {to_table}.{to_col}",
                                    "orphaned_records": orphaned_count
                                })
                    except Exception as fk_check_error:
                        print(f"Error checking FK {table_name}.{fk}: {fk_check_error}")
                        continue

            except Exception as table_fk_error:
                print(f"Error getting FK info for {table_name}: {table_fk_error}")
                continue

        if integrity_issues:
            issue_details = "; ".join([f"{issue['table']}: {issue['orphaned_records']} orphaned records in {issue['foreign_key']}" for issue in integrity_issues])
            validation_results.append({
                "type": "error",
                "title": "Referential Integrity Issues",
                "message": f"Found {len(integrity_issues)} referential integrity violations: {issue_details}"
            })

        # Check indexing recommendations for large tables
        unindexed_large_tables = []
        large_table_threshold = 1000  # Consider tables with 1000+ rows as "large"

        for table_name, row_count in table_stats.items():
            if row_count > large_table_threshold:
                try:
                    # Check if table has indexes
                    index_query = f"PRAGMA index_list({table_name})"
                    index_result = web_explorer.execute_query(index_query)

                    indexes = []
                    if index_result:
                        if isinstance(index_result, dict) and index_result.get('success') and index_result.get('data'):
                            indexes = index_result['data']
                        elif isinstance(index_result, list) and not (len(index_result) > 0 and index_result[0].get('error')):
                            indexes = index_result

                    # Filter out auto-generated indexes
                    user_indexes = [idx for idx in indexes if not idx.get('name', '').startswith('sqlite_')]

                    if len(user_indexes) == 0:
                        unindexed_large_tables.append(f"{table_name} ({row_count:,} rows)")

                except Exception as index_error:
                    print(f"Error checking indexes for {table_name}: {index_error}")
                    continue

        if unindexed_large_tables:
            validation_results.append({
                "type": "info",
                "title": "Indexing Recommendations",
                "message": f"Consider adding indexes to large tables: {', '.join(unindexed_large_tables)}"
            })

        # Check for tables with potential data quality issues
        data_quality_issues = []

        for table_name, row_count in table_stats.items():
            if row_count > 0:  # Only check tables with data
                try:
                    # Get column information
                    columns_query = f"PRAGMA table_info({table_name})"
                    columns_result = web_explorer.execute_query(columns_query)

                    columns = []
                    if columns_result:
                        if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                            columns = columns_result['data']
                        elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                            columns = columns_result

                    # Check for columns with high NULL percentage
                    for col in columns:
                        col_name = col.get('name', '')
                        if col_name and not col.get('notnull', 0):  # Only check nullable columns
                            try:
                                null_query = f"SELECT COUNT(*) as null_count FROM `{table_name}` WHERE `{col_name}` IS NULL"
                                null_result = web_explorer.execute_query(null_query)

                                null_count = 0
                                if null_result:
                                    if isinstance(null_result, dict) and null_result.get('success') and null_result.get('data'):
                                        null_count = null_result['data'][0].get('null_count', 0)
                                    elif isinstance(null_result, list) and not (len(null_result) > 0 and null_result[0].get('error')):
                                        null_count = null_result[0].get('null_count', 0)

                                null_percentage = (null_count / row_count * 100) if row_count > 0 else 0

                                if null_percentage > 50:  # More than 50% NULLs
                                    data_quality_issues.append(f"{table_name}.{col_name} ({null_percentage:.1f}% NULL)")

                            except Exception as null_check_error:
                                print(f"Error checking NULLs in {table_name}.{col_name}: {null_check_error}")
                                continue

                except Exception as dq_error:
                    print(f"Error checking data quality for {table_name}: {dq_error}")
                    continue

        if data_quality_issues:
            validation_results.append({
                "type": "warning",
                "title": "Data Quality Issues",
                "message": f"Columns with high NULL percentages: {', '.join(data_quality_issues[:5])}" + (f" and {len(data_quality_issues)-5} more" if len(data_quality_issues) > 5 else "")
            })

        # Add summary information
        total_tables = len(table_names)
        total_rows = sum(table_stats.values())

        validation_results.append({
            "type": "info",
            "title": "Database Summary",
            "message": f"Validated {total_tables} tables with {total_rows:,} total rows"
        })

        # If no issues found, add success message
        if len([r for r in validation_results if r['type'] in ['error', 'warning']]) == 0:
            validation_results.insert(0, {
                "type": "success",
                "title": "Validation Complete",
                "message": "No critical issues found in database validation"
            })

        print(f"Validation complete. Found {len(validation_results)} results")  # Debug log

        return jsonify({"success": True, "results": validation_results})

    except Exception as e:
        print(f"Error in validate_data: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/performance-analysis')
def performance_analysis():
    """Get performance analysis"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Starting performance analysis...")  # Debug log

        analysis = {
            "table_sizes": [],
            "index_analysis": {},
            "recommendations": [],
            "query_performance": {},
            "storage_analysis": {},
            "constraint_analysis": {}
        }

        # Get all user tables (excluding system tables)
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        print(f"Analyzing tables: {table_names}")  # Debug log

        if not table_names:
            return jsonify({
                "success": False,
                "error": "No user tables found for analysis"
            })

        # Analyze each table with real-time data
        total_indexes = 0
        tables_with_indexes = 0
        total_rows_all_tables = 0

        for table_name in table_names:
            try:
                # Get real-time row count
                count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
                count_result = web_explorer.execute_query(count_query)

                row_count = 0
                if count_result:
                    if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                        row_count = count_result['data'][0].get('row_count', 0)
                    elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                        row_count = count_result[0].get('row_count', 0)

                # Get column count using alternative method
                column_count = 0
                try:
                    # Method 1: Try PRAGMA
                    columns_query = f"PRAGMA table_info({table_name})"
                    columns_result = web_explorer.execute_query(columns_query)

                    if columns_result:
                        if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                            column_count = len(columns_result['data'])
                        elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                            column_count = len(columns_result)

                    # Method 2: Fallback - use SELECT * LIMIT 0 to get column names
                    if column_count == 0:
                        schema_query = f"SELECT * FROM `{table_name}` LIMIT 0"
                        schema_result = web_explorer.execute_query(schema_query)

                        if schema_result:
                            if isinstance(schema_result, dict) and schema_result.get('success'):
                                column_count = len(schema_result.get('columns', []))
                            elif isinstance(schema_result, list):
                                if len(schema_result) > 0:
                                    column_count = len(schema_result[0].keys())

                    # Method 3: Parse from CREATE TABLE statement
                    if column_count == 0:
                        create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                        create_result = web_explorer.execute_query(create_query)

                        if create_result:
                            create_sql = ""
                            if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                                if len(create_result['data']) > 0:
                                    create_sql = create_result['data'][0].get('sql', '')
                            elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                                if len(create_result) > 0:
                                    create_sql = create_result[0].get('sql', '')

                            if create_sql:
                                import re
                                # Count column definitions in CREATE statement
                                match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
                                if match:
                                    columns_text = match.group(1)
                                    column_lines = [line.strip() for line in columns_text.split(',')]
                                    for line in column_lines:
                                        if line and not line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT')):
                                            parts = line.split()
                                            if len(parts) >= 1:
                                                col_name = parts[0].strip('`"\'')
                                                if col_name and col_name.upper() not in ('PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT'):
                                                    column_count += 1
                except Exception as col_error:
                    print(f"Error getting column count for {table_name}: {col_error}")
                    column_count = 1  # Fallback to prevent division by zero

                # Get index information using alternative methods
                index_count = 0
                try:
                    # Method 1: Try PRAGMA index_list
                    indexes_query = f"PRAGMA index_list({table_name})"
                    indexes_result = web_explorer.execute_query(indexes_query)

                    if indexes_result:
                        indexes = []
                        if isinstance(indexes_result, dict) and indexes_result.get('success') and indexes_result.get('data'):
                            indexes = indexes_result['data']
                        elif isinstance(indexes_result, list) and not (len(indexes_result) > 0 and indexes_result[0].get('error')):
                            indexes = indexes_result

                        # Filter out auto-generated indexes
                        user_indexes = [idx for idx in indexes if not idx.get('name', '').startswith('sqlite_autoindex')]
                        index_count = len(user_indexes)

                    # Method 2: Fallback - query sqlite_master for indexes
                    if index_count == 0:
                        index_master_query = f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}' AND name NOT LIKE 'sqlite_%'"
                        index_master_result = web_explorer.execute_query(index_master_query)

                        if index_master_result:
                            if isinstance(index_master_result, dict) and index_master_result.get('success') and index_master_result.get('data'):
                                index_count = len(index_master_result['data'])
                            elif isinstance(index_master_result, list) and not (len(index_master_result) > 0 and index_master_result[0].get('error')):
                                index_count = len(index_master_result)

                except Exception as idx_error:
                    print(f"Error getting index count for {table_name}: {idx_error}")
                    index_count = 0

                # Ensure we have valid values to prevent division by zero
                if column_count == 0:
                    column_count = 1  # Fallback

                # Calculate estimated size (rough estimation)
                avg_bytes_per_field = 50  # Conservative estimate
                estimated_size_bytes = row_count * column_count * avg_bytes_per_field
                estimated_size_mb = estimated_size_bytes / (1024 * 1024)

                # Add index overhead (rough estimate)
                index_overhead_mb = index_count * 0.1  # 0.1MB per index (rough estimate)
                total_estimated_size_mb = estimated_size_mb + index_overhead_mb

                table_info = {
                    "name": table_name,
                    "rows": row_count,
                    "columns": column_count,
                    "indexes": index_count,
                    "estimated_size_mb": round(total_estimated_size_mb, 3),
                    "data_size_mb": round(estimated_size_mb, 3),
                    "index_size_mb": round(index_overhead_mb, 3),
                    "density": round(row_count / max(column_count, 1), 2)  # rows per column ratio
                }

                analysis["table_sizes"].append(table_info)

                # Update totals
                total_indexes += index_count
                total_rows_all_tables += row_count
                if index_count > 0:
                    tables_with_indexes += 1

                print(f"Table {table_name}: {row_count} rows, {column_count} columns, {index_count} indexes")  # Debug log

            except Exception as table_error:
                print(f"Error analyzing table {table_name}: {table_error}")
                continue

        # Sort tables by row count (largest first)
        analysis["table_sizes"].sort(key=lambda x: x["rows"], reverse=True)

        # Index analysis
        total_tables = len(table_names)
        tables_without_indexes = total_tables - tables_with_indexes
        avg_indexes_per_table = round(total_indexes / total_tables, 2) if total_tables > 0 else 0

        analysis["index_analysis"] = {
            "total_indexes": total_indexes,
            "tables_with_indexes": tables_with_indexes,
            "tables_without_indexes": tables_without_indexes,
            "avg_indexes_per_table": avg_indexes_per_table,
            "index_coverage_percentage": round((tables_with_indexes / total_tables * 100), 1) if total_tables > 0 else 0
        }

        # Storage analysis
        total_estimated_size = sum(table["estimated_size_mb"] for table in analysis["table_sizes"])
        largest_table = analysis["table_sizes"][0] if analysis["table_sizes"] else None

        analysis["storage_analysis"] = {
            "total_estimated_size_mb": round(total_estimated_size, 3),
            "total_rows": total_rows_all_tables,
            "average_table_size_mb": round(total_estimated_size / total_tables, 3) if total_tables > 0 else 0,
            "largest_table": {
                "name": largest_table["name"] if largest_table else None,
                "size_mb": largest_table["estimated_size_mb"] if largest_table else 0,
                "rows": largest_table["rows"] if largest_table else 0
            } if largest_table else None
        }

        # Generate recommendations
        recommendations = []

        # Recommend indexes for large tables without them
        for table in analysis["table_sizes"]:
            if table["rows"] > 1000 and table["indexes"] == 0:
                recommendations.append(f"Add indexes to large table: {table['name']} ({table['rows']:,} rows)")
            elif table["rows"] > 500 and table["indexes"] == 0:
                recommendations.append(f"Consider adding indexes to table: {table['name']} ({table['rows']:,} rows)")

        # Recommend indexes for foreign key columns
        for table_name in table_names:
            try:
                # Get foreign key information using alternative method if PRAGMA fails
                fk_query = f"PRAGMA foreign_key_list({table_name})"
                fk_result = web_explorer.execute_query(fk_query)

                foreign_keys = []
                if fk_result:
                    if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                        foreign_keys = fk_result['data']
                    elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                        foreign_keys = fk_result

                # If PRAGMA didn't work, try to find FKs from CREATE TABLE statement
                if not foreign_keys:
                    create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                    create_result = web_explorer.execute_query(create_query)

                    if create_result:
                        create_sql = ""
                        if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                            if len(create_result['data']) > 0:
                                create_sql = create_result['data'][0].get('sql', '')
                        elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                            if len(create_result) > 0:
                                create_sql = create_result[0].get('sql', '')

                        # Look for REFERENCES keyword in CREATE statement
                        if 'REFERENCES' in create_sql.upper():
                            import re
                            # Simple FK detection - look for column_name REFERENCES table(column)
                            fk_pattern = r'(\w+)\s+[^,]*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)'
                            matches = re.findall(fk_pattern, create_sql, re.IGNORECASE)
                            for match in matches:
                                foreign_keys.append({
                                    'from': match[0],
                                    'table': match[1],
                                    'to': match[2]
                                })

                # Get existing indexes for this table
                existing_indexed_columns = set()
                try:
                    indexes_query = f"PRAGMA index_list({table_name})"
                    indexes_result = web_explorer.execute_query(indexes_query)

                    if indexes_result:
                        indexes = []
                        if isinstance(indexes_result, dict) and indexes_result.get('success') and indexes_result.get('data'):
                            indexes = indexes_result['data']
                        elif isinstance(indexes_result, list) and not (len(indexes_result) > 0 and indexes_result[0].get('error')):
                            indexes = indexes_result

                        # Get columns for each index
                        for idx in indexes:
                            idx_name = idx.get('name', '')
                            if idx_name and not idx_name.startswith('sqlite_'):
                                try:
                                    idx_info_query = f"PRAGMA index_info(`{idx_name}`)"
                                    idx_info_result = web_explorer.execute_query(idx_info_query)

                                    if idx_info_result:
                                        idx_columns = []
                                        if isinstance(idx_info_result, dict) and idx_info_result.get('success') and idx_info_result.get('data'):
                                            idx_columns = [col.get('name', '') for col in idx_info_result['data']]
                                        elif isinstance(idx_info_result, list) and not (len(idx_info_result) > 0 and idx_info_result[0].get('error')):
                                            idx_columns = [col.get('name', '') for col in idx_info_result]

                                        existing_indexed_columns.update(idx_columns)
                                except:
                                    continue
                except:
                    pass

                # Check if foreign key columns are indexed
                for fk in foreign_keys:
                    fk_column = fk.get('from', '')
                    if fk_column and fk_column not in existing_indexed_columns:
                        recommendations.append(f"Consider indexing foreign key column: {table_name}.{fk_column}")

            except Exception as fk_error:
                print(f"Error checking foreign keys for {table_name}: {fk_error}")
                continue

        # Recommend optimization for tables with poor density
        for table in analysis["table_sizes"]:
            if table["rows"] > 100 and table["density"] < 5:  # Less than 5 rows per column
                recommendations.append(f"Table {table['name']} has low data density ({table['density']:.1f} rows/column) - consider normalization")

        # Recommend specific optimizations for audit tables
        audit_tables = [table for table in analysis["table_sizes"] if 'audit' in table['name'].lower() or 'log' in table['name'].lower()]
        for table in audit_tables:
            if table["indexes"] < 2:
                recommendations.append(f"Audit table {table['name']} should have indexes on date and foreign key columns for optimal query performance")

        # Check for tables that might benefit from partitioning (very large tables)
        large_tables = [table for table in analysis["table_sizes"] if table["rows"] > 10000]
        if large_tables:
            for table in large_tables:
                recommendations.append(f"Large table {table['name']} ({table['rows']:,} rows) may benefit from archiving old data or partitioning")

        # Performance recommendations based on index coverage
        if analysis["index_analysis"]["index_coverage_percentage"] < 50:
            recommendations.append(f"Low index coverage ({analysis['index_analysis']['index_coverage_percentage']:.1f}%) - most tables lack indexes which may impact query performance")
        elif analysis["index_analysis"]["index_coverage_percentage"] < 80:
            recommendations.append(f"Moderate index coverage ({analysis['index_analysis']['index_coverage_percentage']:.1f}%) - some tables could benefit from additional indexes")

        # Storage optimization recommendations
        if total_estimated_size > 50:  # Database larger than 50MB
            recommendations.append("Consider implementing data archival strategy for large database")

        # Specific recommendations for small but important tables
        important_small_tables = [table for table in analysis["table_sizes"] if table["rows"] < 100 and table["indexes"] == 0 and table["rows"] > 0]
        if len(important_small_tables) > 0:
            table_names_str = ", ".join([t["name"] for t in important_small_tables[:3]])
            if len(important_small_tables) > 3:
                table_names_str += f" and {len(important_small_tables)-3} others"
            recommendations.append(f"Small reference tables ({table_names_str}) may benefit from indexes if frequently queried")

        # Add maintenance recommendations
        if total_rows_all_tables > 1000:
            recommendations.append("Consider regular VACUUM and ANALYZE operations to maintain optimal performance")

        if len(analysis["table_sizes"]) > 10:
            recommendations.append("Large schema detected - consider database documentation and query optimization guidelines")

        # Debugging: Check if salary_audit_log was processed
        salary_audit_table = next((table for table in analysis["table_sizes"] if table["name"] == "salary_audit_log"), None)
        if not salary_audit_table:
            print("Warning: salary_audit_log not found in analysis results")  # Debug log
            # Try to get info about salary_audit_log specifically
            try:
                count_query = "SELECT COUNT(*) as row_count FROM salary_audit_log"
                count_result = web_explorer.execute_query(count_query)
                if count_result:
                    print(f"salary_audit_log query result: {count_result}")  # Debug log
            except Exception as debug_error:
                print(f"Error querying salary_audit_log: {debug_error}")  # Debug log

        analysis["recommendations"] = recommendations

        # Query performance analysis (basic)
        analysis["query_performance"] = {
            "estimated_full_scan_cost": total_rows_all_tables,  # Cost of scanning all tables
            "indexed_tables_percentage": round((tables_with_indexes / total_tables * 100), 1) if total_tables > 0 else 0,
            "optimization_potential": "High" if tables_without_indexes > 2 else "Medium" if tables_without_indexes > 0 else "Low"
        }

        print(f"Performance analysis complete. Analyzed {total_tables} tables with {total_rows_all_tables:,} total rows")  # Debug log

        return jsonify({"success": True, "analysis": analysis})

    except Exception as e:
        print(f"Error in performance_analysis: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

# HTML Templates (inline for simplicity)
def create_templates():
    """Create template files with both main frontend and relationship visualization"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')

    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # File paths
    frontend_file_path = os.path.join(os.path.dirname(__file__), 'frontend_artifact.html')
    relationship_file_path = os.path.join(os.path.dirname(__file__), 'relationship.html')

    # Create main index template
    try:
        with open(frontend_file_path, 'r', encoding='utf-8') as f:
            frontend_html = f.read()

        with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(frontend_html)

        print(f"✅ Main frontend template created from {frontend_file_path}")

    except FileNotFoundError:
        print(f"❌ Error: Could not find frontend_artifact.html at {frontend_file_path}")
        # Create fallback main template
        create_fallback_main_template(templates_dir)

    # Create relationship template
    try:
        with open(relationship_file_path, 'r', encoding='utf-8') as f:
            relationship_html = f.read()

        with open(os.path.join(templates_dir, 'relationship.html'), 'w', encoding='utf-8') as f:
            f.write(relationship_html)

        print(f"✅ Relationship template created from {relationship_file_path}")

    except FileNotFoundError:
        print(f"⚠️ Warning: Could not find relationship.html at {relationship_file_path}")
        print("Creating fallback relationship template...")
        create_fallback_relationship_template(templates_dir)

def create_fallback_main_template(templates_dir):
    """Create fallback main template"""
    fallback_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Explorer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="alert alert-warning">
            <h4>Template Loading Issue</h4>
            <p>Could not load the main frontend template. Please ensure <code>frontend_artifact.html</code> is in the same directory as <code>web_db_explorer.py</code></p>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">🏠 Main Database Explorer</h5>
                        <p class="card-text">Upload and explore your database files</p>
                        <a href="/" class="btn btn-primary">Go to Main Interface</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">🔗 Relationship Visualization</h5>
                        <p class="card-text">Visualize database relationships and schema</p>
                        <a href="/relationship" class="btn btn-success">Go to Relationships</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="mt-4">
            <h5>API Endpoints:</h5>
            <ul>
                <li><a href="/api/docs">API Documentation</a></li>
                <li><a href="/api/status">API Status</a></li>
                <li><a href="/check-templates">Template Status</a></li>
            </ul>
        </div>
    </div>
</body>
</html>"""

    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(fallback_html)

def create_fallback_relationship_template(templates_dir):
    """Create fallback relationship template"""
    fallback_relationship_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Relationship Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        .nav {
            margin-bottom: 20px;
        }
        .nav a {
            background: #007bff;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 6px;
            margin-right: 10px;
            display: inline-block;
        }
        .nav a:hover {
            background: #0056b3;
        }
        .error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .info {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .api-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .api-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .api-card h5 {
            margin: 0 0 10px 0;
            color: #007bff;
        }
        .api-card a {
            color: #007bff;
            text-decoration: none;
        }
        .api-card a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔗 Database Relationship Visualization</h1>
            <p>Enhanced visualization and analysis of database relationships</p>
        </div>
        
        <div class="nav">
            <a href="/">← Back to Main Database Explorer</a>
            <a href="/api/docs">📚 API Documentation</a>
            <a href="/api/status">💚 System Status</a>
            <a href="/check-templates">🔧 Template Status</a>
        </div>
        
        <div class="error">
            <h4>🚨 Template Loading Issue</h4>
            <p>Could not load the full relationship visualization template. Please ensure <code>relationship.html</code> is in the same directory as <code>web_db_explorer.py</code></p>
            
            <h5>To fix this:</h5>
            <ol>
                <li>Copy the <code>relationship.html</code> file to the same directory as <code>web_db_explorer.py</code></li>
                <li>Restart the web server</li>
                <li>Refresh this page</li>
            </ol>
        </div>
        
        <div class="info">
            <h4>🔧 Available Relationship API Endpoints</h4>
            <p>You can still access the relationship functionality through these API endpoints:</p>
            
            <div class="api-list">
                <div class="api-card">
                    <h5>📊 Relationship Graph</h5>
                    <p>Get complete relationship graph data</p>
                    <a href="/api/schema/relationship-graph">View JSON Data</a>
                </div>
                
                <div class="api-card">
                    <h5>🔍 Relationship Analysis</h5>
                    <p>Detailed analysis of database relationships</p>
                    <a href="/api/schema/relationship-analysis">View Analysis</a>
                </div>
                
                <div class="api-card">
                    <h5>📈 Relationship Metrics</h5>
                    <p>Comprehensive relationship metrics</p>
                    <a href="/api/schema/relationship-metrics">View Metrics</a>
                </div>
                
                <div class="api-card">
                    <h5>🔗 Foreign Keys</h5>
                    <p>All foreign key relationships</p>
                    <a href="/api/schema/foreign-keys">View Foreign Keys</a>
                </div>
                
                <div class="api-card">
                    <h5>🏗️ Schema Statistics</h5>
                    <p>Complete schema statistics</p>
                    <a href="/api/schema/statistics">View Statistics</a>
                </div>
                
                <div class="api-card">
                    <h5>💡 Optimization Suggestions</h5>
                    <p>Database optimization recommendations</p>
                    <a href="/api/schema/optimize-suggestions">View Suggestions</a>
                </div>
            </div>
        </div>
        
        <div class="info">
            <h4>🛠️ Quick Actions</h4>
            <p>Try these API endpoints to get started with relationship analysis:</p>
            <ul>
                <li><strong>Connect a Database:</strong> Go to <a href="/">main interface</a> and upload a SQLite file</li>
                <li><strong>View Relationships:</strong> Use <code>/api/schema/relationship-graph</code> to get graph data</li>
                <li><strong>Find Issues:</strong> Use <code>/api/schema/relationship-analysis</code> for integrity checks</li>
                <li><strong>Get Insights:</strong> Use <code>/api/schema/relationship-metrics</code> for metrics</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Auto-refresh page every 10 seconds to check if template is available
        setTimeout(() => {
            fetch('/check-templates')
                .then(response => response.json())
                .then(data => {
                    if (data.templates && data.templates['relationship.html'] && data.templates['relationship.html'].exists) {
                        console.log('Relationship template is now available, refreshing...');
                        window.location.reload();
                    }
                })
                .catch(error => console.log('Template check failed:', error));
        }, 10000);
    </script>
</body>
</html>"""

    with open(os.path.join(templates_dir, 'relationship.html'), 'w', encoding='utf-8') as f:
        f.write(fallback_relationship_html)


# Add this route to serve the swagger spec
@app.route('/api/swagger.yaml')
def swagger_spec():
    """Serve the Swagger specification file"""
    try:
        # Try to read the swagger.yaml file from the api directory
        swagger_file_path = os.path.join(os.path.dirname(__file__), 'api', 'swagger.yaml')

        if os.path.exists(swagger_file_path):
            with open(swagger_file_path, 'r', encoding='utf-8') as f:
                swagger_content = f.read()
            return Response(swagger_content, mimetype='text/yaml')
        else:
            # Fallback: return a basic swagger spec if file not found
            fallback_content = '''openapi: 3.0.3
info:
  title: Universal Database Explorer API
  description: API for database exploration
  version: 1.0.0
servers:
  - url: http://localhost:5001
    description: Development server
paths:
  /api/status:
    get:
      summary: API health check
      responses:
        '200':
          description: API is healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "healthy"
'''
            return Response(fallback_content, mimetype='text/yaml')

    except Exception as e:
        print(f"Error loading swagger.yaml: {e}")
        return jsonify({"error": "Failed to load API documentation"}), 500

# Add this route for the API documentation homepage
@app.route('/api')
def api_docs_redirect():
    """Redirect to Swagger UI documentation"""
    return redirect('/api/docs')

# Add API status endpoint
@app.route('/api/status')
def api_status():
    """API health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Universal Database Explorer API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "connected_database": web_explorer.connected,
        "database_name": os.path.basename(web_explorer.db_path) if web_explorer.db_path else None,
        "features": {
            "visualization": VISUALIZATION_AVAILABLE,
            "file_upload": True,
            "query_execution": True,
            "data_export": True,
            "performance_analysis": True
        }
    })

# Add CORS headers for API endpoints
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Add this to your main() function in web_db_explorer.py
def main():
    """Main function to start the web application"""
    print("🚀 Starting Universal Database Explorer - Web Interface")
    print("=" * 60)

    # Create templates (now handles both main and relationship templates)
    create_templates()

    # Setup Swagger UI
    setup_swagger(app)

    # Check for required dependencies
    missing_deps = []

    try:
        import flask
    except ImportError:
        missing_deps.append("flask")

    if not VISUALIZATION_AVAILABLE:
        missing_deps.append("matplotlib seaborn pandas")

    # Check for Swagger UI dependency
    try:
        import flask_swagger_ui
    except ImportError:
        missing_deps.append("flask-swagger-ui")

    if missing_deps:
        print("⚠️  Missing optional dependencies:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        print("\nSome features may be limited without these dependencies.")
        print()

    # Start the Flask app
    try:
        # Try to find an available port
        import socket
        port = 5001  # Default port

        for test_port in [5001, 5002, 5003, 8000, 8080, 3000]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', test_port))
                    port = test_port
                    break
            except OSError:
                continue

        print(f"🌐 Web Interface Available At:")
        print(f"   📊 Main Database Explorer: http://localhost:{port}")
        print(f"   🔗 Relationship Visualization: http://localhost:{port}/relationship")
        print(f"   📚 API Documentation: http://localhost:{port}/api/docs")
        print(f"   💚 API Status: http://localhost:{port}/api/status")
        print(f"   🔧 Template Status: http://localhost:{port}/check-templates")
        print()
        print("📋 Quick Start Guide:")
        print("1. Open your web browser")
        print(f"2. Go to http://localhost:{port} for the main interface")
        print("3. Upload a SQLite database file")
        print("4. Explore your data with the main interface")
        print(f"5. Go to http://localhost:{port}/relationship for enhanced relationship visualization")
        print()
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)

        # Configure Flask for development
        app.config['DEBUG'] = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True

        # Start the server on the found port
        print(f"🌐 Starting server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
    finally:
        # Cleanup
        if web_explorer:
            web_explorer.close()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    main()

# Additional utility functions and routes

@app.route('/api/sample-queries')
def get_sample_queries():
    """Get sample SQL queries"""
    queries = {
        "basic": [
            {
                "name": "Show all tables",
                "query": "SELECT name FROM sqlite_master WHERE type='table';"
            },
            {
                "name": "Count tables",
                "query": "SELECT COUNT(*) as table_count FROM sqlite_master WHERE type='table';"
            },
            {
                "name": "Database info",
                "query": "PRAGMA database_list;"
            }
        ],
        "analysis": [
            {
                "name": "Table sizes",
                "query": "SELECT name, (SELECT COUNT(*) FROM table_name) as row_count FROM sqlite_master WHERE type='table';"
            },
            {
                "name": "Column info for table",
                "query": "PRAGMA table_info(table_name);"
            },
            {
                "name": "Foreign key check",
                "query": "PRAGMA foreign_key_check;"
            }
        ],
        "data": [
            {
                "name": "Find duplicates",
                "query": "SELECT column1, column2, COUNT(*) FROM table_name GROUP BY column1, column2 HAVING COUNT(*) > 1;"
            },
            {
                "name": "Null value count",
                "query": "SELECT COUNT(*) - COUNT(column_name) as null_count FROM table_name;"
            },
            {
                "name": "Data distribution",
                "query": "SELECT column_name, COUNT(*) as frequency FROM table_name GROUP BY column_name ORDER BY frequency DESC LIMIT 10;"
            }
        ]
    }

    return jsonify(queries)

@app.route('/api/table/<table_name>/generate-sql')
def generate_table_sql(table_name):
    """Generate various SQL statements for a table"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print(f"Generating SQL for table: {table_name}")  # Debug log

        # Skip the problematic get_table_structure method and get info directly
        # Method 1: Get columns using SELECT * LIMIT 0
        columns_query = f"SELECT * FROM `{table_name}` LIMIT 0"
        print(f"Getting columns with query: {columns_query}")  # Debug log

        columns_result = web_explorer.execute_query(columns_query)
        print(f"Columns result: {columns_result}")  # Debug log

        columns = []
        if columns_result:
            if isinstance(columns_result, dict) and columns_result.get('success'):
                columns = columns_result.get('columns', [])
            elif isinstance(columns_result, list):
                # For empty result, try to get columns from first row keys
                if len(columns_result) > 0 and not columns_result[0].get('error'):
                    columns = list(columns_result[0].keys())

        # Method 2: If SELECT didn't work, try getting from sqlite_master
        if not columns:
            print("SELECT method failed, trying sqlite_master...")  # Debug log
            schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            schema_result = web_explorer.execute_query(schema_query)
            print(f"Schema result: {schema_result}")  # Debug log

            if schema_result:
                create_sql = ""
                if isinstance(schema_result, dict) and schema_result.get('success') and schema_result.get('data'):
                    if len(schema_result['data']) > 0:
                        create_sql = schema_result['data'][0].get('sql', '')
                elif isinstance(schema_result, list) and not (len(schema_result) > 0 and schema_result[0].get('error')):
                    if len(schema_result) > 0:
                        create_sql = schema_result[0].get('sql', '')

                print(f"CREATE SQL: {create_sql}")  # Debug log

                # Parse column names from CREATE TABLE statement
                if create_sql:
                    import re
                    # Extract content between parentheses
                    match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
                    if match:
                        columns_text = match.group(1)
                        # Split by comma and extract column names
                        column_lines = [line.strip() for line in columns_text.split(',')]
                        for line in column_lines:
                            if line and not line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT')):
                                # Extract column name (first word, remove quotes)
                                parts = line.split()
                                if len(parts) >= 1:
                                    col_name = parts[0].strip('`"\'')
                                    if col_name and col_name.upper() not in ('PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT'):
                                        columns.append(col_name)

        if not columns:
            return jsonify({"error": f"Could not determine columns for table '{table_name}'"})

        print(f"Final columns: {columns}")  # Debug log

        # Try to get primary key info (optional, won't fail if we can't get it)
        primary_keys = []
        try:
            # Look for PRIMARY KEY in the CREATE statement
            schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            schema_result = web_explorer.execute_query(schema_query)

            if schema_result:
                create_sql = ""
                if isinstance(schema_result, dict) and schema_result.get('success') and schema_result.get('data'):
                    if len(schema_result['data']) > 0:
                        create_sql = schema_result['data'][0].get('sql', '')
                elif isinstance(schema_result, list) and not (len(schema_result) > 0 and schema_result[0].get('error')):
                    if len(schema_result) > 0:
                        create_sql = schema_result[0].get('sql', '')

                if create_sql:
                    # Look for PRIMARY KEY or AUTOINCREMENT
                    import re
                    lines = create_sql.split('\n')
                    for line in lines:
                        line = line.strip()
                        if 'PRIMARY KEY' in line.upper() or 'AUTOINCREMENT' in line.upper():
                            parts = line.split()
                            if len(parts) >= 1:
                                col_name = parts[0].strip('`"\'(),')
                                if col_name in columns and col_name not in primary_keys:
                                    primary_keys.append(col_name)
        except Exception as pk_error:
            print(f"Could not determine primary keys: {pk_error}")
            # Use first column as fallback
            if columns:
                primary_keys = [columns[0]]

        print(f"Primary keys: {primary_keys}")  # Debug log

        # Generate SQL templates
        sql_templates = {
            "select_all": f"SELECT * FROM `{table_name}` LIMIT 10;",
            "select_columns": f"SELECT {', '.join(f'`{col}`' for col in columns[:5])} FROM `{table_name}` LIMIT 10;",
            "count_rows": f"SELECT COUNT(*) as total_rows FROM `{table_name}`;",
        }

        # Add distinct and group by queries
        if columns:
            sql_templates["distinct_values"] = f"SELECT DISTINCT `{columns[0]}` FROM `{table_name}`;"
            sql_templates["group_by"] = f"SELECT `{columns[0]}`, COUNT(*) as count FROM `{table_name}` GROUP BY `{columns[0]}` ORDER BY count DESC;"

        # Add data modification templates
        if columns:
            # INSERT template
            placeholders = ', '.join(['?' for _ in columns])
            column_list = ', '.join(f'`{col}`' for col in columns)
            sql_templates["insert_template"] = f"INSERT INTO `{table_name}` ({column_list}) VALUES ({placeholders});"

            # UPDATE template
            pk_column = primary_keys[0] if primary_keys else columns[0]
            update_column = columns[1] if len(columns) > 1 else columns[0]
            sql_templates["update_template"] = f"UPDATE `{table_name}` SET `{update_column}` = ? WHERE `{pk_column}` = ?;"

            # DELETE template
            sql_templates["delete_template"] = f"DELETE FROM `{table_name}` WHERE `{pk_column}` = ?;"

        # Add specific templates for salary_audit_log table
        if table_name.lower() == 'salary_audit_log':
            sql_templates.update({
                "recent_changes": f"SELECT * FROM `{table_name}` WHERE change_date >= date('now', '-30 days') ORDER BY change_date DESC;",
                "by_employee": f"SELECT * FROM `{table_name}` WHERE employee_id = ? ORDER BY change_date DESC;",
                "salary_increases": f"SELECT * FROM `{table_name}` WHERE new_salary > old_salary ORDER BY change_date DESC;",
                "salary_decreases": f"SELECT * FROM `{table_name}` WHERE new_salary < old_salary ORDER BY change_date DESC;",
                "changes_by_user": f"SELECT changed_by, COUNT(*) as change_count FROM `{table_name}` GROUP BY changed_by ORDER BY change_count DESC;",
                "average_change": f"SELECT AVG(new_salary - old_salary) as avg_change FROM `{table_name}` WHERE new_salary != old_salary;",
                "with_employee_details": f"""SELECT sal.*, emp.first_name, emp.last_name, emp.email 
                                           FROM `{table_name}` sal 
                                           LEFT JOIN employees emp ON sal.employee_id = emp.employee_id 
                                           ORDER BY sal.change_date DESC;"""
            })

        print(f"Generated {len(sql_templates)} SQL templates")  # Debug log

        return jsonify({
            "success": True,
            "sql_templates": sql_templates,
            "table_info": {
                "name": table_name,
                "columns": columns,
                "primary_keys": primary_keys,
                "column_count": len(columns)
            }
        })

    except Exception as e:
        print(f"Error generating SQL for table {table_name}: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/database-stats')
def get_database_stats():
    """Get detailed database statistics"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Starting comprehensive database statistics analysis...")  # Debug log

        stats = {
            "database_overview": {},
            "table_stats": [],
            "column_analysis": {},
            "constraint_stats": {},
            "index_stats": {},
            "data_quality": {},
            "schema_complexity": {},
            "storage_stats": {}
        }

        # Get database file information
        db_file_size = 0
        if web_explorer.db_path and os.path.exists(web_explorer.db_path):
            db_file_size = os.path.getsize(web_explorer.db_path)

        # Get all user tables (excluding system tables)
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        print(f"Found tables: {table_names}")  # Debug log

        if not table_names:
            return jsonify({
                "success": False,
                "error": "No user tables found in database"
            })

        # ===== FIX: Improved Index Counting =====
        # Method 1: Get total user-created indexes from sqlite_master
        total_indexes = 0
        try:
            # Count all user-created indexes (exclude auto-generated ones)
            indexes_count_query = """
                                  SELECT COUNT(*) as index_count
                                  FROM sqlite_master
                                  WHERE type='index'
                                    AND name NOT LIKE 'sqlite_%'
                                    AND sql IS NOT NULL \
                                  """
            indexes_count_result = web_explorer.execute_query(indexes_count_query)

            if indexes_count_result:
                if isinstance(indexes_count_result, dict) and indexes_count_result.get('success') and indexes_count_result.get('data'):
                    total_indexes = indexes_count_result['data'][0].get('index_count', 0)
                elif isinstance(indexes_count_result, list) and not (len(indexes_count_result) > 0 and indexes_count_result[0].get('error')):
                    total_indexes = indexes_count_result[0].get('index_count', 0)

            print(f"Total user-created indexes from sqlite_master: {total_indexes}")
        except Exception as idx_count_error:
            print(f"Error counting indexes from sqlite_master: {idx_count_error}")

            # Fallback: Count indexes per table and sum them up
            total_indexes_fallback = 0
            for table_name in table_names:
                try:
                    indexes_query = f"PRAGMA index_list(`{table_name}`)"
                    indexes_result = web_explorer.execute_query(indexes_query)

                    if indexes_result:
                        indexes = []
                        if isinstance(indexes_result, dict) and indexes_result.get('success') and indexes_result.get('data'):
                            indexes = indexes_result['data']
                        elif isinstance(indexes_result, list) and not (len(indexes_result) > 0 and indexes_result[0].get('error')):
                            indexes = indexes_result

                        # Count user-created indexes only
                        user_indexes = [idx for idx in indexes if not idx.get('name', '').startswith('sqlite_autoindex') and not idx.get('name', '').startswith('sqlite_')]
                        total_indexes_fallback += len(user_indexes)
                        print(f"Table {table_name}: {len(user_indexes)} user indexes")
                except Exception as table_idx_error:
                    print(f"Error getting indexes for table {table_name}: {table_idx_error}")
                    continue

            total_indexes = total_indexes_fallback
            print(f"Total indexes via fallback method: {total_indexes}")

        # Initialize counters
        total_columns = 0
        total_rows = 0
        total_triggers = 0
        total_views = 0
        column_types = {}
        tables_with_pk = 0
        tables_with_fk = 0
        total_foreign_keys = 0
        data_quality_issues = []

        # Analyze each table with real-time data
        tables_with_indexes = 0

        for table_name in table_names:
            try:
                print(f"Analyzing table: {table_name}")  # Debug log

                # Get real-time row count
                count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
                count_result = web_explorer.execute_query(count_query)

                row_count = 0
                if count_result:
                    if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                        row_count = count_result['data'][0].get('row_count', 0)
                    elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                        row_count = count_result[0].get('row_count', 0)

                # ... rest of the table analysis code remains the same ...
                # (column analysis, foreign keys, etc.)

                # ===== FIX: Improved Per-Table Index Counting =====
                indexes_count = 0
                try:
                    # Method 1: Try PRAGMA index_list
                    indexes_query = f"PRAGMA index_list(`{table_name}`)"
                    indexes_result = web_explorer.execute_query(indexes_query)

                    if indexes_result:
                        indexes = []
                        if isinstance(indexes_result, dict) and indexes_result.get('success') and indexes_result.get('data'):
                            indexes = indexes_result['data']
                        elif isinstance(indexes_result, list) and not (len(indexes_result) > 0 and indexes_result[0].get('error')):
                            indexes = indexes_result

                        # Filter out auto-generated indexes and count user-created ones
                        user_indexes = []
                        for idx in indexes:
                            idx_name = idx.get('name', '')
                            # Exclude auto-generated SQLite indexes
                            if not idx_name.startswith('sqlite_autoindex') and not idx_name.startswith('sqlite_'):
                                user_indexes.append(idx)

                        indexes_count = len(user_indexes)
                        print(f"Table {table_name} has {indexes_count} user-created indexes")

                    # Method 2: Fallback - check sqlite_master for this table's indexes
                    if indexes_count == 0:
                        table_indexes_query = f"""
                            SELECT COUNT(*) as table_index_count 
                            FROM sqlite_master 
                            WHERE type='index' 
                              AND tbl_name='{table_name}' 
                              AND name NOT LIKE 'sqlite_%'
                              AND sql IS NOT NULL
                        """
                        table_indexes_result = web_explorer.execute_query(table_indexes_query)

                        if table_indexes_result:
                            if isinstance(table_indexes_result, dict) and table_indexes_result.get('success') and table_indexes_result.get('data'):
                                indexes_count = table_indexes_result['data'][0].get('table_index_count', 0)
                            elif isinstance(table_indexes_result, list) and not (len(table_indexes_result) > 0 and table_indexes_result[0].get('error')):
                                indexes_count = table_indexes_result[0].get('table_index_count', 0)

                except Exception as idx_error:
                    print(f"Error getting index count for {table_name}: {idx_error}")
                    indexes_count = 0

                # Update tables with indexes count
                if indexes_count > 0:
                    tables_with_indexes += 1

                # ... continue with rest of table analysis ...

            except Exception as table_error:
                print(f"Error analyzing table {table_name}: {table_error}")
                continue

        # Get views count with better error handling
        try:
            views_query = "SELECT COUNT(*) as view_count FROM sqlite_master WHERE type='view'"
            views_result = web_explorer.execute_query(views_query)

            if views_result:
                if isinstance(views_result, dict) and views_result.get('success') and views_result.get('data'):
                    total_views = views_result['data'][0].get('view_count', 0)
                elif isinstance(views_result, list) and not (len(views_result) > 0 and views_result[0].get('error')):
                    total_views = views_result[0].get('view_count', 0)
        except Exception as views_error:
            print(f"Error counting views: {views_error}")
            total_views = 0

        # Get triggers count with better error handling
        try:
            triggers_query = "SELECT COUNT(*) as trigger_count FROM sqlite_master WHERE type='trigger'"
            triggers_result = web_explorer.execute_query(triggers_query)

            if triggers_result:
                if isinstance(triggers_result, dict) and triggers_result.get('success') and triggers_result.get('data'):
                    total_triggers = triggers_result['data'][0].get('trigger_count', 0)
                elif isinstance(triggers_result, list) and not (len(triggers_result) > 0 and triggers_result[0].get('error')):
                    total_triggers = triggers_result[0].get('trigger_count', 0)
        except Exception as triggers_error:
            print(f"Error counting triggers: {triggers_error}")
            total_triggers = 0

        # Database overview with corrected totals
        stats["database_overview"] = {
            "database_name": os.path.basename(web_explorer.db_path) if web_explorer.db_path else "Unknown",
            "file_size_mb": round(db_file_size / (1024 * 1024), 3),
            "total_tables": len(table_names),
            "total_views": total_views,
            "total_triggers": total_triggers,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "schema_version": "SQLite",
            "last_analyzed": datetime.now().isoformat()
        }

        # ===== FIX: Corrected Index Stats =====
        tables_without_indexes = len(table_names) - tables_with_indexes

        stats["index_stats"] = {
            "total_indexes": total_indexes,  # This should now show the correct count
            "tables_with_indexes": tables_with_indexes,
            "tables_without_indexes": tables_without_indexes,
            "avg_indexes_per_table": round(total_indexes / len(table_names), 2) if table_names else 0,
            "index_coverage_percentage": round((tables_with_indexes / len(table_names) * 100), 1) if table_names else 0,
            "indexing_strategy": "Optimal" if tables_without_indexes == 0 else "Good" if tables_without_indexes <= 2 else "Needs Improvement"
        }

        print(f"Final index stats: total={total_indexes}, with_indexes={tables_with_indexes}, coverage={stats['index_stats']['index_coverage_percentage']}%")

        # ... rest of the function remains the same ...

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        print(f"Error in get_database_stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/export-schema')
def export_schema():
    """Export database schema as SQL"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        schema_sql = []
        schema_sql.append("-- Database Schema Export")
        schema_sql.append(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        schema_sql.append(f"-- Database: {os.path.basename(web_explorer.db_path)}")
        schema_sql.append("")

        if web_explorer.schema:
            # Export tables
            for table in web_explorer.schema.tables:
                schema_sql.append(f"-- Table: {table.name}")
                schema_sql.append(f"CREATE TABLE {table.name} (")

                column_defs = []
                for col in table.columns:
                    col_def = f"    {col['name']} {col['type']}"
                    if not col['nullable']:
                        col_def += " NOT NULL"
                    if col['default']:
                        col_def += f" DEFAULT {col['default']}"
                    column_defs.append(col_def)

                # Add primary key
                if table.primary_keys:
                    pk_def = f"    PRIMARY KEY ({', '.join(table.primary_keys)})"
                    column_defs.append(pk_def)

                schema_sql.append(',\n'.join(column_defs))
                schema_sql.append(");")
                schema_sql.append("")

                # Add indexes
                for idx in table.indexes:
                    if not idx['name'].startswith('sqlite_'):
                        unique_keyword = "UNIQUE " if idx.get('unique') else ""
                        columns_str = ', '.join(idx.get('columns', []))
                        schema_sql.append(f"CREATE {unique_keyword}INDEX {idx['name']} ON {table.name} ({columns_str});")

                schema_sql.append("")

        # Create temp file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'schema.sql')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(schema_sql))

        return send_file(temp_path, as_attachment=True, download_name='database_schema.sql')

    except Exception as e:
        return jsonify({"error": str(e)})

# Additional API routes to add to web_db_explorer.py
# These should be inserted after the existing routes

@app.route('/api/table/<table_name>/relationships')
def get_table_relationships(table_name):
    """Get table relationships"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        # Get structure for the specified table
        structure_result = web_explorer.get_table_structure(table_name)
        if not structure_result:
            return jsonify({"error": f"Table '{table_name}' not found"})

        # Handle different structure formats
        if isinstance(structure_result, dict):
            if 'structure' in structure_result:
                structure = structure_result['structure']
            elif 'columns' in structure_result:
                structure = structure_result
            else:
                return jsonify({"error": "Invalid table structure format"})
        else:
            return jsonify({"error": "Could not retrieve table structure"})

        # Initialize relationships data
        relationships = []
        related_tables = set()

        # Tables this table references (outgoing foreign keys)
        if 'foreign_keys' in structure and structure['foreign_keys']:
            for fk in structure['foreign_keys']:
                relationships.append({
                    "type": "references",
                    "table": fk['referenced_table'],
                    "relationship": "many_to_one",
                    "foreign_key": {
                        "column": fk['column'],
                        "referenced_table": fk['referenced_table'],
                        "referenced_column": fk['referenced_column']
                    }
                })
                related_tables.add(fk['referenced_table'])

        # Tables that reference this table (incoming foreign keys)
        if web_explorer.schema and web_explorer.schema.tables:
            for table in web_explorer.schema.tables:
                if table.name == table_name:
                    continue  # Skip self

                for fk in table.foreign_keys:
                    if fk['referenced_table'] == table_name:
                        relationships.append({
                            "type": "referenced_by",
                            "table": table.name,
                            "relationship": "one_to_many",
                            "foreign_key": {
                                "column": fk['column'],
                                "referenced_table": fk['referenced_table'],
                                "referenced_column": fk['referenced_column']
                            }
                        })
                        related_tables.add(table.name)

        # Get additional information about related tables
        related_table_info = []
        for related_table in related_tables:
            try:
                # Find the table info in schema
                table_info = None
                if web_explorer.schema and web_explorer.schema.tables:
                    for table in web_explorer.schema.tables:
                        if table.name == related_table:
                            table_info = {
                                "name": table.name,
                                "columns": len(table.columns),
                                "rows": table.row_count,
                                "primary_keys": table.primary_keys
                            }
                            break

                if not table_info:
                    table_info = {
                        "name": related_table,
                        "columns": 0,
                        "rows": 0,
                        "primary_keys": []
                    }

                related_table_info.append(table_info)
            except Exception as e:
                print(f"Error getting info for related table {related_table}: {e}")
                related_table_info.append({
                    "name": related_table,
                    "columns": 0,
                    "rows": 0,
                    "primary_keys": []
                })

        return jsonify({
            "success": True,
            "table_name": table_name,
            "relationships": relationships,
            "related_tables": list(related_tables),
            "related_table_info": related_table_info,
            "relationship_count": len(relationships)
        })

    except Exception as e:
        print(f"Error in get_table_relationships for table {table_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/join-tables', methods=['POST'])
def join_tables():
    """Perform table JOIN operation"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    table1 = data.get('table1', '').strip()
    table2 = data.get('table2', '').strip()
    join_condition = data.get('join_condition', '').strip()
    columns = data.get('columns', '*').strip()
    where_clause = data.get('where_clause', '').strip()
    limit = data.get('limit', 100)
    join_type = data.get('join_type', 'INNER').strip().upper()  # New parameter

    if not all([table1, table2, join_condition]):
        return jsonify({"error": "table1, table2, and join_condition are required"})

    # Validate join_type
    valid_join_types = ['INNER', 'LEFT', 'RIGHT', 'FULL OUTER', 'CROSS']
    if join_type not in valid_join_types:
        return jsonify({"error": f"Invalid join_type. Must be one of: {', '.join(valid_join_types)}"})

    try:
        print(f"Executing {join_type} JOIN: {table1} + {table2}")  # Debug log
        print(f"Join condition: {join_condition}")  # Debug log
        print(f"Columns: {columns}")  # Debug log
        print(f"Where clause: {where_clause}")  # Debug log
        print(f"Limit: {limit}")  # Debug log

        # Build the JOIN query with specified join type
        if join_type == 'CROSS':
            # CROSS JOIN doesn't use ON condition
            query_parts = [
                f"SELECT {columns}",
                f"FROM `{table1}`",
                f"CROSS JOIN `{table2}`"
            ]
        else:
            query_parts = [
                f"SELECT {columns}",
                f"FROM `{table1}`",
                f"{join_type} JOIN `{table2}` ON {join_condition}"
            ]

        # Add WHERE clause if provided
        if where_clause:
            query_parts.append(f"WHERE {where_clause}")

        # Add LIMIT if specified
        if limit:
            query_parts.append(f"LIMIT {limit}")

        join_query = " ".join(query_parts)

        print(f"Final {join_type} JOIN query: {join_query}")  # Debug log

        # Execute the JOIN query
        result = web_explorer.execute_query(join_query)

        print(f"JOIN query result: {result}")  # Debug log

        # Handle different response formats
        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    return jsonify({
                        "success": True,
                        "data": result.get('data', []),
                        "row_count": len(result.get('data', [])),
                        "columns": result.get('columns', []),
                        "join_type": join_type,
                        "query": join_query
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', f'{join_type} JOIN query failed')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    # Extract columns from first row if data exists
                    columns_list = []
                    if len(result) > 0:
                        columns_list = list(result[0].keys())

                    return jsonify({
                        "success": True,
                        "data": result,
                        "row_count": len(result),
                        "columns": columns_list,
                        "join_type": join_type,
                        "query": join_query
                    })

        return jsonify({
            "success": False,
            "error": "No response from database"
        })

    except Exception as e:
        print(f"Error in join_tables: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/table/<table_name>/analyze')
def analyze_table(table_name):
    """Analyze table data"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    include_stats = request.args.get('include_stats', 'true').lower() == 'true'
    include_distribution = request.args.get('include_distribution', 'false').lower() == 'true'

    try:
        # Get table structure
        structure_result = web_explorer.get_table_structure(table_name)
        if not structure_result:
            return jsonify({"error": f"Table '{table_name}' not found"})

        # Handle different structure formats
        if isinstance(structure_result, dict):
            if 'structure' in structure_result:
                structure = structure_result['structure']
            elif 'columns' in structure_result:
                structure = structure_result
            else:
                return jsonify({"error": "Invalid table structure format"})
        else:
            return jsonify({"error": "Could not retrieve table structure"})

        # Get total row count using multiple methods
        total_rows = 0
        try:
            # Method 1: Try to get from structure
            if 'row_count' in structure:
                total_rows = structure['row_count']
            else:
                # Method 2: Query the database directly
                count_query = f"SELECT COUNT(*) as total_count FROM `{table_name}`"
                count_result = web_explorer.execute_query(count_query)

                if count_result:
                    if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                        total_rows = count_result['data'][0].get('total_count', 0)
                    elif isinstance(count_result, list) and len(count_result) > 0 and not count_result[0].get('error'):
                        total_rows = count_result[0].get('total_count', 0)
        except Exception as e:
            print(f"Error getting row count: {e}")
            total_rows = 0

        # Get columns
        columns = []
        if 'columns' in structure and isinstance(structure['columns'], list):
            columns = structure['columns']
        else:
            return jsonify({"error": "Could not retrieve table columns"})

        analysis = {
            "basic_stats": {
                "total_rows": total_rows,
                "total_columns": len(columns)
            }
        }

        if include_stats:
            # Analyze numeric columns
            numeric_analysis = []
            text_analysis = []

            for column in columns:
                col_name = column['name'] if isinstance(column, dict) else str(column)
                col_type = (column.get('type', '') if isinstance(column, dict) else '').upper()

                # Numeric column analysis
                if any(t in col_type for t in ['INT', 'REAL', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE']):
                    try:
                        stats_query = f"""
                            SELECT 
                                MIN(`{col_name}`) as min_val,
                                MAX(`{col_name}`) as max_val,
                                AVG(`{col_name}`) as avg_val,
                                COUNT(*) - COUNT(`{col_name}`) as null_count
                            FROM `{table_name}`
                        """
                        stats_result = web_explorer.execute_query(stats_query)

                        if stats_result:
                            stats_data = None
                            if isinstance(stats_result, dict) and stats_result.get('success') and stats_result.get('data'):
                                stats_data = stats_result['data'][0]
                            elif isinstance(stats_result, list) and len(stats_result) > 0 and not stats_result[0].get('error'):
                                stats_data = stats_result[0]

                            if stats_data:
                                avg_val = stats_data.get('avg_val', 0)
                                try:
                                    avg_val = round(float(avg_val), 2) if avg_val is not None else 0
                                except (ValueError, TypeError):
                                    avg_val = 0

                                numeric_analysis.append({
                                    "column": col_name,
                                    "min_value": stats_data.get('min_val'),
                                    "max_value": stats_data.get('max_val'),
                                    "avg_value": avg_val,
                                    "null_count": stats_data.get('null_count', 0)
                                })
                    except Exception as e:
                        print(f"Error analyzing numeric column {col_name}: {e}")
                        pass

                # Text column analysis
                elif any(t in col_type for t in ['TEXT', 'VARCHAR', 'CHAR', 'STRING']):
                    try:
                        text_query = f"""
                            SELECT 
                                MIN(LENGTH(`{col_name}`)) as min_len,
                                MAX(LENGTH(`{col_name}`)) as max_len,
                                AVG(LENGTH(`{col_name}`)) as avg_len,
                                COUNT(*) - COUNT(`{col_name}`) as null_count
                            FROM `{table_name}`
                            WHERE `{col_name}` IS NOT NULL
                        """
                        text_result = web_explorer.execute_query(text_query)

                        if text_result:
                            text_data = None
                            if isinstance(text_result, dict) and text_result.get('success') and text_result.get('data'):
                                text_data = text_result['data'][0]
                            elif isinstance(text_result, list) and len(text_result) > 0 and not text_result[0].get('error'):
                                text_data = text_result[0]

                            if text_data:
                                avg_len = text_data.get('avg_len', 0)
                                try:
                                    avg_len = round(float(avg_len), 2) if avg_len is not None else 0
                                except (ValueError, TypeError):
                                    avg_len = 0

                                text_analysis.append({
                                    "column": col_name,
                                    "min_length": text_data.get('min_len', 0),
                                    "max_length": text_data.get('max_len', 0),
                                    "avg_length": avg_len,
                                    "null_count": text_data.get('null_count', 0)
                                })
                    except Exception as e:
                        print(f"Error analyzing text column {col_name}: {e}")
                        pass

            analysis["numeric_analysis"] = numeric_analysis
            analysis["text_analysis"] = text_analysis

        if include_distribution:
            # Data distribution analysis for categorical columns
            data_distribution = []

            for column in columns:
                col_name = column['name'] if isinstance(column, dict) else str(column)
                col_type = (column.get('type', '') if isinstance(column, dict) else '').upper()

                # Only analyze text columns with reasonable cardinality
                if any(t in col_type for t in ['TEXT', 'VARCHAR', 'CHAR', 'STRING']):
                    try:
                        # Check cardinality first
                        card_query = f"SELECT COUNT(DISTINCT `{col_name}`) as distinct_count FROM `{table_name}`"
                        card_result = web_explorer.execute_query(card_query)

                        distinct_count = 0
                        if card_result:
                            if isinstance(card_result, dict) and card_result.get('success') and card_result.get('data'):
                                distinct_count = card_result['data'][0].get('distinct_count', 0)
                            elif isinstance(card_result, list) and len(card_result) > 0 and not card_result[0].get('error'):
                                distinct_count = card_result[0].get('distinct_count', 0)

                        if distinct_count <= 50:  # Only for low cardinality
                            dist_query = f"""
                                SELECT 
                                    `{col_name}` as value,
                                    COUNT(*) as count,
                                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM `{table_name}`), 2) as percentage
                                FROM `{table_name}`
                                GROUP BY `{col_name}`
                                ORDER BY count DESC
                                LIMIT 10
                            """
                            dist_result = web_explorer.execute_query(dist_query)

                            if dist_result:
                                dist_data = []
                                if isinstance(dist_result, dict) and dist_result.get('success') and dist_result.get('data'):
                                    dist_data = dist_result['data']
                                elif isinstance(dist_result, list) and not (len(dist_result) > 0 and dist_result[0].get('error')):
                                    dist_data = dist_result

                                if dist_data:
                                    value_counts = []
                                    for row in dist_data:
                                        try:
                                            percentage = float(row.get('percentage', 0))
                                        except (ValueError, TypeError):
                                            percentage = 0

                                        value_counts.append({
                                            "value": str(row.get('value', '')),
                                            "count": row.get('count', 0),
                                            "percentage": percentage
                                        })

                                    if value_counts:
                                        data_distribution.append({
                                            "column": col_name,
                                            "value_counts": value_counts
                                        })
                    except Exception as e:
                        print(f"Error analyzing distribution for column {col_name}: {e}")
                        pass

            analysis["data_distribution"] = data_distribution

        return jsonify({
            "success": True,
            "analysis": analysis,
            "table_name": table_name
        })

    except Exception as e:
        print(f"Error in analyze_table for table {table_name}: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({"error": str(e)})

@app.route('/api/schema/create-table', methods=['POST'])
def create_table():
    """Create new table"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    table_name = data.get('table_name', '').strip()
    columns = data.get('columns', [])

    if not table_name or not columns:
        return jsonify({"error": "table_name and columns are required"})

    try:
        # Build CREATE TABLE statement
        column_defs = []

        for col in columns:
            col_name = col.get('name', '').strip()
            col_type = col.get('type', '').strip()
            constraints = col.get('constraints', [])
            default = col.get('default', '').strip()

            if not col_name or not col_type:
                return jsonify({"error": f"Column name and type are required for all columns"})

            # Start building the column definition
            col_def = f"`{col_name}` {col_type}"

            # Add constraints
            if constraints:
                col_def += " " + " ".join(constraints)

            # Add default value
            if default:
                col_def += f" DEFAULT {default}"

            column_defs.append(col_def)

        # Create the full CREATE TABLE statement
        create_query = f"CREATE TABLE `{table_name}` ({', '.join(column_defs)})"

        print(f"Executing CREATE TABLE query: {create_query}")  # Debug log

        # Execute the query
        result = web_explorer.execute_query(create_query)

        print(f"Query result: {result}")  # Debug log

        # Handle different response formats
        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    # Refresh schema to include new table
                    if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                        try:
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                        except:
                            pass  # Schema refresh failed, but table was created

                    return jsonify({
                        "success": True,
                        "message": f"Table '{table_name}' created successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', 'Unknown error creating table')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    # Refresh schema to include new table
                    if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                        try:
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                        except:
                            pass  # Schema refresh failed, but table was created

                    return jsonify({
                        "success": True,
                        "message": f"Table '{table_name}' created successfully"
                    })
        else:
            return jsonify({
                "success": False,
                "error": "No response from database"
            })

    except Exception as e:
        print(f"Exception in create_table: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/schema/create-index', methods=['POST'])
def create_index():
    """Create new index"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    index_name = data.get('index_name', '').strip()
    table_name = data.get('table_name', '').strip()
    columns = data.get('columns', [])
    unique = data.get('unique', False)

    if not all([index_name, table_name, columns]):
        return jsonify({"error": "index_name, table_name, and columns are required"})

    try:
        # Build CREATE INDEX statement
        unique_keyword = "UNIQUE " if unique else ""
        columns_str = ", ".join(f"`{col}`" for col in columns)

        create_query = f"CREATE {unique_keyword}INDEX `{index_name}` ON `{table_name}` ({columns_str})"

        print(f"Executing CREATE INDEX query: {create_query}")  # Debug log

        # Execute the query
        result = web_explorer.execute_query(create_query)

        print(f"Index creation result: {result}")  # Debug log

        # Handle different response formats
        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    # Refresh schema to include new index
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but index was created

                    return jsonify({
                        "success": True,
                        "message": f"Index '{index_name}' created successfully on table '{table_name}'"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', 'Unknown error creating index')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    # Success case - no error in result
                    # Refresh schema to include new index
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but index was created

                    return jsonify({
                        "success": True,
                        "message": f"Index '{index_name}' created successfully on table '{table_name}'"
                    })
        else:
            return jsonify({
                "success": False,
                "error": "No response from database"
            })

    except Exception as e:
        print(f"Exception in create_index: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/schema/create-view', methods=['POST'])
def create_view():
    """Create new view"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    view_name = data.get('view_name', '').strip()
    select_query = data.get('select_query', '').strip()

    if not all([view_name, select_query]):
        return jsonify({"error": "view_name and select_query are required"})

    try:
        # Build CREATE VIEW statement
        create_query = f"CREATE VIEW `{view_name}` AS {select_query}"

        print(f"Executing CREATE VIEW query: {create_query}")  # Debug log

        # Execute the query
        result = web_explorer.execute_query(create_query)

        print(f"View creation result: {result}")  # Debug log

        # Handle different response formats
        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    # Refresh schema to include new view
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but view was created

                    return jsonify({
                        "success": True,
                        "message": f"View '{view_name}' created successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', 'Unknown error creating view')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    # Success case - no error in result
                    # Refresh schema to include new view
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but view was created

                    return jsonify({
                        "success": True,
                        "message": f"View '{view_name}' created successfully"
                    })
        else:
            return jsonify({
                "success": False,
                "error": "No response from database"
            })

    except Exception as e:
        print(f"Exception in create_view: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/triggers')
def get_triggers():
    """List all triggers"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        triggers = []

        if web_explorer.schema and web_explorer.schema.triggers:
            for trigger in web_explorer.schema.triggers:
                # Parse trigger SQL to extract table and event information
                trigger_info = {
                    "name": trigger['name'],
                    "sql": trigger.get('sql', '')
                }

                # Try to extract table name from SQL
                sql = trigger.get('sql', '').upper()
                if ' ON ' in sql:
                    try:
                        parts = sql.split(' ON ')[1].split()
                        trigger_info["table"] = parts[0].strip()
                    except:
                        trigger_info["table"] = "unknown"

                triggers.append(trigger_info)

        return jsonify({
            "success": True,
            "triggers": triggers
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/triggers/<trigger_name>')
def get_trigger_details(trigger_name):
    """Get trigger details"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        if web_explorer.schema and web_explorer.schema.triggers:
            for trigger in web_explorer.schema.triggers:
                if trigger['name'] == trigger_name:
                    # Parse trigger details from SQL
                    sql = trigger.get('sql', '')
                    trigger_details = {
                        "name": trigger['name'],
                        "sql": sql,
                        "table": "unknown",
                        "event": "unknown",
                        "when_condition": None
                    }

                    # Try to parse event and table from SQL
                    sql_upper = sql.upper()
                    if ' ON ' in sql_upper:
                        try:
                            # Extract table name
                            parts = sql_upper.split(' ON ')[1].split()
                            trigger_details["table"] = parts[0].strip()

                            # Extract event
                            if 'BEFORE INSERT' in sql_upper:
                                trigger_details["event"] = "BEFORE INSERT"
                            elif 'AFTER INSERT' in sql_upper:
                                trigger_details["event"] = "AFTER INSERT"
                            elif 'BEFORE UPDATE' in sql_upper:
                                trigger_details["event"] = "BEFORE UPDATE"
                            elif 'AFTER UPDATE' in sql_upper:
                                trigger_details["event"] = "AFTER UPDATE"
                            elif 'BEFORE DELETE' in sql_upper:
                                trigger_details["event"] = "BEFORE DELETE"
                            elif 'AFTER DELETE' in sql_upper:
                                trigger_details["event"] = "AFTER DELETE"
                        except:
                            pass

                    return jsonify({
                        "success": True,
                        "trigger": trigger_details
                    })

        return jsonify({"error": f"Trigger '{trigger_name}' not found"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/triggers/<trigger_name>', methods=['DELETE'])
def drop_trigger(trigger_name):
    """Drop trigger"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        # First, check if trigger exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='trigger' AND name = '{trigger_name}'"
        check_result = web_explorer.execute_query(check_query)

        trigger_exists = False
        if check_result:
            if isinstance(check_result, dict) and check_result.get('success') and check_result.get('data'):
                trigger_exists = len(check_result['data']) > 0
            elif isinstance(check_result, list) and not (len(check_result) > 0 and check_result[0].get('error')):
                trigger_exists = len(check_result) > 0

        if not trigger_exists:
            return jsonify({
                "success": False,
                "error": f"Trigger '{trigger_name}' not found"
            })

        print(f"Dropping trigger: {trigger_name}")  # Debug log

        # Execute DROP TRIGGER statement
        drop_query = f"DROP TRIGGER `{trigger_name}`"

        print(f"Executing DROP TRIGGER query: {drop_query}")  # Debug log

        result = web_explorer.execute_query(drop_query)

        print(f"Drop trigger result: {result}")  # Debug log

        # Handle different response formats
        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    # Refresh schema to remove dropped trigger
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but trigger was dropped

                    return jsonify({
                        "success": True,
                        "message": f"Trigger '{trigger_name}' dropped successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', 'Unknown error dropping trigger')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    # Success case - no error in result
                    # Refresh schema to remove dropped trigger
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but trigger was dropped

                    return jsonify({
                        "success": True,
                        "message": f"Trigger '{trigger_name}' dropped successfully"
                    })
        else:
            return jsonify({
                "success": False,
                "error": "No response from database"
            })

    except Exception as e:
        print(f"Exception in drop_trigger: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/schema/create-trigger', methods=['POST'])
def create_trigger():
    """Create new trigger"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    trigger_name = data.get('trigger_name', '').strip()
    table_name = data.get('table_name', '').strip()
    event = data.get('event', '').strip()
    trigger_body = data.get('trigger_body', '').strip()
    when_condition = data.get('when_condition', '').strip()

    if not all([trigger_name, table_name, event, trigger_body]):
        return jsonify({"error": "trigger_name, table_name, event, and trigger_body are required"})

    try:
        # Build CREATE TRIGGER statement
        when_clause = f"WHEN {when_condition}" if when_condition else ""

        # Clean up the trigger body - remove extra escaping
        clean_trigger_body = trigger_body.replace("\\'", "'")

        create_query = f"""
CREATE TRIGGER `{trigger_name}`
{event} ON `{table_name}`
{when_clause}
BEGIN
    {clean_trigger_body}
END
        """.strip()

        print(f"Executing CREATE TRIGGER query: {create_query}")  # Debug log

        # Execute the query
        result = web_explorer.execute_query(create_query)

        print(f"Trigger creation result: {result}")  # Debug log

        # Handle different response formats
        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    # Refresh schema to include new trigger
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but trigger was created

                    return jsonify({
                        "success": True,
                        "message": f"Trigger '{trigger_name}' created successfully on table '{table_name}'"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', 'Unknown error creating trigger')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    # Success case - no error in result
                    # Refresh schema to include new trigger
                    try:
                        if hasattr(web_explorer, 'explorer') and hasattr(web_explorer.explorer, 'connector'):
                            web_explorer.schema = web_explorer.explorer.connector.get_schema()
                    except Exception as refresh_error:
                        print(f"Schema refresh failed: {refresh_error}")
                        pass  # Schema refresh failed, but trigger was created

                    return jsonify({
                        "success": True,
                        "message": f"Trigger '{trigger_name}' created successfully on table '{table_name}'"
                    })
        else:
            return jsonify({
                "success": False,
                "error": "No response from database"
            })

    except Exception as e:
        print(f"Exception in create_trigger: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/table/<table_name>/duplicates', methods=['POST'])
def find_duplicates(table_name):
    """Find duplicate records"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json() or {}
    columns = data.get('columns', [])
    limit = data.get('limit', 100)

    try:
        # Get table structure
        structure_result = web_explorer.get_table_structure(table_name)
        if not structure_result:
            return jsonify({"error": f"Table '{table_name}' not found"})

        # Handle different structure formats
        if isinstance(structure_result, dict):
            if 'structure' in structure_result:
                structure = structure_result['structure']
            elif 'columns' in structure_result:
                structure = structure_result
            else:
                return jsonify({"error": "Invalid table structure format"})
        else:
            return jsonify({"error": "Could not retrieve table structure"})

        # Use all columns if none specified
        if not columns:
            if 'columns' in structure and isinstance(structure['columns'], list):
                columns = [col['name'] if isinstance(col, dict) else str(col) for col in structure['columns']]
            else:
                return jsonify({"error": "Could not determine table columns"})

        if not columns:
            return jsonify({"error": "No columns found in table"})

        # Validate that specified columns exist in the table
        table_columns = []
        if 'columns' in structure:
            table_columns = [col['name'] if isinstance(col, dict) else str(col) for col in structure['columns']]

        invalid_columns = [col for col in columns if col not in table_columns]
        if invalid_columns:
            return jsonify({"error": f"Columns not found in table: {', '.join(invalid_columns)}"})

        # Build the duplicate detection query
        columns_str = ", ".join(f"`{col}`" for col in columns)  # Use backticks for column names

        duplicate_query = f"""
            SELECT {columns_str}, COUNT(*) as duplicate_count
            FROM `{table_name}`
            GROUP BY {columns_str}
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
            LIMIT {limit}
        """

        print(f"Executing duplicate query: {duplicate_query}")  # Debug log

        # Execute the query
        duplicates_result = web_explorer.execute_query(duplicate_query)

        # Handle different response formats
        duplicates = []
        if duplicates_result:
            if isinstance(duplicates_result, dict):
                if duplicates_result.get('success') and duplicates_result.get('data'):
                    duplicates = duplicates_result['data']
                elif 'error' in duplicates_result:
                    return jsonify({"success": False, "error": duplicates_result['error']})
            elif isinstance(duplicates_result, list):
                if len(duplicates_result) > 0 and duplicates_result[0].get('error'):
                    return jsonify({"success": False, "error": duplicates_result[0]['error']})
                else:
                    duplicates = duplicates_result

        return jsonify({
            "success": True,
            "duplicates": duplicates,
            "total_duplicate_groups": len(duplicates),
            "columns_checked": columns,
            "table_name": table_name
        })

    except Exception as e:
        print(f"Error in find_duplicates for table {table_name}: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({"error": str(e)})

@app.route('/api/table/<table_name>/null-analysis')
def null_analysis(table_name):
    """Analyze NULL values"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        # Get table structure
        structure_result = web_explorer.get_table_structure(table_name)
        if not structure_result or 'structure' not in structure_result:
            return jsonify({"error": f"Table '{table_name}' not found"})

        structure = structure_result['structure']

        # Get total row count - try multiple methods
        total_rows = 0
        try:
            # Method 1: Try to get from structure
            if 'row_count' in structure:
                total_rows = structure['row_count']
            else:
                # Method 2: Query the database directly
                count_query = f"SELECT COUNT(*) as total_count FROM {table_name}"
                count_result = web_explorer.execute_query(count_query)
                if count_result and count_result.get('success') and count_result.get('data'):
                    total_rows = count_result['data'][0].get('total_count', 0)
                elif count_result and len(count_result) > 0 and not count_result[0].get('error'):
                    total_rows = count_result[0].get('total_count', 0)
        except Exception as e:
            print(f"Error getting row count: {e}")
            total_rows = 0

        null_analysis_data = []

        # Analyze each column
        for column in structure.get('columns', []):
            col_name = column['name']

            try:
                # Query for NULL count in this column
                null_query = f"SELECT COUNT(*) as null_count FROM {table_name} WHERE {col_name} IS NULL"
                null_result = web_explorer.execute_query(null_query)

                null_count = 0
                if null_result:
                    if null_result.get('success') and null_result.get('data'):
                        # New format response
                        null_count = null_result['data'][0].get('null_count', 0)
                    elif len(null_result) > 0 and not null_result[0].get('error'):
                        # Old format response
                        null_count = null_result[0].get('null_count', 0)

                non_null_count = max(0, total_rows - null_count)
                null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

                null_analysis_data.append({
                    "column": col_name,
                    "data_type": column.get('type', 'UNKNOWN'),
                    "nullable": column.get('nullable', True),
                    "null_count": null_count,
                    "null_percentage": round(null_percentage, 2),
                    "non_null_count": non_null_count
                })

            except Exception as col_error:
                print(f"Error analyzing column {col_name}: {col_error}")
                # Skip columns that can't be analyzed but don't fail the whole request
                null_analysis_data.append({
                    "column": col_name,
                    "data_type": column.get('type', 'UNKNOWN'),
                    "nullable": column.get('nullable', True),
                    "null_count": 0,
                    "null_percentage": 0,
                    "non_null_count": total_rows,
                    "error": f"Could not analyze: {str(col_error)}"
                })

        return jsonify({
            "success": True,
            "null_analysis": null_analysis_data,
            "total_rows": total_rows,
            "table_name": table_name
        })

    except Exception as e:
        print(f"Error in null_analysis for table {table_name}: {e}")
        return jsonify({"error": str(e)})


# AI Chat API endpoints
# AI Chat API endpoints - Updated for LM Studio
@app.route('/api/ai-chat/models')
def get_ai_models():
    """Get available AI models from LM Studio"""
    return jsonify({
        "success": True,
        "models": AI_CHAT_CONFIG["lm_studio"]["models"],
        "default_model": AI_CHAT_CONFIG["default_model"],
        "base_url": AI_CHAT_CONFIG["lm_studio"]["base_url"]
    })

@app.route('/api/ai-chat/schema')
def get_schema_for_ai():
    """Get database schema for AI context"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    assistant = DatabaseAIAssistant(web_explorer)
    schema = assistant.get_database_schema_context()

    return jsonify({
        "success": True,
        "schema": schema,
        "formatted": True
    })

@app.route('/api/ai-chat/query', methods=['POST'])
def ai_chat_query():
    """Main AI chat endpoint using LM Studio"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    user_question = data.get('question', '').strip()
    model_id = data.get('model_id', AI_CHAT_CONFIG["default_model"])
    lm_studio_url = data.get('lm_studio_url')  # Allow custom LM Studio URL
    chat_mode = data.get('mode', 'auto')
    execute_sql = data.get('execute_sql', False)

    if not user_question:
        return jsonify({"error": "No question provided"})

    try:
        assistant = DatabaseAIAssistant(web_explorer, lm_studio_url)

        # Determine the best mode based on the question
        if chat_mode == 'auto':
            question_lower = user_question.lower()
            if any(word in question_lower for word in ['select', 'query', 'find', 'show', 'get', 'list', 'count']):
                chat_mode = 'sql'
            elif any(word in question_lower for word in ['analyze', 'trend', 'pattern', 'insight', 'statistics']):
                chat_mode = 'analysis'
            else:
                chat_mode = 'chat'

        # Create appropriate prompt
        system_message, user_message = assistant.create_ai_prompt(user_question, chat_mode)

        print(f"AI Chat - Mode: {chat_mode}, Model: {model_id}")

        # Try the specified model first, then fallback models
        models_to_try = [model_id]

        # Add fallback models if the primary fails
        for model in AI_CHAT_CONFIG["lm_studio"]["models"]:
            if model["model_id"] != model_id:
                models_to_try.append(model["model_id"])

        ai_result = None
        successful_model = None

        for model in models_to_try:
            print(f"Trying model: {model}")
            ai_result = assistant.call_lm_studio_model(system_message, user_message, model)
            if ai_result.get('success'):
                successful_model = model
                break
            else:
                print(f"Model {model} failed: {ai_result.get('error')}")

        if not ai_result or not ai_result.get('success'):
            return jsonify({
                "success": False,
                "error": "All AI models failed",
                "details": ai_result.get('error') if ai_result else "No response",
                "models_tried": models_to_try,
                "suggestion": "Please ensure LM Studio is running and a model is loaded."
            })

        ai_response = ai_result.get('response', '')
        model_used = successful_model or model_id

        # Try to extract and execute SQL if requested
        sql_query = None
        query_results = None

        if chat_mode in ['sql', 'auto'] or execute_sql:
            sql_query = assistant.extract_sql_from_response(ai_response)

            if sql_query and execute_sql:
                try:
                    query_results = web_explorer.execute_query(sql_query)
                    print(f"AI-generated SQL executed: {sql_query}")
                except Exception as sql_error:
                    query_results = {"error": str(sql_error)}

        # Build response
        response = {
            "success": True,
            "ai_response": ai_response,
            "question": user_question,
            "model_used": model_used,
            "mode": chat_mode,
            "sql_query": sql_query,
            "query_results": query_results if execute_sql else None,
            "usage": ai_result.get("usage", {}),
            "finish_reason": ai_result.get("finish_reason", "unknown"),
            "suggestions": []
        }

        # Add helpful suggestions
        if sql_query and not execute_sql:
            response["suggestions"].append("Set 'execute_sql': true to run this query automatically")

        if chat_mode == 'sql' and not sql_query:
            response["suggestions"].append("Try rephrasing your question to be more specific about what data you want")

        return jsonify(response)

    except Exception as e:
        print(f"Error in ai_chat_query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/ai-chat/conversation', methods=['POST'])
def ai_conversation():
    """Multi-turn conversation with context using LM Studio"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    user_message = data.get('message', '').strip()
    conversation_history = data.get('history', [])
    model_id = data.get('model_id', AI_CHAT_CONFIG["default_model"])
    lm_studio_url = data.get('lm_studio_url')

    if not user_message:
        return jsonify({"error": "No message provided"})

    try:
        assistant = DatabaseAIAssistant(web_explorer, lm_studio_url)

        # Build conversation context
        schema = assistant.get_database_schema_context()

        system_message = f"""You are a database assistant with access to this schema:

{schema}

Maintain context from the conversation history and provide helpful responses about the database."""

        # Build conversation context from history
        conversation_context = ""
        for entry in conversation_history[-5:]:  # Last 5 exchanges
            conversation_context += f"\nPrevious User: {entry.get('user', '')}"
            conversation_context += f"\nPrevious Assistant: {entry.get('assistant', '')}"

        full_user_message = f"{conversation_context}\n\nCurrent User: {user_message}"

        # Call LM Studio model
        ai_result = assistant.call_lm_studio_model(system_message, full_user_message, model_id)

        if not ai_result.get('success'):
            return jsonify({
                "success": False,
                "error": "AI model call failed",
                "details": ai_result.get('error')
            })

        ai_response = ai_result.get('response', '')

        # Check if response contains SQL and offer to execute
        sql_query = assistant.extract_sql_from_response(ai_response)

        return jsonify({
            "success": True,
            "message": ai_response,
            "sql_detected": sql_query is not None,
            "sql_query": sql_query,
            "model_used": model_id,
            "context_length": len(conversation_history),
            "usage": ai_result.get("usage", {}),
            "finish_reason": ai_result.get("finish_reason", "unknown")
        })

    except Exception as e:
        print(f"Error in ai_conversation: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/ai-chat/explain-schema', methods=['POST'])
def ai_explain_schema():
    """AI explanation of database schema using LM Studio"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    focus_area = data.get('focus', 'overview')  # overview, relationships, tables, performance
    model_id = data.get('model_id', AI_CHAT_CONFIG["default_model"])
    lm_studio_url = data.get('lm_studio_url')

    try:
        assistant = DatabaseAIAssistant(web_explorer, lm_studio_url)
        schema = assistant.get_database_schema_context()

        system_messages = {
            'overview': "You are a database expert. Explain database schemas in simple, clear terms.",
            'relationships': "You are a database expert. Analyze and explain table relationships and foreign key connections.",
            'tables': "You are a database expert. Describe the purpose and structure of database tables.",
            'performance': "You are a database performance expert. Analyze schemas for optimization opportunities."
        }

        user_messages = {
            'overview': f"Explain this database schema in simple terms. What kind of system is this and what does it track?\n\n{schema}",
            'relationships': f"Analyze the relationships in this database schema. Explain how the tables connect and what the foreign keys represent.\n\n{schema}",
            'tables': f"Describe each table in this database. What data does each table store and what is its purpose?\n\n{schema}",
            'performance': f"Analyze this database schema for performance. What are the potential performance bottlenecks and optimization opportunities?\n\n{schema}"
        }

        system_message = system_messages.get(focus_area, system_messages['overview'])
        user_message = user_messages.get(focus_area, user_messages['overview'])

        ai_result = assistant.call_lm_studio_model(system_message, user_message, model_id)

        if not ai_result.get('success'):
            return jsonify({
                "success": False,
                "error": "AI model call failed",
                "details": ai_result.get('error')
            })

        return jsonify({
            "success": True,
            "explanation": ai_result.get('response', ''),
            "focus_area": focus_area,
            "model_used": model_id,
            "usage": ai_result.get("usage", {}),
            "finish_reason": ai_result.get("finish_reason", "unknown")
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/ai-chat/suggest-queries', methods=['POST'])
def ai_suggest_queries():
    """AI-generated query suggestions using LM Studio"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    query_type = data.get('type', 'analysis')  # analysis, reporting, maintenance
    model_id = data.get('model_id', AI_CHAT_CONFIG["default_model"])
    lm_studio_url = data.get('lm_studio_url')

    try:
        assistant = DatabaseAIAssistant(web_explorer, lm_studio_url)
        schema = assistant.get_database_schema_context()

        system_message = "You are a SQL expert. Generate useful SQL queries based on database schemas."

        user_messages = {
            'analysis': f"Based on this database schema, suggest 5 interesting analytical SQL queries that would give business insights. Provide each query with a brief explanation.\n\n{schema}",
            'reporting': f"Suggest 5 useful reporting queries for this database. Focus on queries that would be useful for regular reports and dashboards.\n\n{schema}",
            'maintenance': f"Suggest 5 database maintenance and health check queries. Include queries to check data quality, performance, and integrity.\n\n{schema}"
        }

        user_message = user_messages.get(query_type, user_messages['analysis'])

        # Use a model optimized for code/SQL generation if available
        if query_type in ['analysis', 'reporting']:
            # Try to use DeepSeek Coder for SQL tasks
            if any(m["model_id"] == "deepseek-coder-v2-lite-instruct" for m in AI_CHAT_CONFIG["lm_studio"]["models"]):
                model_id = "deepseek-coder-v2-lite-instruct"

        ai_result = assistant.call_lm_studio_model(system_message, user_message, model_id, max_tokens=2048)

        if not ai_result.get('success'):
            return jsonify({
                "success": False,
                "error": "AI model call failed",
                "details": ai_result.get('error')
            })

        return jsonify({
            "success": True,
            "suggestions": ai_result.get('response', ''),
            "query_type": query_type,
            "model_used": model_id,
            "usage": ai_result.get("usage", {}),
            "finish_reason": ai_result.get("finish_reason", "unknown")
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/ai-chat/validate-connection', methods=['POST'])
def validate_lm_studio_connection():
    """Validate LM Studio connection and list available models"""
    data = request.get_json()
    lm_studio_url = data.get('lm_studio_url', AI_CHAT_CONFIG["lm_studio"]["base_url"])

    try:
        # Test connection by calling the models endpoint
        models_url = f"{lm_studio_url}/models"

        print(f"Testing LM Studio connection at: {models_url}")

        response = requests.get(models_url, timeout=10)

        if response.status_code == 200:
            try:
                models_data = response.json()
                available_models = []

                if "data" in models_data:
                    for model in models_data["data"]:
                        available_models.append({
                            "id": model.get("id", "unknown"),
                            "object": model.get("object", "model"),
                            "created": model.get("created", 0),
                            "owned_by": model.get("owned_by", "unknown")
                        })

                return jsonify({
                    "valid": True,
                    "connected": True,
                    "base_url": lm_studio_url,
                    "available_models": available_models,
                    "total_models": len(available_models),
                    "message": f"Successfully connected to LM Studio at {lm_studio_url}"
                })

            except json.JSONDecodeError:
                return jsonify({
                    "valid": True,
                    "connected": True,
                    "base_url": lm_studio_url,
                    "available_models": [],
                    "message": "Connected but could not parse models list"
                })
        else:
            return jsonify({
                "valid": False,
                "connected": False,
                "error": f"LM Studio responded with status {response.status_code}",
                "suggestion": "Check if LM Studio is running and accessible"
            })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "valid": False,
            "connected": False,
            "error": f"Cannot connect to LM Studio at {lm_studio_url}",
            "suggestion": "Ensure LM Studio is running on the specified URL"
        })
    except requests.exceptions.Timeout:
        return jsonify({
            "valid": False,
            "connected": False,
            "error": "Connection timeout",
            "suggestion": "LM Studio may be starting up or processing requests"
        })
    except Exception as e:
        return jsonify({
            "valid": False,
            "connected": False,
            "error": str(e),
            "suggestion": "Check LM Studio installation and configuration"
        })

@app.route('/api/ai-chat/test-model', methods=['POST'])
def test_lm_studio_model():
    """Test a specific model in LM Studio"""
    data = request.get_json()
    model_id = data.get('model_id', AI_CHAT_CONFIG["default_model"])
    lm_studio_url = data.get('lm_studio_url')

    try:
        assistant = DatabaseAIAssistant(web_explorer, lm_studio_url)

        # Simple test prompt
        system_message = "You are a helpful assistant. Respond briefly and clearly."
        user_message = "Generate a simple SQL SELECT statement to get all records from a 'users' table."

        result = assistant.call_lm_studio_model(system_message, user_message, model_id, max_tokens=100)

        return jsonify({
            "success": result.get('success', False),
            "model_id": model_id,
            "response": result.get('response', ''),
            "error": result.get('error') if not result.get('success') else None,
            "usage": result.get('usage', {}),
            "finish_reason": result.get('finish_reason', 'unknown'),
            "base_url": assistant.base_url
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "model_id": model_id,
            "error": str(e)
        })

@app.route('/api/ai-chat/debug-lm-studio', methods=['POST'])
def debug_lm_studio():
    """Debug LM Studio connectivity and model availability"""
    data = request.get_json()
    lm_studio_url = data.get('lm_studio_url', AI_CHAT_CONFIG["lm_studio"]["base_url"])
    model_id = data.get('model_id', AI_CHAT_CONFIG["default_model"])

    results = {}

    # Test 1: Basic connectivity
    try:
        response = requests.get(f"{lm_studio_url}/models", timeout=10)
        results["connectivity_test"] = {
            "status": response.status_code,
            "success": response.status_code == 200,
            "response": response.text[:300] if response.text else "No response body"
        }
    except Exception as e:
        results["connectivity_test"] = {"error": str(e)}

    # Test 2: Model availability
    try:
        response = requests.get(f"{lm_studio_url}/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            available_models = []
            if "data" in models_data:
                available_models = [model.get("id", "unknown") for model in models_data["data"]]

            results["models_test"] = {
                "success": True,
                "available_models": available_models,
                "target_model_available": model_id in available_models,
                "total_models": len(available_models)
            }
        else:
            results["models_test"] = {
                "success": False,
                "error": f"Models endpoint returned {response.status_code}"
            }
    except Exception as e:
        results["models_test"] = {"error": str(e)}

    # Test 3: Simple chat completion
    try:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, I am working!' in exactly those words."}
            ],
            "temperature": 0.1,
            "max_tokens": 20,
            "stream": False
        }

        response = requests.post(
            f"{lm_studio_url}/chat/completions",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            chat_data = response.json()
            if "choices" in chat_data and len(chat_data["choices"]) > 0:
                content = chat_data["choices"][0].get("message", {}).get("content", "")
                results["chat_test"] = {
                    "success": True,
                    "response": content,
                    "usage": chat_data.get("usage", {}),
                    "finish_reason": chat_data["choices"][0].get("finish_reason", "unknown")
                }
            else:
                results["chat_test"] = {
                    "success": False,
                    "error": "Invalid chat response format",
                    "response": chat_data
                }
        else:
            results["chat_test"] = {
                "success": False,
                "status": response.status_code,
                "error": response.text[:300]
            }
    except Exception as e:
        results["chat_test"] = {"error": str(e)}

    # Test 4: Database schema integration
    if web_explorer.connected:
        try:
            assistant = DatabaseAIAssistant(web_explorer, lm_studio_url)
            schema = assistant.get_database_schema_context()
            results["schema_test"] = {
                "success": True,
                "schema_length": len(schema),
                "has_tables": "Table:" in schema,
                "preview": schema[:200] + "..." if len(schema) > 200 else schema
            }
        except Exception as e:
            results["schema_test"] = {"error": str(e)}
    else:
        results["schema_test"] = {"error": "No database connected"}

    return jsonify({
        "success": True,
        "debug_results": results,
        "lm_studio_url": lm_studio_url,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat()
    })


# Enhanced Schema Management Backend APIs
# Add these routes to your web_db_explorer.py file

# ===== SCHEMA INDEXES MANAGEMENT =====

@app.route('/api/schema/indexes')
def get_all_indexes():
    """Get all indexes across all tables"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Getting all indexes...")  # Debug log

        indexes = []

        # Method 1: Get user-created indexes only (exclude auto-generated ones)
        all_indexes_query = """
                            SELECT name, tbl_name, sql
                            FROM sqlite_master
                            WHERE type='index'
                              AND name NOT LIKE 'sqlite_%'
                              AND sql IS NOT NULL \
                            """
        all_indexes_result = web_explorer.execute_query(all_indexes_query)

        print(f"Direct sqlite_master query result: {all_indexes_result}")  # Debug log

        if all_indexes_result:
            index_rows = []
            if isinstance(all_indexes_result, dict) and all_indexes_result.get('success') and all_indexes_result.get('data'):
                index_rows = all_indexes_result['data']
            elif isinstance(all_indexes_result, list) and not (len(all_indexes_result) > 0 and all_indexes_result[0].get('error')):
                index_rows = all_indexes_result

            print(f"Found {len(index_rows)} user-created indexes in sqlite_master")  # Debug log

            for idx_row in index_rows:
                idx_name = idx_row.get('name', '')
                table_name = idx_row.get('tbl_name', '')
                index_sql = idx_row.get('sql', '')

                if not idx_name or not table_name:
                    continue

                print(f"Processing index: {idx_name} on table {table_name}")  # Debug log

                # Get index columns using PRAGMA index_info
                idx_columns = []
                unique = False

                try:
                    # Get index details using PRAGMA
                    idx_info_query = f"PRAGMA index_info('{idx_name}')"
                    idx_info_result = web_explorer.execute_query(idx_info_query)

                    if idx_info_result:
                        idx_info_data = []
                        if isinstance(idx_info_result, dict) and idx_info_result.get('success') and idx_info_result.get('data'):
                            idx_info_data = idx_info_result['data']
                        elif isinstance(idx_info_result, list) and not (len(idx_info_result) > 0 and idx_info_result[0].get('error')):
                            idx_info_data = idx_info_result

                        idx_columns = [col.get('name', '') for col in idx_info_data if col.get('name')]

                    # Check if index is unique by examining the SQL
                    if index_sql and 'UNIQUE' in index_sql.upper():
                        unique = True

                except Exception as idx_detail_error:
                    print(f"Error getting details for index {idx_name}: {idx_detail_error}")
                    # Fallback: try to parse columns from SQL
                    if index_sql and '(' in index_sql:
                        import re
                        match = re.search(r'\(([^)]+)\)', index_sql)
                        if match:
                            columns_str = match.group(1)
                            idx_columns = [col.strip().strip('`"\'') for col in columns_str.split(',')]

                # Determine if it's a primary key index
                is_primary = False
                if index_sql:
                    is_primary = 'PRIMARY KEY' in index_sql.upper()

                # If we couldn't get columns, set a default
                if not idx_columns:
                    idx_columns = ['unknown']

                index_info = {
                    "name": idx_name,
                    "table_name": table_name,
                    "unique": unique,
                    "primary": is_primary,
                    "columns": idx_columns,
                    "type": "BTREE",  # SQLite default
                    "sql": index_sql,
                    "source": "sqlite_master"
                }

                indexes.append(index_info)
                print(f"Added index: {idx_name} with columns {idx_columns}")  # Debug log

        print(f"Found {len(indexes)} user-created indexes total")  # Debug log

        return jsonify({
            "success": True,
            "indexes": indexes,
            "total_count": len(indexes),
            "debug_info": {
                "method_used": "sqlite_master_user_only",
                "raw_count": len(indexes),
                "excluded_auto_indexes": True
            }
        })

    except Exception as e:
        print(f"Error in get_all_indexes: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})



@app.route('/api/schema/indexes/<index_name>')
def get_index_details(index_name):
    """Get detailed information about a specific index"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print(f"Getting details for index: {index_name}")  # Debug log

        # Get index information from sqlite_master
        index_query = f"SELECT * FROM sqlite_master WHERE type='index' AND name='{index_name}'"
        index_result = web_explorer.execute_query(index_query)

        if not index_result:
            return jsonify({"error": f"Index '{index_name}' not found"})

        index_data = None
        if isinstance(index_result, dict) and index_result.get('success') and index_result.get('data'):
            if len(index_result['data']) > 0:
                index_data = index_result['data'][0]
        elif isinstance(index_result, list) and not (len(index_result) > 0 and index_result[0].get('error')):
            if len(index_result) > 0:
                index_data = index_result[0]

        if not index_data:
            return jsonify({"error": f"Index '{index_name}' not found"})

        table_name = index_data.get('tbl_name', '')

        # Get index columns using PRAGMA index_info
        columns = []
        try:
            idx_info_query = f"PRAGMA index_info(`{index_name}`)"
            idx_info_result = web_explorer.execute_query(idx_info_query)

            if idx_info_result:
                idx_info_data = []
                if isinstance(idx_info_result, dict) and idx_info_result.get('success') and idx_info_result.get('data'):
                    idx_info_data = idx_info_result['data']
                elif isinstance(idx_info_result, list) and not (len(idx_info_result) > 0 and idx_info_result[0].get('error')):
                    idx_info_data = idx_info_result

                columns = [col.get('name', '') for col in idx_info_data if col.get('name')]
        except Exception as col_error:
            print(f"Error getting columns for index {index_name}: {col_error}")

        # Check if index is unique
        unique = False
        if table_name:
            try:
                unique_query = f"PRAGMA index_list({table_name})"
                unique_result = web_explorer.execute_query(unique_query)

                if unique_result:
                    unique_data = []
                    if isinstance(unique_result, dict) and unique_result.get('success') and unique_result.get('data'):
                        unique_data = unique_result['data']
                    elif isinstance(unique_result, list) and not (len(unique_result) > 0 and unique_result[0].get('error')):
                        unique_data = unique_result

                    for idx_info in unique_data:
                        if idx_info.get('name') == index_name:
                            unique = idx_info.get('unique', 0) == 1
                            break
            except Exception as unique_error:
                print(f"Error checking if index is unique: {unique_error}")

        # Estimate index statistics (basic implementation)
        statistics = {
            "size_mb": "Unknown",
            "cardinality": "Unknown",
            "usage_count": "Unknown"
        }

        # Try to get row count for cardinality estimation
        if table_name and columns:
            try:
                # Simple cardinality estimation
                first_col = columns[0] if columns else None
                if first_col:
                    cardinality_query = f"SELECT COUNT(DISTINCT `{first_col}`) as distinct_count FROM `{table_name}`"
                    card_result = web_explorer.execute_query(cardinality_query)

                    if card_result:
                        if isinstance(card_result, dict) and card_result.get('success') and card_result.get('data'):
                            distinct_count = card_result['data'][0].get('distinct_count', 0)
                            statistics["cardinality"] = distinct_count
                        elif isinstance(card_result, list) and not (len(card_result) > 0 and card_result[0].get('error')):
                            distinct_count = card_result[0].get('distinct_count', 0)
                            statistics["cardinality"] = distinct_count
            except Exception as stats_error:
                print(f"Error calculating index statistics: {stats_error}")

        index_details = {
            "name": index_name,
            "table_name": table_name,
            "columns": columns,
            "unique": unique,
            "primary": 'PRIMARY KEY' in (index_data.get('sql', '') or '').upper(),
            "type": "BTREE",
            "sql": index_data.get('sql'),
            "statistics": statistics
        }

        return jsonify({
            "success": True,
            "index": index_details
        })

    except Exception as e:
        print(f"Error in get_index_details: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/indexes/<index_name>', methods=['DELETE'])
def drop_index_enhanced(index_name):
    """Drop an index (enhanced version)"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        # First check if index exists and is not a primary key index
        index_query = f"SELECT sql FROM sqlite_master WHERE type='index' AND name='{index_name}'"
        index_result = web_explorer.execute_query(index_query)

        index_exists = False
        is_primary = False

        if index_result:
            if isinstance(index_result, dict) and index_result.get('success') and index_result.get('data'):
                if len(index_result['data']) > 0:
                    index_exists = True
                    sql = index_result['data'][0].get('sql', '')
                    is_primary = 'PRIMARY KEY' in sql.upper() if sql else False
            elif isinstance(index_result, list) and not (len(index_result) > 0 and index_result[0].get('error')):
                if len(index_result) > 0:
                    index_exists = True
                    sql = index_result[0].get('sql', '')
                    is_primary = 'PRIMARY KEY' in sql.upper() if sql else False

        if not index_exists:
            return jsonify({
                "success": False,
                "error": f"Index '{index_name}' not found"
            })

        if is_primary:
            return jsonify({
                "success": False,
                "error": f"Cannot drop primary key index '{index_name}'"
            })

        # Drop the index
        drop_query = f"DROP INDEX `{index_name}`"
        print(f"Executing DROP INDEX query: {drop_query}")  # Debug log

        result = web_explorer.execute_query(drop_query)

        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    return jsonify({
                        "success": True,
                        "message": f"Index '{index_name}' dropped successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', 'Unknown error dropping index')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    return jsonify({
                        "success": True,
                        "message": f"Index '{index_name}' dropped successfully"
                    })

        return jsonify({
            "success": False,
            "error": "No response from database"
        })

    except Exception as e:
        print(f"Error in drop_index_enhanced: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ===== SCHEMA VIEWS MANAGEMENT =====

@app.route('/api/schema/views')
def get_all_views():
    """Get all views in the database"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Getting all views...")  # Debug log

        # Get all views from sqlite_master
        views_query = "SELECT name, sql FROM sqlite_master WHERE type='view'"
        views_result = web_explorer.execute_query(views_query)

        views = []
        if views_result:
            views_data = []
            if isinstance(views_result, dict) and views_result.get('success') and views_result.get('data'):
                views_data = views_result['data']
            elif isinstance(views_result, list) and not (len(views_result) > 0 and views_result[0].get('error')):
                views_data = views_result

            for view_row in views_data:
                view_name = view_row.get('name', '')
                view_sql = view_row.get('sql', '')

                # Try to extract base tables from SQL (basic implementation)
                base_tables = extract_tables_from_sql(view_sql)

                # Try to get column count
                column_count = 0
                try:
                    columns_query = f"PRAGMA table_info(`{view_name}`)"
                    columns_result = web_explorer.execute_query(columns_query)

                    if columns_result:
                        if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                            column_count = len(columns_result['data'])
                        elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                            column_count = len(columns_result)
                except Exception as col_error:
                    print(f"Error getting column count for view {view_name}: {col_error}")

                view_info = {
                    "name": view_name,
                    "sql": view_sql,
                    "base_tables": ", ".join(base_tables) if base_tables else "Multiple",
                    "column_count": column_count,
                    "created_date": "Unknown"  # SQLite doesn't store creation dates
                }

                views.append(view_info)

        print(f"Found {len(views)} views")  # Debug log

        return jsonify({
            "success": True,
            "views": views,
            "total_count": len(views)
        })

    except Exception as e:
        print(f"Error in get_all_views: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/views/<view_name>')
def get_view_details(view_name):
    """Get detailed information about a specific view"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print(f"Getting details for view: {view_name}")  # Debug log

        # Get view information from sqlite_master
        view_query = f"SELECT * FROM sqlite_master WHERE type='view' AND name='{view_name}'"
        view_result = web_explorer.execute_query(view_query)

        if not view_result:
            return jsonify({"error": f"View '{view_name}' not found"})

        view_data = None
        if isinstance(view_result, dict) and view_result.get('success') and view_result.get('data'):
            if len(view_result['data']) > 0:
                view_data = view_result['data'][0]
        elif isinstance(view_result, list) and not (len(view_result) > 0 and view_result[0].get('error')):
            if len(view_result) > 0:
                view_data = view_result[0]

        if not view_data:
            return jsonify({"error": f"View '{view_name}' not found"})

        view_sql = view_data.get('sql', '')

        # Extract base tables
        base_tables = extract_tables_from_sql(view_sql)

        # Get columns information
        columns = []
        try:
            columns_query = f"PRAGMA table_info(`{view_name}`)"
            columns_result = web_explorer.execute_query(columns_query)

            if columns_result:
                columns_data = []
                if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                    columns_data = columns_result['data']
                elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                    columns_data = columns_result

                columns = [{"name": col.get('name', ''), "type": col.get('type', '')} for col in columns_data]
        except Exception as col_error:
            print(f"Error getting columns for view {view_name}: {col_error}")

        view_details = {
            "name": view_name,
            "sql": view_sql,
            "base_tables": ", ".join(base_tables) if base_tables else "Multiple",
            "columns": columns,
            "column_count": len(columns),
            "created_date": "Unknown"
        }

        return jsonify({
            "success": True,
            "view": view_details
        })

    except Exception as e:
        print(f"Error in get_view_details: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/views/<view_name>', methods=['DELETE'])
def drop_view_enhanced(view_name):
    """Drop a view (enhanced version)"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        # First check if view exists
        view_query = f"SELECT name FROM sqlite_master WHERE type='view' AND name='{view_name}'"
        view_result = web_explorer.execute_query(view_query)

        view_exists = False
        if view_result:
            if isinstance(view_result, dict) and view_result.get('success') and view_result.get('data'):
                view_exists = len(view_result['data']) > 0
            elif isinstance(view_result, list) and not (len(view_result) > 0 and view_result[0].get('error')):
                view_exists = len(view_result) > 0

        if not view_exists:
            return jsonify({
                "success": False,
                "error": f"View '{view_name}' not found"
            })

        # Drop the view
        drop_query = f"DROP VIEW `{view_name}`"
        print(f"Executing DROP VIEW query: {drop_query}")  # Debug log

        result = web_explorer.execute_query(drop_query)

        if result:
            if isinstance(result, dict):
                if result.get('success'):
                    return jsonify({
                        "success": True,
                        "message": f"View '{view_name}' dropped successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": result.get('error', 'Unknown error dropping view')
                    })
            elif isinstance(result, list):
                if len(result) > 0 and result[0].get('error'):
                    return jsonify({
                        "success": False,
                        "error": result[0]['error']
                    })
                else:
                    return jsonify({
                        "success": True,
                        "message": f"View '{view_name}' dropped successfully"
                    })

        return jsonify({
            "success": False,
            "error": "No response from database"
        })

    except Exception as e:
        print(f"Error in drop_view_enhanced: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ===== SCHEMA CONSTRAINTS MANAGEMENT =====

@app.route('/api/schema/constraints')
def get_all_constraints():
    """Get all constraints in the database"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Getting all constraints...")  # Debug log

        constraints = []

        # Get all user tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        # Analyze each table for constraints
        for table_name in table_names:
            try:
                # Get table schema from sqlite_master to parse constraints
                schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                schema_result = web_explorer.execute_query(schema_query)

                if schema_result:
                    create_sql = ""
                    if isinstance(schema_result, dict) and schema_result.get('success') and schema_result.get('data'):
                        if len(schema_result['data']) > 0:
                            create_sql = schema_result['data'][0].get('sql', '')
                    elif isinstance(schema_result, list) and not (len(schema_result) > 0 and schema_result[0].get('error')):
                        if len(schema_result) > 0:
                            create_sql = schema_result[0].get('sql', '')

                    if create_sql:
                        # Parse constraints from CREATE TABLE statement
                        table_constraints = parse_constraints_from_sql(create_sql, table_name)
                        constraints.extend(table_constraints)

                # Get foreign key constraints
                fk_query = f"PRAGMA foreign_key_list({table_name})"
                fk_result = web_explorer.execute_query(fk_query)

                if fk_result:
                    fk_data = []
                    if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                        fk_data = fk_result['data']
                    elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                        fk_data = fk_result

                    for fk in fk_data:
                        constraint_info = {
                            "name": f"fk_{table_name}_{fk.get('from', '')}",
                            "table_name": table_name,
                            "type": "FOREIGN KEY",
                            "columns": [fk.get('from', '')],
                            "referenced_table": fk.get('table', ''),
                            "referenced_columns": [fk.get('to', '')],
                            "on_delete": fk.get('on_delete', 'NO ACTION'),
                            "on_update": fk.get('on_update', 'NO ACTION')
                        }
                        constraints.append(constraint_info)

            except Exception as table_error:
                print(f"Error analyzing constraints for table {table_name}: {table_error}")
                continue

        print(f"Found {len(constraints)} constraints")  # Debug log

        return jsonify({
            "success": True,
            "constraints": constraints,
            "total_count": len(constraints)
        })

    except Exception as e:
        print(f"Error in get_all_constraints: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/foreign-keys')
def get_all_foreign_keys():
    """Get all foreign keys in the database"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Getting all foreign keys...")  # Debug log

        foreign_keys = []

        # First, ensure foreign keys are enabled
        try:
            web_explorer.execute_query("PRAGMA foreign_keys = ON")
        except:
            pass

        # Get all user tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        print(f"Found tables for foreign key analysis: {table_names}")  # Debug log

        # Method 1: Parse CREATE TABLE statements directly (most reliable)
        for table_name in table_names:
            try:
                print(f"Analyzing foreign keys for table: {table_name}")

                # Get CREATE TABLE statement
                create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                create_result = web_explorer.execute_query(create_query)

                create_sql = ""
                if create_result:
                    if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                        if len(create_result['data']) > 0:
                            create_sql = create_result['data'][0].get('sql', '')
                    elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                        if len(create_result) > 0:
                            create_sql = create_result[0].get('sql', '')

                print(f"CREATE SQL for {table_name}: {create_sql[:200]}...")  # Debug log

                if create_sql:
                    # Method 1: Look for FOREIGN KEY constraints
                    import re

                    # Pattern 1: FOREIGN KEY (column) REFERENCES table (column)
                    fk_pattern1 = r'FOREIGN KEY\s*\(\s*([^)]+)\s*\)\s*REFERENCES\s+([^(\s]+)\s*\(\s*([^)]+)\s*\)'
                    matches1 = re.findall(fk_pattern1, create_sql, re.IGNORECASE | re.MULTILINE)

                    for match in matches1:
                        from_column = match[0].strip().strip('`"\'')
                        to_table = match[1].strip().strip('`"\'')
                        to_column = match[2].strip().strip('`"\'')

                        fk_info = {
                            "table_name": table_name,
                            "column": from_column,
                            "referenced_table": to_table,
                            "referenced_column": to_column,
                            "on_delete": "CASCADE",  # Default from your schema
                            "on_update": "CASCADE",
                            "match": "NONE",
                            "source": "foreign_key_constraint"
                        }
                        foreign_keys.append(fk_info)
                        print(f"Found FK (FOREIGN KEY): {table_name}.{from_column} -> {to_table}.{to_column}")

                    # Pattern 2: Column-level REFERENCES
                    # Look for: column_name data_type REFERENCES table(column)
                    fk_pattern2 = r'(\w+)\s+[^,\n]*?\s+REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)'
                    matches2 = re.findall(fk_pattern2, create_sql, re.IGNORECASE | re.MULTILINE)

                    for match in matches2:
                        from_column = match[0].strip()
                        to_table = match[1].strip()
                        to_column = match[2].strip()

                        # Avoid duplicates
                        duplicate = any(
                            fk['table_name'] == table_name and
                            fk['column'] == from_column and
                            fk['referenced_table'] == to_table
                            for fk in foreign_keys
                        )

                        if not duplicate:
                            fk_info = {
                                "table_name": table_name,
                                "column": from_column,
                                "referenced_table": to_table,
                                "referenced_column": to_column,
                                "on_delete": "CASCADE",  # Default from your schema
                                "on_update": "CASCADE",
                                "match": "NONE",
                                "source": "column_references"
                            }
                            foreign_keys.append(fk_info)
                            print(f"Found FK (REFERENCES): {table_name}.{from_column} -> {to_table}.{to_column}")

            except Exception as table_error:
                print(f"Error getting foreign keys for table {table_name}: {table_error}")
                continue

        # Method 2: Fallback to PRAGMA if no FKs found via SQL parsing
        if len(foreign_keys) == 0:
            print("No FKs found via SQL parsing, trying PRAGMA fallback...")

            for table_name in table_names:
                try:
                    fk_query = f"PRAGMA foreign_key_list('{table_name}')"
                    fk_result = web_explorer.execute_query(fk_query)

                    if fk_result:
                        fk_data = []
                        if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                            fk_data = fk_result['data']
                        elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                            fk_data = fk_result

                        for fk in fk_data:
                            fk_info = {
                                "table_name": table_name,
                                "column": fk.get('from', ''),
                                "referenced_table": fk.get('table', ''),
                                "referenced_column": fk.get('to', ''),
                                "on_delete": fk.get('on_delete', 'NO ACTION'),
                                "on_update": fk.get('on_update', 'NO ACTION'),
                                "match": fk.get('match', 'NONE'),
                                "source": "pragma_fallback"
                            }
                            foreign_keys.append(fk_info)

                except Exception as pragma_error:
                    print(f"PRAGMA failed for {table_name}: {pragma_error}")
                    continue

        print(f"Found {len(foreign_keys)} total foreign keys")  # Debug log

        return jsonify({
            "success": True,
            "foreign_keys": foreign_keys,
            "total_count": len(foreign_keys),
            "debug_info": {
                "tables_analyzed": len(table_names),
                "method": "sql_parsing_and_pragma_fallback"
            }
        })

    except Exception as e:
        print(f"Error in get_all_foreign_keys: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ===== UTILITY FUNCTIONS =====

def extract_tables_from_sql(sql):
    """Extract table names from SQL statement (basic implementation)"""
    if not sql:
        return []

    import re

    # Remove comments and normalize whitespace
    sql = re.sub(r'--.*?\n', ' ', sql)
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    sql = re.sub(r'\s+', ' ', sql)

    # Look for FROM and JOIN clauses
    tables = []

    # Find tables after FROM
    from_matches = re.findall(r'\bFROM\s+`?(\w+)`?', sql, re.IGNORECASE)
    tables.extend(from_matches)

    # Find tables after JOIN
    join_matches = re.findall(r'\bJOIN\s+`?(\w+)`?', sql, re.IGNORECASE)
    tables.extend(join_matches)

    # Remove duplicates and system tables
    unique_tables = list(set(tables))
    user_tables = [t for t in unique_tables if not t.startswith('sqlite_')]

    return user_tables


def parse_constraints_from_sql(create_sql, table_name):
    """Parse constraints from CREATE TABLE SQL (basic implementation)"""
    constraints = []

    if not create_sql:
        return constraints

    import re

    # Extract the content between parentheses
    match = re.search(r'\((.*)\)', create_sql, re.DOTALL)
    if not match:
        return constraints

    content = match.group(1)

    # Look for PRIMARY KEY constraints
    pk_matches = re.findall(r'PRIMARY\s+KEY\s*\(\s*([^)]+)\s*\)', content, re.IGNORECASE)
    for pk_match in pk_matches:
        columns = [col.strip().strip('`"\'') for col in pk_match.split(',')]
        constraints.append({
            "name": f"pk_{table_name}",
            "table_name": table_name,
            "type": "PRIMARY KEY",
            "columns": columns
        })

    # Look for UNIQUE constraints
    unique_matches = re.findall(r'UNIQUE\s*\(\s*([^)]+)\s*\)', content, re.IGNORECASE)
    for i, unique_match in enumerate(unique_matches):
        columns = [col.strip().strip('`"\'') for col in unique_match.split(',')]
        constraints.append({
            "name": f"unique_{table_name}_{i+1}",
            "table_name": table_name,
            "type": "UNIQUE",
            "columns": columns
        })

    # Look for CHECK constraints
    check_matches = re.findall(r'CHECK\s*\(\s*([^)]+)\s*\)', content, re.IGNORECASE)
    for i, check_match in enumerate(check_matches):
        constraints.append({
            "name": f"check_{table_name}_{i+1}",
            "table_name": table_name,
            "type": "CHECK",
            "condition": check_match.strip()
        })

    # Look for column-level constraints
    lines = [line.strip() for line in content.split(',')]
    for line in lines:
        line = line.strip()
        if not line or line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT')):
            continue

        # Column-level PRIMARY KEY
        if 'PRIMARY KEY' in line.upper():
            parts = line.split()
            if len(parts) >= 1:
                col_name = parts[0].strip('`"\'')
                constraints.append({
                    "name": f"pk_{table_name}_{col_name}",
                    "table_name": table_name,
                    "type": "PRIMARY KEY",
                    "columns": [col_name]
                })

        # Column-level UNIQUE
        if 'UNIQUE' in line.upper() and 'UNIQUE(' not in line.upper():
            parts = line.split()
            if len(parts) >= 1:
                col_name = parts[0].strip('`"\'')
                constraints.append({
                    "name": f"unique_{table_name}_{col_name}",
                    "table_name": table_name,
                    "type": "UNIQUE",
                    "columns": [col_name]
                })

    return constraints


# ===== ADDITIONAL ENHANCED SCHEMA ENDPOINTS =====

@app.route('/api/schema/tables/<table_name>/constraints')
def get_table_constraints(table_name):
    """Get all constraints for a specific table"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print(f"Getting constraints for table: {table_name}")  # Debug log

        constraints = []

        # Get table schema
        schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        schema_result = web_explorer.execute_query(schema_query)

        if schema_result:
            create_sql = ""
            if isinstance(schema_result, dict) and schema_result.get('success') and schema_result.get('data'):
                if len(schema_result['data']) > 0:
                    create_sql = schema_result['data'][0].get('sql', '')
            elif isinstance(schema_result, list) and not (len(schema_result) > 0 and schema_result[0].get('error')):
                if len(schema_result) > 0:
                    create_sql = schema_result[0].get('sql', '')

            if create_sql:
                # Parse constraints from CREATE TABLE statement
                constraints = parse_constraints_from_sql(create_sql, table_name)

        # Get foreign key constraints
        fk_query = f"PRAGMA foreign_key_list({table_name})"
        fk_result = web_explorer.execute_query(fk_query)

        if fk_result:
            fk_data = []
            if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                fk_data = fk_result['data']
            elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                fk_data = fk_result

            for fk in fk_data:
                constraint_info = {
                    "name": f"fk_{table_name}_{fk.get('from', '')}",
                    "table_name": table_name,
                    "type": "FOREIGN KEY",
                    "columns": [fk.get('from', '')],
                    "referenced_table": fk.get('table', ''),
                    "referenced_columns": [fk.get('to', '')],
                    "on_delete": fk.get('on_delete', 'NO ACTION'),
                    "on_update": fk.get('on_update', 'NO ACTION')
                }
                constraints.append(constraint_info)

        return jsonify({
            "success": True,
            "constraints": constraints,
            "table_name": table_name,
            "total_count": len(constraints)
        })

    except Exception as e:
        print(f"Error in get_table_constraints: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/integrity-check')
def check_database_integrity():
    """Run database integrity checks"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Running database integrity check...")  # Debug log

        integrity_results = []

        # Run SQLite integrity check
        try:
            integrity_query = "PRAGMA integrity_check"
            integrity_result = web_explorer.execute_query(integrity_query)

            if integrity_result:
                integrity_data = []
                if isinstance(integrity_result, dict) and integrity_result.get('success') and integrity_result.get('data'):
                    integrity_data = integrity_result['data']
                elif isinstance(integrity_result, list) and not (len(integrity_result) > 0 and integrity_result[0].get('error')):
                    integrity_data = integrity_result

                for row in integrity_data:
                    message = row.get('integrity_check', '') if isinstance(row, dict) else str(row)
                    if message == 'ok':
                        integrity_results.append({
                            "type": "success",
                            "category": "Database Integrity",
                            "message": "Database integrity check passed"
                        })
                    else:
                        integrity_results.append({
                            "type": "error",
                            "category": "Database Integrity",
                            "message": message
                        })
        except Exception as integrity_error:
            integrity_results.append({
                "type": "error",
                "category": "Database Integrity",
                "message": f"Could not run integrity check: {str(integrity_error)}"
            })

        # Check foreign key constraints
        try:
            fk_check_query = "PRAGMA foreign_key_check"
            fk_check_result = web_explorer.execute_query(fk_check_query)

            if fk_check_result:
                fk_violations = []
                if isinstance(fk_check_result, dict) and fk_check_result.get('success') and fk_check_result.get('data'):
                    fk_violations = fk_check_result['data']
                elif isinstance(fk_check_result, list) and not (len(fk_check_result) > 0 and fk_check_result[0].get('error')):
                    fk_violations = fk_check_result

                if len(fk_violations) == 0:
                    integrity_results.append({
                        "type": "success",
                        "category": "Foreign Key Constraints",
                        "message": "All foreign key constraints are valid"
                    })
                else:
                    for violation in fk_violations:
                        table = violation.get('table', 'unknown')
                        rowid = violation.get('rowid', 'unknown')
                        parent = violation.get('parent', 'unknown')
                        fkid = violation.get('fkid', 'unknown')

                        integrity_results.append({
                            "type": "error",
                            "category": "Foreign Key Violation",
                            "message": f"Table '{table}' row {rowid} violates foreign key constraint {fkid} (references '{parent}')"
                        })
        except Exception as fk_error:
            integrity_results.append({
                "type": "warning",
                "category": "Foreign Key Check",
                "message": f"Could not check foreign key constraints: {str(fk_error)}"
            })

        # Quick schema consistency check
        try:
            # Check if all referenced tables in foreign keys exist
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            tables_result = web_explorer.execute_query(tables_query)

            existing_tables = set()
            if tables_result:
                if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                    existing_tables = {row['name'] for row in tables_result['data']}
                elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                    existing_tables = {row['name'] for row in tables_result if 'name' in row}

            # Check each table's foreign keys
            for table_name in existing_tables:
                try:
                    fk_query = f"PRAGMA foreign_key_list({table_name})"
                    fk_result = web_explorer.execute_query(fk_query)

                    if fk_result:
                        fk_data = []
                        if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                            fk_data = fk_result['data']
                        elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                            fk_data = fk_result

                        for fk in fk_data:
                            referenced_table = fk.get('table', '')
                            if referenced_table and referenced_table not in existing_tables:
                                integrity_results.append({
                                    "type": "error",
                                    "category": "Schema Consistency",
                                    "message": f"Table '{table_name}' references non-existent table '{referenced_table}'"
                                })
                except Exception as table_fk_error:
                    continue

        except Exception as schema_error:
            integrity_results.append({
                "type": "warning",
                "category": "Schema Check",
                "message": f"Could not complete schema consistency check: {str(schema_error)}"
            })

        # Summary
        error_count = len([r for r in integrity_results if r['type'] == 'error'])
        warning_count = len([r for r in integrity_results if r['type'] == 'warning'])
        success_count = len([r for r in integrity_results if r['type'] == 'success'])

        summary = {
            "total_checks": len(integrity_results),
            "errors": error_count,
            "warnings": warning_count,
            "successes": success_count,
            "overall_status": "error" if error_count > 0 else "warning" if warning_count > 0 else "success"
        }

        return jsonify({
            "success": True,
            "integrity_results": integrity_results,
            "summary": summary
        })

    except Exception as e:
        print(f"Error in check_database_integrity: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/optimize-suggestions')
def get_optimization_suggestions():
    """Get database optimization suggestions"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Generating optimization suggestions...")  # Debug log

        suggestions = []

        # Get database info
        db_info = web_explorer.get_database_info()

        if not db_info or 'error' in db_info:
            return jsonify({"error": "Could not get database information"})

        # Analyze tables for optimization opportunities
        if 'tables' in db_info:
            # Check for tables without indexes
            large_tables_without_indexes = []
            tables_with_many_rows = []

            for table in db_info['tables']:
                if table['rows'] > 1000:
                    tables_with_many_rows.append(table)

                    # Check if table has indexes
                    try:
                        structure_result = web_explorer.get_table_structure(table['name'])
                        if structure_result and 'structure' in structure_result:
                            indexes = structure_result['structure'].get('indexes', [])
                            if len(indexes) == 0 or all(idx.get('name', '').startswith('sqlite_') for idx in indexes):
                                large_tables_without_indexes.append(table)
                    except Exception as e:
                        print(f"Error checking indexes for {table['name']}: {e}")

            # Generate suggestions based on analysis
            if large_tables_without_indexes:
                suggestions.append({
                    "category": "Indexing",
                    "priority": "High",
                    "title": "Add indexes to large tables",
                    "description": f"Tables with many rows should have indexes for better query performance",
                    "affected_tables": [t['name'] for t in large_tables_without_indexes],
                    "recommendation": "Consider adding indexes on frequently queried columns, especially those used in WHERE clauses and JOIN conditions",
                    "estimated_impact": "High - Can significantly improve query performance"
                })

            if len(tables_with_many_rows) > 10:
                suggestions.append({
                    "category": "Data Management",
                    "priority": "Medium",
                    "title": "Consider data archival strategy",
                    "description": f"Database has {len(tables_with_many_rows)} tables with significant data",
                    "recommendation": "Implement data archival for historical records to maintain performance",
                    "estimated_impact": "Medium - Can improve overall database performance and maintenance"
                })

            # Check for foreign key relationships
            tables_with_fks = 0
            tables_without_fks = 0

            for table in db_info['tables']:
                try:
                    structure_result = web_explorer.get_table_structure(table['name'])
                    if structure_result and 'structure' in structure_result:
                        fks = structure_result['structure'].get('foreign_keys', [])
                        if len(fks) > 0:
                            tables_with_fks += 1
                        else:
                            tables_without_fks += 1
                except Exception as e:
                    tables_without_fks += 1

            if tables_without_fks > tables_with_fks and len(db_info['tables']) > 3:
                suggestions.append({
                    "category": "Data Integrity",
                    "priority": "Medium",
                    "title": "Improve referential integrity",
                    "description": f"Only {tables_with_fks} out of {len(db_info['tables'])} tables have foreign key relationships",
                    "recommendation": "Consider adding foreign key constraints to enforce referential integrity between related tables",
                    "estimated_impact": "Medium - Improves data consistency and prevents orphaned records"
                })

        # Database file size optimization
        file_size_mb = db_info.get('file_size_mb', 0)
        if file_size_mb > 100:
            suggestions.append({
                "category": "Storage",
                "priority": "Low",
                "title": "Database maintenance",
                "description": f"Database file size is {file_size_mb}MB",
                "recommendation": "Run VACUUM command periodically to reclaim unused space and optimize file structure",
                "estimated_impact": "Low to Medium - Can reduce file size and improve I/O performance"
            })

        # Analyze for missing primary keys
        try:
            tables_without_pk = []
            for table in db_info.get('tables', []):
                try:
                    structure_result = web_explorer.get_table_structure(table['name'])
                    if structure_result and 'structure' in structure_result:
                        pk_keys = structure_result['structure'].get('primary_keys', [])
                        if not pk_keys:
                            tables_without_pk.append(table['name'])
                except Exception as e:
                    continue

            if tables_without_pk:
                suggestions.append({
                    "category": "Schema Design",
                    "priority": "High",
                    "title": "Add primary keys to tables",
                    "description": f"{len(tables_without_pk)} tables lack primary keys",
                    "affected_tables": tables_without_pk,
                    "recommendation": "Every table should have a primary key for optimal performance and replication",
                    "estimated_impact": "High - Essential for data integrity and performance"
                })
        except Exception as pk_error:
            print(f"Error checking primary keys: {pk_error}")

        # Check for normalized schema
        if len(db_info.get('tables', [])) < 3:
            suggestions.append({
                "category": "Schema Design",
                "priority": "Low",
                "title": "Consider schema normalization",
                "description": "Database has very few tables which might indicate denormalized design",
                "recommendation": "Review if data is properly normalized to reduce redundancy and improve maintainability",
                "estimated_impact": "Variable - Depends on current schema design"
            })

        # Sort suggestions by priority
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)

        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "high_priority": len([s for s in suggestions if s["priority"] == "High"]),
            "medium_priority": len([s for s in suggestions if s["priority"] == "Medium"]),
            "low_priority": len([s for s in suggestions if s["priority"] == "Low"])
        })

    except Exception as e:
        print(f"Error in get_optimization_suggestions: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/maintenance/vacuum')
def run_vacuum():
    """Run VACUUM command to optimize database file"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Running VACUUM command...")  # Debug log

        # Get file size before VACUUM
        file_size_before = 0
        if web_explorer.db_path and os.path.exists(web_explorer.db_path):
            file_size_before = os.path.getsize(web_explorer.db_path)

        # Run VACUUM
        vacuum_result = web_explorer.execute_query("VACUUM")

        if not vacuum_result:
            return jsonify({
                "success": False,
                "error": "VACUUM command failed"
            })

        # Check if VACUUM succeeded
        success = False
        error_message = None

        if isinstance(vacuum_result, dict):
            success = vacuum_result.get('success', False)
            error_message = vacuum_result.get('error')
        elif isinstance(vacuum_result, list):
            if len(vacuum_result) > 0 and vacuum_result[0].get('error'):
                error_message = vacuum_result[0]['error']
            else:
                success = True

        if not success:
            return jsonify({
                "success": False,
                "error": error_message or "VACUUM command failed"
            })

        # Get file size after VACUUM
        file_size_after = 0
        if web_explorer.db_path and os.path.exists(web_explorer.db_path):
            file_size_after = os.path.getsize(web_explorer.db_path)

        # Calculate space saved
        space_saved_bytes = file_size_before - file_size_after
        space_saved_mb = space_saved_bytes / (1024 * 1024)

        return jsonify({
            "success": True,
            "message": "VACUUM command completed successfully",
            "file_size_before_mb": round(file_size_before / (1024 * 1024), 2),
            "file_size_after_mb": round(file_size_after / (1024 * 1024), 2),
            "space_saved_mb": round(space_saved_mb, 2),
            "space_saved_percent": round((space_saved_bytes / file_size_before * 100), 2) if file_size_before > 0 else 0
        })

    except Exception as e:
        print(f"Error in run_vacuum: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/maintenance/analyze')
def run_analyze():
    """Run ANALYZE command to update table statistics"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Running ANALYZE command...")  # Debug log

        # Run ANALYZE
        analyze_result = web_explorer.execute_query("ANALYZE")

        if not analyze_result:
            return jsonify({
                "success": False,
                "error": "ANALYZE command failed"
            })

        # Check if ANALYZE succeeded
        success = False
        error_message = None

        if isinstance(analyze_result, dict):
            success = analyze_result.get('success', False)
            error_message = analyze_result.get('error')
        elif isinstance(analyze_result, list):
            if len(analyze_result) > 0 and analyze_result[0].get('error'):
                error_message = analyze_result[0]['error']
            else:
                success = True

        if not success:
            return jsonify({
                "success": False,
                "error": error_message or "ANALYZE command failed"
            })

        return jsonify({
            "success": True,
            "message": "ANALYZE command completed successfully",
            "description": "Table statistics have been updated for optimal query planning"
        })

    except Exception as e:
        print(f"Error in run_analyze: {e}")  # Debug log
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ===== ENHANCED SCHEMA STATISTICS =====

# Fix the get_relationship_graph function in web_db_explorer.py
# Replace the existing function around line 2870

# Replace the get_relationship_graph function in web_db_explorer.py
# The issue is that PRAGMA foreign_key_list is being treated as a modify operation
# We need to fix the query execution and regex parsing

@app.route('/api/schema/relationship-graph')
def get_relationship_graph():
    """Get complete relationship graph data for visualization"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Building relationship graph...")  # Debug log

        # First, ensure foreign keys are enabled
        try:
            web_explorer.execute_query("PRAGMA foreign_keys = ON")
        except:
            pass

        # Get all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        print(f"Found tables: {table_names}")

        # Build nodes (tables)
        nodes = []
        edges = []
        table_stats = {}

        for table_name in table_names:
            try:
                # Get table statistics
                count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
                count_result = web_explorer.execute_query(count_query)

                row_count = 0
                if count_result:
                    if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                        row_count = count_result['data'][0].get('row_count', 0)
                    elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                        row_count = count_result[0].get('row_count', 0)

                # Get column count using PRAGMA
                columns_query = f"PRAGMA table_info(`{table_name}`)"
                columns_result = web_explorer.execute_query(columns_query)

                column_count = 0
                if columns_result:
                    if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                        column_count = len(columns_result['data'])
                    elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                        column_count = len(columns_result)

                print(f"Table {table_name}: {row_count} rows, {column_count} columns")

                # Determine node category based on table name and characteristics
                category = "data"  # default
                if table_name.lower() in ['users', 'employees', 'customers', 'people']:
                    category = "entity"
                elif table_name.lower() in ['orders', 'transactions', 'events', 'logs']:
                    category = "transaction"
                elif table_name.lower() in ['categories', 'types', 'status', 'lookup', 'regions', 'countries', 'jobs']:
                    category = "lookup"

                node = {
                    "id": table_name,
                    "name": table_name,
                    "type": "table",
                    "category": category,
                    "row_count": row_count,
                    "column_count": column_count,
                    "foreign_key_count": 0,  # Will be updated below
                    "size": max(10, min(50, row_count / 10)),  # Node size based on rows
                    "importance": row_count  # Will be updated with FK count
                }

                nodes.append(node)
                table_stats[table_name] = {
                    "rows": row_count,
                    "columns": column_count,
                    "fks": 0  # Will be updated
                }

            except Exception as table_error:
                print(f"Error analyzing table {table_name}: {table_error}")
                continue

        # Build edges (relationships) by parsing CREATE TABLE statements
        # Since PRAGMA foreign_key_list isn't working properly, we'll parse the SQL directly
        all_relationships = []
        foreign_key_counts = {}

        for table_name in table_names:
            try:
                print(f"Getting foreign keys for table: {table_name}")

                # Get CREATE TABLE statement
                create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                create_result = web_explorer.execute_query(create_query)

                create_sql = ""
                if create_result:
                    if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                        if len(create_result['data']) > 0:
                            create_sql = create_result['data'][0].get('sql', '')
                    elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                        if len(create_result) > 0:
                            create_sql = create_result[0].get('sql', '')

                print(f"CREATE SQL for {table_name}: {create_sql}")

                if create_sql and 'FOREIGN KEY' in create_sql.upper():
                    import re

                    # Parse foreign keys from CREATE TABLE statement
                    # Look for: FOREIGN KEY (column) REFERENCES table (column)
                    fk_pattern = r'FOREIGN KEY\s*\(\s*([^)]+)\s*\)\s*REFERENCES\s+([^(\s]+)\s*\(\s*([^)]+)\s*\)'
                    matches = re.findall(fk_pattern, create_sql, re.IGNORECASE | re.MULTILINE)

                    print(f"FK pattern matches for {table_name}: {matches}")

                    for match in matches:
                        from_column = match[0].strip()
                        to_table = match[1].strip()
                        to_column = match[2].strip()

                        # Clean up column and table names (remove quotes/backticks)
                        from_column = from_column.strip('`"\'')
                        to_table = to_table.strip('`"\'')
                        to_column = to_column.strip('`"\'')

                        print(f"Processing FK: {table_name}.{from_column} -> {to_table}.{to_column}")

                        if to_table in table_names:  # Ensure referenced table exists
                            # Calculate relationship strength (simplified)
                            strength = 1.0
                            relationship_type = "many_to_one"  # Most foreign keys are many-to-one

                            # For self-referencing (like manager_id), it's typically one-to-many
                            if table_name == to_table:
                                relationship_type = "one_to_many"

                            edge = {
                                "id": f"{table_name}_{to_table}_{from_column}",
                                "source": table_name,
                                "target": to_table,
                                "type": "foreign_key",
                                "relationship_type": relationship_type,
                                "from_column": from_column,
                                "to_column": to_column,
                                "strength": strength,
                                "on_delete": "CASCADE",  # Default from your schema
                                "on_update": "CASCADE",
                                "weight": max(1, min(10, strength * 10))
                            }

                            edges.append(edge)
                            all_relationships.append(edge)

                            # Update foreign key counts
                            foreign_key_counts[table_name] = foreign_key_counts.get(table_name, 0) + 1

                            print(f"Added edge: {table_name} -> {to_table}")

            except Exception as fk_error:
                print(f"Error getting foreign keys for {table_name}: {fk_error}")
                import traceback
                traceback.print_exc()
                continue

        # Update nodes with foreign key counts and importance
        for node in nodes:
            fk_count = foreign_key_counts.get(node["name"], 0)
            node["foreign_key_count"] = fk_count
            node["importance"] = node["row_count"] + (fk_count * 100)

            # Update table stats
            if node["name"] in table_stats:
                table_stats[node["name"]]["fks"] = fk_count

        # Calculate graph metrics
        connected_node_ids = set()
        for edge in edges:
            connected_node_ids.add(edge["source"])
            connected_node_ids.add(edge["target"])

        graph_metrics = {
            "total_tables": len(nodes),
            "total_relationships": len(edges),
            "connected_tables": len(connected_node_ids),
            "isolated_tables": len(nodes) - len(connected_node_ids),
            "avg_relationships_per_table": len(edges) / len(nodes) if len(nodes) > 0 else 0,
            "most_connected_table": max(nodes, key=lambda x: x["foreign_key_count"])["name"] if nodes else None,
            "largest_table": max(nodes, key=lambda x: x["row_count"])["name"] if nodes else None
        }

        print(f"Final graph: {len(nodes)} nodes, {len(edges)} edges")
        print(f"Edges: {[f'{e['source']}->{e['target']}' for e in edges]}")

        return jsonify({
            "success": True,
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "metrics": graph_metrics
            },
            "relationships": all_relationships,
            "table_stats": table_stats
        })

    except Exception as e:
        print(f"Error building relationship graph: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# Also add this improved debug endpoint
@app.route('/api/debug/parse-foreign-keys')
def debug_parse_foreign_keys():
    """Debug foreign key parsing specifically"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        debug_info = {}

        # Get all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        for table_name in table_names:
            # Get CREATE TABLE SQL
            create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            create_result = web_explorer.execute_query(create_query)

            create_sql = ""
            if create_result:
                if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                    if len(create_result['data']) > 0:
                        create_sql = create_result['data'][0].get('sql', '')
                elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                    if len(create_result) > 0:
                        create_sql = create_result[0].get('sql', '')

            # Parse foreign keys using the new regex
            foreign_keys = []
            if create_sql and 'FOREIGN KEY' in create_sql.upper():
                import re
                fk_pattern = r'FOREIGN KEY\s*\(\s*([^)]+)\s*\)\s*REFERENCES\s+([^(\s]+)\s*\(\s*([^)]+)\s*\)'
                matches = re.findall(fk_pattern, create_sql, re.IGNORECASE | re.MULTILINE)

                for match in matches:
                    from_column = match[0].strip().strip('`"\'')
                    to_table = match[1].strip().strip('`"\'')
                    to_column = match[2].strip().strip('`"\'')

                    foreign_keys.append({
                        "from_column": from_column,
                        "to_table": to_table,
                        "to_column": to_column
                    })

            debug_info[table_name] = {
                "create_sql": create_sql,
                "has_foreign_key_text": 'FOREIGN KEY' in create_sql.upper(),
                "parsed_foreign_keys": foreign_keys
            }

        return jsonify({
            "success": True,
            "debug_info": debug_info
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ===== ENHANCED RELATIONSHIP VISUALIZATION APIS =====

@app.route('/api/schema/relationship-analysis')
def get_relationship_analysis():
    """Get detailed relationship analytics"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Performing relationship analysis...")

        analysis = {
            "integrity_issues": [],
            "orphaned_records": [],
            "cardinality_analysis": [],
            "performance_insights": [],
            "recommendations": []
        }

        # Get all tables and their foreign keys
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        # Analyze each table's relationships
        for table_name in table_names:
            try:
                # Get foreign keys
                fk_query = f"PRAGMA foreign_key_list(`{table_name}`)"
                fk_result = web_explorer.execute_query(fk_query)

                if fk_result:
                    fk_data = []
                    if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                        fk_data = fk_result['data']
                    elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                        fk_data = fk_result

                    for fk in fk_data:
                        from_column = fk.get('from', '')
                        to_table = fk.get('table', '')
                        to_column = fk.get('to', '')

                        # Check for orphaned records
                        try:
                            orphan_query = f"""
                                SELECT COUNT(*) as orphaned_count
                                FROM `{table_name}` t1
                                LEFT JOIN `{to_table}` t2 ON t1.`{from_column}` = t2.`{to_column}`
                                WHERE t1.`{from_column}` IS NOT NULL 
                                AND t2.`{to_column}` IS NULL
                            """
                            orphan_result = web_explorer.execute_query(orphan_query)

                            orphaned_count = 0
                            if orphan_result:
                                if isinstance(orphan_result, dict) and orphan_result.get('success') and orphan_result.get('data'):
                                    orphaned_count = orphan_result['data'][0].get('orphaned_count', 0)
                                elif isinstance(orphan_result, list) and not (len(orphan_result) > 0 and orphan_result[0].get('error')):
                                    orphaned_count = orphan_result[0].get('orphaned_count', 0)

                            if orphaned_count > 0:
                                analysis["orphaned_records"].append({
                                    "table": table_name,
                                    "column": from_column,
                                    "referenced_table": to_table,
                                    "referenced_column": to_column,
                                    "orphaned_count": orphaned_count,
                                    "severity": "high" if orphaned_count > 100 else "medium" if orphaned_count > 10 else "low"
                                })

                        except Exception as orphan_error:
                            print(f"Error checking orphaned records: {orphan_error}")

                        # Analyze cardinality
                        try:
                            cardinality_query = f"""
                                SELECT 
                                    COUNT(*) as total_relationships,
                                    COUNT(DISTINCT `{from_column}`) as distinct_foreign_keys,
                                    MIN(`{from_column}`) as min_fk,
                                    MAX(`{from_column}`) as max_fk
                                FROM `{table_name}`
                                WHERE `{from_column}` IS NOT NULL
                            """
                            card_result = web_explorer.execute_query(cardinality_query)

                            if card_result:
                                card_data = None
                                if isinstance(card_result, dict) and card_result.get('success') and card_result.get('data'):
                                    card_data = card_result['data'][0]
                                elif isinstance(card_result, list) and not (len(card_result) > 0 and card_result[0].get('error')):
                                    card_data = card_result[0]

                                if card_data:
                                    total_rel = card_data.get('total_relationships', 0)
                                    distinct_fks = card_data.get('distinct_foreign_keys', 0)

                                    if total_rel > 0 and distinct_fks > 0:
                                        avg_refs_per_key = total_rel / distinct_fks
                                        uniqueness_ratio = distinct_fks / total_rel

                                        relationship_type = "one_to_one"
                                        if avg_refs_per_key > 2:
                                            relationship_type = "one_to_many"
                                        elif uniqueness_ratio < 0.5:
                                            relationship_type = "many_to_one"

                                        analysis["cardinality_analysis"].append({
                                            "from_table": table_name,
                                            "from_column": from_column,
                                            "to_table": to_table,
                                            "to_column": to_column,
                                            "relationship_type": relationship_type,
                                            "total_relationships": total_rel,
                                            "unique_foreign_keys": distinct_fks,
                                            "avg_references_per_key": round(avg_refs_per_key, 2),
                                            "uniqueness_ratio": round(uniqueness_ratio, 3)
                                        })

                        except Exception as card_error:
                            print(f"Error analyzing cardinality: {card_error}")

            except Exception as table_error:
                print(f"Error analyzing table {table_name}: {table_error}")
                continue

        # Generate performance insights
        for rel in analysis["cardinality_analysis"]:
            if rel["avg_references_per_key"] > 100:
                analysis["performance_insights"].append({
                    "type": "high_cardinality",
                    "table": rel["from_table"],
                    "issue": f"High cardinality relationship ({rel['avg_references_per_key']} avg refs per key)",
                    "recommendation": "Consider indexing the foreign key column for better JOIN performance",
                    "impact": "high"
                })

        # Generate recommendations
        if len(analysis["orphaned_records"]) > 0:
            analysis["recommendations"].append({
                "type": "data_integrity",
                "title": "Fix orphaned records",
                "description": f"Found {len(analysis['orphaned_records'])} relationships with orphaned records",
                "action": "Run referential integrity cleanup or add proper CASCADE rules",
                "priority": "high"
            })

        if len(analysis["cardinality_analysis"]) == 0:
            analysis["recommendations"].append({
                "type": "schema_design",
                "title": "Add foreign key relationships",
                "description": "No foreign key relationships detected",
                "action": "Consider adding foreign key constraints to enforce referential integrity",
                "priority": "medium"
            })

        # Performance recommendations
        high_card_rels = [r for r in analysis["cardinality_analysis"] if r["avg_references_per_key"] > 50]
        if high_card_rels:
            analysis["recommendations"].append({
                "type": "performance",
                "title": "Index high-cardinality foreign keys",
                "description": f"Found {len(high_card_rels)} high-cardinality relationships",
                "action": "Add indexes to foreign key columns for better JOIN performance",
                "priority": "medium"
            })

        return jsonify({
            "success": True,
            "analysis": analysis,
            "summary": {
                "total_relationships": len(analysis["cardinality_analysis"]),
                "orphaned_records": len(analysis["orphaned_records"]),
                "integrity_issues": len(analysis["integrity_issues"]),
                "performance_insights": len(analysis["performance_insights"]),
                "recommendations": len(analysis["recommendations"])
            }
        })

    except Exception as e:
        print(f"Error in relationship analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/join-paths', methods=['POST'])
def find_join_paths():
    """Find optimal JOIN paths between tables"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    start_table = data.get('start_table', '').strip()
    end_table = data.get('end_table', '').strip()
    max_hops = data.get('max_hops', 3)

    if not start_table or not end_table:
        return jsonify({"error": "start_table and end_table are required"})

    try:
        print(f"Finding JOIN path from {start_table} to {end_table}")

        # Build relationship graph
        graph = {}
        all_tables = set()

        # Get all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        # Build adjacency graph
        for table_name in table_names:
            graph[table_name] = []
            all_tables.add(table_name)

            try:
                # Get foreign keys
                fk_query = f"PRAGMA foreign_key_list(`{table_name}`)"
                fk_result = web_explorer.execute_query(fk_query)

                if fk_result:
                    fk_data = []
                    if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                        fk_data = fk_result['data']
                    elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                        fk_data = fk_result

                    for fk in fk_data:
                        to_table = fk.get('table', '')
                        from_column = fk.get('from', '')
                        to_column = fk.get('to', '')

                        if to_table in table_names:
                            # Add bidirectional edges
                            graph[table_name].append({
                                "table": to_table,
                                "from_column": from_column,
                                "to_column": to_column,
                                "direction": "outgoing",
                                "join_condition": f"{table_name}.{from_column} = {to_table}.{to_column}"
                            })

                            if to_table not in graph:
                                graph[to_table] = []

                            graph[to_table].append({
                                "table": table_name,
                                "from_column": to_column,
                                "to_column": from_column,
                                "direction": "incoming",
                                "join_condition": f"{to_table}.{to_column} = {table_name}.{from_column}"
                            })

            except Exception as table_error:
                print(f"Error building graph for table {table_name}: {table_error}")
                continue

        # Find shortest path using BFS
        def find_shortest_path(start, end, max_depth):
            if start == end:
                return [start]

            queue = [(start, [start])]
            visited = set([start])

            while queue:
                current_table, path = queue.pop(0)

                if len(path) > max_depth:
                    continue

                for neighbor in graph.get(current_table, []):
                    next_table = neighbor["table"]

                    if next_table == end:
                        return path + [next_table]

                    if next_table not in visited:
                        visited.add(next_table)
                        queue.append((next_table, path + [next_table]))

            return None

        # Find path
        path = find_shortest_path(start_table, end_table, max_hops)

        if not path:
            return jsonify({
                "success": False,
                "error": f"No JOIN path found between {start_table} and {end_table} within {max_hops} hops"
            })

        # Build JOIN details
        join_details = []
        join_conditions = []
        join_sql_parts = []

        for i in range(len(path) - 1):
            current_table = path[i]
            next_table = path[i + 1]

            # Find the relationship
            relationship = None
            for neighbor in graph.get(current_table, []):
                if neighbor["table"] == next_table:
                    relationship = neighbor
                    break

            if relationship:
                join_details.append({
                    "from_table": current_table,
                    "to_table": next_table,
                    "from_column": relationship["from_column"],
                    "to_column": relationship["to_column"],
                    "join_condition": relationship["join_condition"],
                    "direction": relationship["direction"]
                })

                join_conditions.append(relationship["join_condition"])

                # Build SQL parts
                if i == 0:
                    join_sql_parts.append(f"FROM `{current_table}` t1")

                join_type = "INNER JOIN"  # Default
                alias = f"t{i + 2}"
                join_sql_parts.append(f"{join_type} `{next_table}` {alias} ON {relationship['join_condition'].replace(current_table, f't{i+1}').replace(next_table, alias)}")

        # Generate complete SQL
        select_clause = "SELECT " + ", ".join([f"t{i+1}.*" for i in range(len(path))])
        join_sql = select_clause + "\n" + "\n".join(join_sql_parts) + "\nLIMIT 10;"

        # Alternative JOIN strategies
        alternatives = []

        # Left JOIN version
        left_join_parts = []
        for i, part in enumerate(join_sql_parts):
            if i == 0:
                left_join_parts.append(part)
            else:
                left_join_parts.append(part.replace("INNER JOIN", "LEFT JOIN"))

        left_join_sql = select_clause + "\n" + "\n".join(left_join_parts) + "\nLIMIT 10;"
        alternatives.append({
            "name": "Left JOIN (Include NULL relationships)",
            "sql": left_join_sql,
            "description": "Includes records from the first table even if no matching records in subsequent tables"
        })

        return jsonify({
            "success": True,
            "path": {
                "tables": path,
                "hops": len(path) - 1,
                "join_details": join_details,
                "sql": join_sql,
                "alternatives": alternatives
            },
            "performance": {
                "estimated_complexity": len(path) - 1,
                "recommendation": "Consider adding indexes on join columns for better performance" if len(path) > 2 else "Simple join should perform well"
            }
        })

    except Exception as e:
        print(f"Error finding JOIN path: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/query-templates')
def get_query_templates():
    """Get intelligent query templates based on schema"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        # Get schema information
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        templates = {
            "basic_queries": [],
            "relationship_queries": [],
            "analytical_queries": [],
            "reporting_queries": []
        }

        # Generate basic queries for each table
        for table_name in table_names:
            templates["basic_queries"].extend([
                {
                    "name": f"View all {table_name}",
                    "description": f"Select all records from {table_name}",
                    "sql": f"SELECT * FROM `{table_name}` LIMIT 100;",
                    "category": "basic"
                },
                {
                    "name": f"Count {table_name} records",
                    "description": f"Count total records in {table_name}",
                    "sql": f"SELECT COUNT(*) as total_records FROM `{table_name}`;",
                    "category": "basic"
                }
            ])

        # Generate relationship queries
        for table_name in table_names:
            try:
                fk_query = f"PRAGMA foreign_key_list(`{table_name}`)"
                fk_result = web_explorer.execute_query(fk_query)

                if fk_result:
                    fk_data = []
                    if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                        fk_data = fk_result['data']
                    elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                        fk_data = fk_result

                    for fk in fk_data:
                        to_table = fk.get('table', '')
                        from_column = fk.get('from', '')
                        to_column = fk.get('to', '')

                        if to_table:
                            templates["relationship_queries"].append({
                                "name": f"{table_name} with {to_table} details",
                                "description": f"Join {table_name} with related {to_table} information",
                                "sql": f"""SELECT t1.*, t2.*
FROM `{table_name}` t1
INNER JOIN `{to_table}` t2 ON t1.`{from_column}` = t2.`{to_column}`
LIMIT 50;""",
                                "category": "relationship"
                            })

            except Exception as fk_error:
                continue

        # Generate analytical queries
        templates["analytical_queries"] = [
            {
                "name": "Table sizes analysis",
                "description": "Compare table sizes across the database",
                "sql": f"""SELECT 
    '{table_name}' as table_name,
    COUNT(*) as record_count
FROM `{table_name}`
UNION ALL
""".join([f"SELECT '{t}' as table_name, COUNT(*) as record_count FROM `{t}`" for t in table_names[:5]]).rstrip("UNION ALL\n") + "\nORDER BY record_count DESC;",
                "category": "analytical"
            }
        ]

        return jsonify({
            "success": True,
            "templates": templates,
            "total_templates": sum(len(category) for category in templates.values())
        })

    except Exception as e:
        print(f"Error generating query templates: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# Add these utility functions at the end of your file

def calculate_relationship_strength(from_table, from_column, to_table, to_column, web_explorer):
    """Calculate the strength of a relationship based on data distribution"""
    try:
        # Query to analyze the relationship
        analysis_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT `{from_column}`) as distinct_values,
                COUNT(CASE WHEN `{from_column}` IS NULL THEN 1 END) as null_values
            FROM `{from_table}`
        """

        result = web_explorer.execute_query(analysis_query)

        if result:
            data = None
            if isinstance(result, dict) and result.get('success') and result.get('data'):
                data = result['data'][0]
            elif isinstance(result, list) and not (len(result) > 0 and result[0].get('error')):
                data = result[0]

            if data:
                total_records = data.get('total_records', 0)
                distinct_values = data.get('distinct_values', 0)
                null_values = data.get('null_values', 0)

                if total_records > 0:
                    # Calculate uniqueness ratio (higher = stronger relationship)
                    uniqueness_ratio = distinct_values / total_records

                    # Calculate completeness (lower null percentage = stronger)
                    completeness = 1 - (null_values / total_records)

                    # Combined strength score (0-1)
                    strength = (uniqueness_ratio * 0.7) + (completeness * 0.3)
                    return min(1.0, max(0.1, strength))

        return 0.5  # Default strength

    except Exception as e:
        print(f"Error calculating relationship strength: {e}")
        return 0.5


def detect_relationship_type(from_table, from_column, to_table, to_column, web_explorer):
    """Detect the type of relationship (one-to-one, one-to-many, many-to-many)"""
    try:
        # Check cardinality from source table
        source_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT `{from_column}`) as distinct_foreign_keys
            FROM `{from_table}`
            WHERE `{from_column}` IS NOT NULL
        """

        source_result = web_explorer.execute_query(source_query)

        if source_result:
            source_data = None
            if isinstance(source_result, dict) and source_result.get('success') and source_result.get('data'):
                source_data = source_result['data'][0]
            elif isinstance(source_result, list) and not (len(source_result) > 0 and source_result[0].get('error')):
                source_data = source_result[0]

            if source_data:
                total_rows = source_data.get('total_rows', 0)
                distinct_fks = source_data.get('distinct_foreign_keys', 0)

                if total_rows > 0 and distinct_fks > 0:
                    uniqueness_ratio = distinct_fks / total_rows

                    if uniqueness_ratio > 0.95:
                        return "one_to_one"
                    elif uniqueness_ratio > 0.1:
                        return "many_to_one"
                    else:
                        return "many_to_many"

        return "one_to_many"  # Default

    except Exception as e:
        print(f"Error detecting relationship type: {e}")
        return "one_to_many"


def find_circular_dependencies(graph):
    """Find circular dependencies in the relationship graph"""
    def has_cycle_dfs(node, visited, rec_stack, graph):
        visited[node] = True
        rec_stack[node] = True

        for neighbor in graph.get(node, []):
            target = neighbor["table"]
            if not visited.get(target, False):
                if has_cycle_dfs(target, visited, rec_stack, graph):
                    return True
            elif rec_stack.get(target, False):
                return True

        rec_stack[node] = False
        return False

    visited = {}
    rec_stack = {}
    cycles = []

    for node in graph:
        if not visited.get(node, False):
            if has_cycle_dfs(node, visited, rec_stack, graph):
                cycles.append(node)

    return cycles


# ===== ENHANCED RELATIONSHIP VISUALIZATION FRONTEND INTEGRATION =====

@app.route('/api/schema/relationship-suggestions')
def get_relationship_suggestions():
    """Suggest potential relationships based on column names and data patterns"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Analyzing potential relationships...")

        suggestions = []

        # Get all tables and their columns
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        # Collect all columns
        table_columns = {}
        for table_name in table_names:
            try:
                columns_query = f"PRAGMA table_info(`{table_name}`)"
                columns_result = web_explorer.execute_query(columns_query)

                columns = []
                if columns_result:
                    if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                        columns = [col['name'] for col in columns_result['data']]
                    elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                        columns = [col['name'] for col in columns_result]

                table_columns[table_name] = columns

            except Exception as col_error:
                print(f"Error getting columns for {table_name}: {col_error}")
                continue

        # Look for potential foreign key relationships
        for table_name, columns in table_columns.items():
            for column in columns:
                # Check for common foreign key patterns
                potential_matches = []

                # Pattern 1: column ends with _id and matches table_id pattern
                if column.endswith('_id'):
                    base_name = column[:-3]  # Remove '_id'

                    # Look for tables that match
                    for other_table in table_names:
                        if other_table != table_name:
                            # Check if other table has 'id' column or matching primary key
                            other_columns = table_columns.get(other_table, [])

                            if ('id' in other_columns or
                                    f"{other_table}_id" in other_columns or
                                    base_name.lower() in other_table.lower()):

                                potential_matches.append({
                                    "from_table": table_name,
                                    "from_column": column,
                                    "to_table": other_table,
                                    "to_column": "id" if "id" in other_columns else f"{other_table}_id",
                                    "confidence": 0.8,
                                    "reason": f"Column '{column}' likely references '{other_table}' table"
                                })

                # Pattern 2: column name matches table name + _id
                for other_table in table_names:
                    if other_table != table_name and column.lower() == f"{other_table.lower()}_id":
                        other_columns = table_columns.get(other_table, [])
                        if "id" in other_columns:
                            potential_matches.append({
                                "from_table": table_name,
                                "from_column": column,
                                "to_table": other_table,
                                "to_column": "id",
                                "confidence": 0.9,
                                "reason": f"Column '{column}' directly references '{other_table}.id'"
                            })

                suggestions.extend(potential_matches)

        # Remove duplicates and sort by confidence
        unique_suggestions = []
        seen = set()

        for suggestion in suggestions:
            key = f"{suggestion['from_table']}.{suggestion['from_column']}->{suggestion['to_table']}.{suggestion['to_column']}"
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)

        unique_suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            "success": True,
            "suggestions": unique_suggestions[:20],  # Limit to top 20
            "total_suggestions": len(unique_suggestions)
        })

    except Exception as e:
        print(f"Error generating relationship suggestions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/relationship-metrics')
def get_relationship_metrics():
    """Get comprehensive relationship metrics for the database"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Calculating relationship metrics...")

        metrics = {
            "connectivity": {},
            "complexity": {},
            "integrity": {},
            "performance": {}
        }

        # Get relationship graph data
        graph_response = get_relationship_graph()
        graph_data = json.loads(graph_response.data)

        if not graph_data.get('success'):
            return jsonify({"error": "Could not build relationship graph"})

        graph = graph_data['graph']
        nodes = graph['nodes']
        edges = graph['edges']

        # Connectivity metrics
        total_tables = len(nodes)
        connected_tables = len(set([e['source'] for e in edges] + [e['target'] for e in edges]))
        isolated_tables = total_tables - connected_tables

        metrics["connectivity"] = {
            "total_tables": total_tables,
            "connected_tables": connected_tables,
            "isolated_tables": isolated_tables,
            "connectivity_ratio": round(connected_tables / total_tables, 3) if total_tables > 0 else 0,
            "total_relationships": len(edges),
            "average_relationships_per_table": round(len(edges) / total_tables, 2) if total_tables > 0 else 0
        }

        # Complexity metrics
        relationship_types = {}
        for edge in edges:
            rel_type = edge.get('relationship_type', 'unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        # Calculate node degrees (number of connections per table)
        node_degrees = {}
        for edge in edges:
            source = edge['source']
            target = edge['target']
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1

        max_degree = max(node_degrees.values()) if node_degrees else 0
        avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0

        metrics["complexity"] = {
            "relationship_types": relationship_types,
            "max_table_connections": max_degree,
            "average_table_connections": round(avg_degree, 2),
            "most_connected_table": max(node_degrees.items(), key=lambda x: x[1])[0] if node_degrees else None,
            "complexity_score": round((max_degree * 0.4) + (avg_degree * 0.6), 2)
        }

        # Integrity metrics (from relationship analysis)
        analysis_response = get_relationship_analysis()
        analysis_data = json.loads(analysis_response.data)

        if analysis_data.get('success'):
            analysis = analysis_data['analysis']

            metrics["integrity"] = {
                "orphaned_records": len(analysis.get('orphaned_records', [])),
                "integrity_issues": len(analysis.get('integrity_issues', [])),
                "integrity_score": max(0, 100 - (len(analysis.get('orphaned_records', [])) * 5)),
                "recommendations": len(analysis.get('recommendations', []))
            }

        # Performance metrics
        strong_relationships = len([e for e in edges if e.get('strength', 0) > 0.7])
        weak_relationships = len([e for e in edges if e.get('strength', 0) < 0.3])

        metrics["performance"] = {
            "strong_relationships": strong_relationships,
            "weak_relationships": weak_relationships,
            "indexed_foreign_keys": 0,  # Would need additional analysis
            "performance_score": round(((strong_relationships / len(edges)) * 100) if edges else 0, 1),
            "optimization_potential": "High" if weak_relationships > strong_relationships else "Medium" if weak_relationships > 0 else "Low"
        }

        # Overall health score
        connectivity_score = metrics["connectivity"]["connectivity_ratio"] * 100
        integrity_score = metrics["integrity"].get("integrity_score", 100)
        performance_score = metrics["performance"]["performance_score"]

        overall_health = round((connectivity_score + integrity_score + performance_score) / 3, 1)

        return jsonify({
            "success": True,
            "metrics": metrics,
            "overall_health_score": overall_health,
            "health_rating": "Excellent" if overall_health > 80 else "Good" if overall_health > 60 else "Fair" if overall_health > 40 else "Poor",
            "generated_at": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error calculating relationship metrics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/export-relationship-diagram', methods=['POST'])
def export_relationship_diagram():
    """Export relationship diagram data for external visualization tools"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    data = request.get_json()
    export_format = data.get('format', 'json')  # json, graphml, dot, cytoscape

    try:
        # Get relationship graph data
        graph_response = get_relationship_graph()
        graph_data = json.loads(graph_response.data)

        if not graph_data.get('success'):
            return jsonify({"error": "Could not build relationship graph"})

        graph = graph_data['graph']
        nodes = graph['nodes']
        edges = graph['edges']

        if export_format == 'json':
            # Standard JSON format
            export_data = {
                "format": "json",
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "database": os.path.basename(web_explorer.db_path) if web_explorer.db_path else "Unknown",
                    "total_tables": len(nodes),
                    "total_relationships": len(edges)
                },
                "nodes": nodes,
                "edges": edges,
                "metrics": graph['metrics']
            }

        elif export_format == 'cytoscape':
            # Cytoscape.js format
            cytoscape_elements = []

            # Add nodes
            for node in nodes:
                cytoscape_elements.append({
                    "data": {
                        "id": node["id"],
                        "label": node["name"],
                        "category": node["category"],
                        "row_count": node["row_count"],
                        "column_count": node["column_count"]
                    },
                    "classes": node["category"]
                })

            # Add edges
            for edge in edges:
                cytoscape_elements.append({
                    "data": {
                        "id": edge["id"],
                        "source": edge["source"],
                        "target": edge["target"],
                        "label": f"{edge['from_column']} → {edge['to_column']}",
                        "relationship_type": edge["relationship_type"],
                        "strength": edge["strength"]
                    },
                    "classes": edge["relationship_type"]
                })

            export_data = {
                "format": "cytoscape",
                "elements": cytoscape_elements
            }

        elif export_format == 'dot':
            # GraphViz DOT format
            dot_lines = ['digraph database_schema {']
            dot_lines.append('  rankdir=LR;')
            dot_lines.append('  node [shape=box, style=filled];')

            # Add nodes
            for node in nodes:
                color = {"entity": "lightblue", "transaction": "lightgreen", "lookup": "lightyellow"}.get(node["category"], "lightgray")
                dot_lines.append(f'  "{node["name"]}" [fillcolor={color}, label="{node["name"]}\\n{node["row_count"]} rows"];')

            # Add edges
            for edge in edges:
                label = f"{edge['from_column']} → {edge['to_column']}"
                dot_lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [label="{label}"];')

            dot_lines.append('}')

            export_data = {
                "format": "dot",
                "content": '\n'.join(dot_lines)
            }

        elif export_format == 'graphml':
            # GraphML format for tools like yEd
            graphml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="name" for="node" attr.name="name" attr.type="string"/>
  <key id="category" for="node" attr.name="category" attr.type="string"/>
  <key id="row_count" for="node" attr.name="row_count" attr.type="int"/>
  <key id="relationship" for="edge" attr.name="relationship" attr.type="string"/>
  <graph id="database_schema" edgedefault="directed">
'''

            # Add nodes
            for node in nodes:
                graphml_content += f'''    <node id="{node["id"]}">
      <data key="name">{node["name"]}</data>
      <data key="category">{node["category"]}</data>
      <data key="row_count">{node["row_count"]}</data>
    </node>
'''

            # Add edges
            for edge in edges:
                graphml_content += f'''    <edge source="{edge["source"]}" target="{edge["target"]}">
      <data key="relationship">{edge["from_column"]} → {edge["to_column"]}</data>
    </edge>
'''

            graphml_content += '''  </graph>
</graphml>'''

            export_data = {
                "format": "graphml",
                "content": graphml_content
            }

        else:
            return jsonify({"error": f"Unsupported export format: {export_format}"})

        return jsonify({
            "success": True,
            "export_data": export_data,
            "format": export_format
        })

    except Exception as e:
        print(f"Error exporting relationship diagram: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/relationship')
def relationship_visualization():
    """Enhanced Database Relationship Visualization Page"""
    return render_template('relationship.html')

@app.route('/relationships')  # Alternative URL
def relationships_redirect():
    """Redirect to relationship visualization"""
    return redirect('/relationship')

@app.route('/schema/relationships')  # Another alternative
def schema_relationships():
    """Schema relationships visualization"""
    return render_template('relationship.html')

@app.route('/check-templates')
def check_templates():
    """Debug endpoint to check template availability"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')

    template_status = {
        "templates_directory": templates_dir,
        "templates_directory_exists": os.path.exists(templates_dir),
        "templates": {}
    }

    required_templates = ['index.html', 'relationship.html']

    for template in required_templates:
        template_path = os.path.join(templates_dir, template)
        template_status["templates"][template] = {
            "exists": os.path.exists(template_path),
            "path": template_path,
            "size": os.path.getsize(template_path) if os.path.exists(template_path) else 0
        }

    return jsonify(template_status)

@app.route('/api/navigation')
def get_navigation():
    """Get available navigation options"""
    return jsonify({
        "success": True,
        "navigation": {
            "main_pages": [
                {
                    "title": "Database Explorer",
                    "url": "/",
                    "description": "Main database exploration interface",
                    "icon": "🏠"
                },
                {
                    "title": "Relationship Visualization",
                    "url": "/relationship",
                    "description": "Enhanced database relationship visualization",
                    "icon": "🔗"
                }
            ],
            "api_endpoints": [
                {
                    "title": "API Documentation",
                    "url": "/api/docs",
                    "description": "Interactive API documentation",
                    "icon": "📚"
                }
            ]
        }
    })

# Add this debug route to test foreign key detection
# Add this to your web_db_explorer.py file

@app.route('/api/debug/foreign-keys')
def debug_foreign_keys():
    """Debug foreign key detection"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        debug_info = {}

        # Get all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        for table_name in table_names:
            table_debug = {
                "pragma_foreign_key_list": None,
                "create_table_sql": None,
                "parsed_foreign_keys": []
            }

            # Test PRAGMA foreign_key_list
            try:
                fk_query = f"PRAGMA foreign_key_list(`{table_name}`)"
                fk_result = web_explorer.execute_query(fk_query)
                table_debug["pragma_foreign_key_list"] = fk_result
            except Exception as e:
                table_debug["pragma_foreign_key_list"] = f"Error: {str(e)}"

            # Get CREATE TABLE SQL
            try:
                create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                create_result = web_explorer.execute_query(create_query)

                if create_result:
                    if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                        if len(create_result['data']) > 0:
                            create_sql = create_result['data'][0].get('sql', '')
                            table_debug["create_table_sql"] = create_sql
                    elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                        if len(create_result) > 0:
                            create_sql = create_result[0].get('sql', '')
                            table_debug["create_table_sql"] = create_sql

                    # Try to parse foreign keys from SQL
                    if create_sql and 'REFERENCES' in create_sql.upper():
                        import re
                        fk_pattern = r'(\w+)\s+[^,\)]*\s+REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)'
                        matches = re.findall(fk_pattern, create_sql, re.IGNORECASE | re.MULTILINE)
                        table_debug["parsed_foreign_keys"] = matches

            except Exception as e:
                table_debug["create_table_sql"] = f"Error: {str(e)}"

            debug_info[table_name] = table_debug

        return jsonify({
            "success": True,
            "debug_info": debug_info
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/debug/foreign-key-status')
def check_foreign_key_status():
    """Check if foreign keys are enabled"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        # Check if foreign keys are enabled
        fk_status_result = web_explorer.execute_query("PRAGMA foreign_keys")

        # Enable foreign keys if not enabled
        enable_result = web_explorer.execute_query("PRAGMA foreign_keys = ON")

        # Check status again
        fk_status_after = web_explorer.execute_query("PRAGMA foreign_keys")

        return jsonify({
            "success": True,
            "foreign_keys_before": fk_status_result,
            "enable_result": enable_result,
            "foreign_keys_after": fk_status_after
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# Add these missing API endpoints to your web_db_explorer.py file

@app.route('/api/schema/statistics')
def get_schema_statistics():
    """Get comprehensive schema statistics"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Generating comprehensive schema statistics...")  # Debug log

        statistics = {
            "overview": {},
            "table_analysis": {},
            "relationship_analysis": {},
            "index_analysis": {},
            "constraint_analysis": {},
            "storage_analysis": {},
            "data_quality": {}
        }

        # Get database file information
        db_file_size = 0
        if web_explorer.db_path and os.path.exists(web_explorer.db_path):
            db_file_size = os.path.getsize(web_explorer.db_path)

        # Get all user tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        # Get views count
        views_query = "SELECT COUNT(*) as view_count FROM sqlite_master WHERE type='view'"
        views_result = web_explorer.execute_query(views_query)
        total_views = 0
        if views_result:
            if isinstance(views_result, dict) and views_result.get('success') and views_result.get('data'):
                total_views = views_result['data'][0].get('view_count', 0)
            elif isinstance(views_result, list) and not (len(views_result) > 0 and views_result[0].get('error')):
                total_views = views_result[0].get('view_count', 0)

        # Get indexes count
        indexes_query = "SELECT COUNT(*) as index_count FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        indexes_result = web_explorer.execute_query(indexes_query)
        total_indexes = 0
        if indexes_result:
            if isinstance(indexes_result, dict) and indexes_result.get('success') and indexes_result.get('data'):
                total_indexes = indexes_result['data'][0].get('index_count', 0)
            elif isinstance(indexes_result, list) and not (len(indexes_result) > 0 and indexes_result[0].get('error')):
                total_indexes = indexes_result[0].get('index_count', 0)

        # Get triggers count
        triggers_query = "SELECT COUNT(*) as trigger_count FROM sqlite_master WHERE type='trigger'"
        triggers_result = web_explorer.execute_query(triggers_query)
        total_triggers = 0
        if triggers_result:
            if isinstance(triggers_result, dict) and triggers_result.get('success') and triggers_result.get('data'):
                total_triggers = triggers_result['data'][0].get('trigger_count', 0)
            elif isinstance(triggers_result, list) and not (len(triggers_result) > 0 and triggers_result[0].get('error')):
                total_triggers = triggers_result[0].get('trigger_count', 0)

        # Analyze tables
        total_rows = 0
        total_columns = 0
        largest_table = {"name": None, "rows": 0}
        tables_with_data = 0
        empty_tables = 0

        for table_name in table_names:
            try:
                # Get row count
                count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
                count_result = web_explorer.execute_query(count_query)

                row_count = 0
                if count_result:
                    if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                        row_count = count_result['data'][0].get('row_count', 0)
                    elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                        row_count = count_result[0].get('row_count', 0)

                total_rows += row_count

                if row_count > 0:
                    tables_with_data += 1
                else:
                    empty_tables += 1

                if row_count > largest_table["rows"]:
                    largest_table = {"name": table_name, "rows": row_count}

                # Get column count
                columns_query = f"PRAGMA table_info(`{table_name}`)"
                columns_result = web_explorer.execute_query(columns_query)

                column_count = 0
                if columns_result:
                    if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                        column_count = len(columns_result['data'])
                    elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                        column_count = len(columns_result)

                total_columns += column_count

            except Exception as table_error:
                print(f"Error analyzing table {table_name}: {table_error}")
                continue

        # Overview statistics
        statistics["overview"] = {
            "total_tables": len(table_names),
            "total_views": total_views,
            "total_indexes": total_indexes,
            "total_triggers": total_triggers,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "file_size_mb": round(db_file_size / (1024 * 1024), 3),
            "database_type": "SQLite",
            "last_analyzed": datetime.now().isoformat()
        }

        # Table analysis
        statistics["table_analysis"] = {
            "tables_with_data": tables_with_data,
            "empty_tables": empty_tables,
            "largest_table": largest_table,
            "average_rows_per_table": round(total_rows / len(table_names), 2) if table_names else 0,
            "average_columns_per_table": round(total_columns / len(table_names), 2) if table_names else 0,
            "data_distribution": "Even" if empty_tables <= len(table_names) * 0.2 else "Uneven"
        }

        # Relationship analysis
        total_foreign_keys = 0
        tables_with_relationships = 0

        for table_name in table_names:
            try:
                fk_query = f"PRAGMA foreign_key_list(`{table_name}`)"
                fk_result = web_explorer.execute_query(fk_query)

                fk_count = 0
                if fk_result:
                    if isinstance(fk_result, dict) and fk_result.get('success') and fk_result.get('data'):
                        fk_count = len(fk_result['data'])
                    elif isinstance(fk_result, list) and not (len(fk_result) > 0 and fk_result[0].get('error')):
                        fk_count = len(fk_result)

                total_foreign_keys += fk_count
                if fk_count > 0:
                    tables_with_relationships += 1

            except Exception as fk_error:
                continue

        statistics["relationship_analysis"] = {
            "total_foreign_keys": total_foreign_keys,
            "tables_with_relationships": tables_with_relationships,
            "tables_without_relationships": len(table_names) - tables_with_relationships,
            "relationship_density": round((tables_with_relationships / len(table_names) * 100), 1) if table_names else 0,
            "average_fks_per_table": round(total_foreign_keys / len(table_names), 2) if table_names else 0
        }

        # Index analysis
        tables_with_indexes = 0
        for table_name in table_names:
            try:
                idx_query = f"PRAGMA index_list(`{table_name}`)"
                idx_result = web_explorer.execute_query(idx_query)

                if idx_result:
                    indexes = []
                    if isinstance(idx_result, dict) and idx_result.get('success') and idx_result.get('data'):
                        indexes = idx_result['data']
                    elif isinstance(idx_result, list) and not (len(idx_result) > 0 and idx_result[0].get('error')):
                        indexes = idx_result

                    # Count user-created indexes (not auto-generated)
                    user_indexes = [idx for idx in indexes if not idx.get('name', '').startswith('sqlite_')]
                    if len(user_indexes) > 0:
                        tables_with_indexes += 1

            except Exception as idx_error:
                continue

        statistics["index_analysis"] = {
            "total_user_indexes": total_indexes,
            "tables_with_indexes": tables_with_indexes,
            "tables_without_indexes": len(table_names) - tables_with_indexes,
            "index_coverage": round((tables_with_indexes / len(table_names) * 100), 1) if table_names else 0,
            "indexing_strategy": "Good" if tables_with_indexes > len(table_names) * 0.7 else "Needs Improvement"
        }

        # Constraint analysis (basic)
        tables_with_pk = 0
        for table_name in table_names:
            try:
                # Check for primary keys in table structure
                columns_query = f"PRAGMA table_info(`{table_name}`)"
                columns_result = web_explorer.execute_query(columns_query)

                if columns_result:
                    columns = []
                    if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                        columns = columns_result['data']
                    elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                        columns = columns_result

                    # Check if any column is a primary key
                    has_pk = any(col.get('pk', 0) == 1 for col in columns)
                    if has_pk:
                        tables_with_pk += 1

            except Exception as pk_error:
                continue

        statistics["constraint_analysis"] = {
            "tables_with_primary_keys": tables_with_pk,
            "tables_without_primary_keys": len(table_names) - tables_with_pk,
            "primary_key_coverage": round((tables_with_pk / len(table_names) * 100), 1) if table_names else 0,
            "total_foreign_keys": total_foreign_keys
        }

        # Storage analysis
        estimated_data_size = total_rows * 50  # Rough estimate: 50 bytes per row average
        estimated_index_overhead = total_indexes * 1024 * 10  # Rough estimate: 10KB per index

        statistics["storage_analysis"] = {
            "file_size_bytes": db_file_size,
            "file_size_mb": round(db_file_size / (1024 * 1024), 3),
            "estimated_data_size_mb": round(estimated_data_size / (1024 * 1024), 3),
            "estimated_index_overhead_mb": round(estimated_index_overhead / (1024 * 1024), 3),
            "storage_efficiency": round((estimated_data_size / max(db_file_size, 1)) * 100, 1),
            "largest_table_percentage": round((largest_table["rows"] / max(total_rows, 1)) * 100, 1)
        }

        # Data quality indicators
        data_quality_score = 100
        quality_issues = []

        if empty_tables > 0:
            data_quality_score -= (empty_tables / len(table_names)) * 20
            quality_issues.append(f"{empty_tables} empty tables")

        if tables_with_pk < len(table_names):
            missing_pk = len(table_names) - tables_with_pk
            data_quality_score -= (missing_pk / len(table_names)) * 30
            quality_issues.append(f"{missing_pk} tables without primary keys")

        if total_foreign_keys == 0:
            data_quality_score -= 25
            quality_issues.append("No foreign key relationships")

        statistics["data_quality"] = {
            "quality_score": max(0, round(data_quality_score, 1)),
            "quality_rating": "Excellent" if data_quality_score > 90 else "Good" if data_quality_score > 70 else "Fair" if data_quality_score > 50 else "Poor",
            "quality_issues": quality_issues,
            "recommendations": generate_quality_recommendations(statistics)
        }

        return jsonify({
            "success": True,
            "statistics": statistics
        })

    except Exception as e:
        print(f"Error generating schema statistics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


def generate_quality_recommendations(statistics):
    """Generate recommendations based on statistics"""
    recommendations = []

    # Check for empty tables
    if statistics["table_analysis"]["empty_tables"] > 0:
        recommendations.append("Consider removing or populating empty tables")

    # Check for missing primary keys
    if statistics["constraint_analysis"]["primary_key_coverage"] < 100:
        recommendations.append("Add primary keys to tables that don't have them")

    # Check for missing relationships
    if statistics["relationship_analysis"]["total_foreign_keys"] == 0:
        recommendations.append("Consider adding foreign key relationships to improve data integrity")

    # Check for indexing
    if statistics["index_analysis"]["index_coverage"] < 50:
        recommendations.append("Add indexes to improve query performance")

    # Check for large file size
    if statistics["storage_analysis"]["file_size_mb"] > 100:
        recommendations.append("Consider running VACUUM to optimize file size")

    if not recommendations:
        recommendations.append("Database schema appears to be well-designed")

    return recommendations


@app.route('/api/schema/maintenance/reindex')
def run_reindex():
    """Rebuild all indexes in the database"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Running REINDEX command...")

        # Run REINDEX
        reindex_result = web_explorer.execute_query("REINDEX")

        if not reindex_result:
            return jsonify({
                "success": False,
                "error": "REINDEX command failed"
            })

        # Check if REINDEX succeeded
        success = False
        error_message = None

        if isinstance(reindex_result, dict):
            success = reindex_result.get('success', False)
            error_message = reindex_result.get('error')
        elif isinstance(reindex_result, list):
            if len(reindex_result) > 0 and reindex_result[0].get('error'):
                error_message = reindex_result[0]['error']
            else:
                success = True

        if not success:
            return jsonify({
                "success": False,
                "error": error_message or "REINDEX command failed"
            })

        return jsonify({
            "success": True,
            "message": "REINDEX command completed successfully",
            "description": "All indexes have been rebuilt for optimal performance"
        })

    except Exception as e:
        print(f"Error in run_reindex: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/maintenance/optimize')
def run_full_optimization():
    """Run full database optimization (VACUUM + ANALYZE + REINDEX)"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Running full database optimization...")

        results = {
            "vacuum": None,
            "analyze": None,
            "reindex": None
        }

        # Get file size before optimization
        file_size_before = 0
        if web_explorer.db_path and os.path.exists(web_explorer.db_path):
            file_size_before = os.path.getsize(web_explorer.db_path)

        # Run VACUUM
        try:
            vacuum_result = web_explorer.execute_query("VACUUM")
            results["vacuum"] = {"success": True, "message": "VACUUM completed"}
        except Exception as vacuum_error:
            results["vacuum"] = {"success": False, "error": str(vacuum_error)}

        # Run ANALYZE
        try:
            analyze_result = web_explorer.execute_query("ANALYZE")
            results["analyze"] = {"success": True, "message": "ANALYZE completed"}
        except Exception as analyze_error:
            results["analyze"] = {"success": False, "error": str(analyze_error)}

        # Run REINDEX
        try:
            reindex_result = web_explorer.execute_query("REINDEX")
            results["reindex"] = {"success": True, "message": "REINDEX completed"}
        except Exception as reindex_error:
            results["reindex"] = {"success": False, "error": str(reindex_error)}

        # Get file size after optimization
        file_size_after = 0
        if web_explorer.db_path and os.path.exists(web_explorer.db_path):
            file_size_after = os.path.getsize(web_explorer.db_path)

        # Calculate space saved
        space_saved_bytes = file_size_before - file_size_after
        space_saved_mb = space_saved_bytes / (1024 * 1024)

        successful_operations = sum(1 for result in results.values() if result.get("success"))

        return jsonify({
            "success": True,
            "message": f"Optimization completed. {successful_operations}/3 operations successful.",
            "results": results,
            "file_size_before_mb": round(file_size_before / (1024 * 1024), 2),
            "file_size_after_mb": round(file_size_after / (1024 * 1024), 2),
            "space_saved_mb": round(space_saved_mb, 2),
            "space_saved_percent": round((space_saved_bytes / file_size_before * 100), 2) if file_size_before > 0 else 0
        })

    except Exception as e:
        print(f"Error in run_full_optimization: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/api/schema/health-check')
def run_health_check():
    """Run comprehensive database health check"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        print("Running database health check...")

        health_results = {
            "overall_score": 0,
            "checks": [],
            "recommendations": [],
            "status": "unknown"
        }

        # 1. Database integrity check
        try:
            integrity_query = "PRAGMA integrity_check"
            integrity_result = web_explorer.execute_query(integrity_query)

            if integrity_result:
                integrity_data = []
                if isinstance(integrity_result, dict) and integrity_result.get('success') and integrity_result.get('data'):
                    integrity_data = integrity_result['data']
                elif isinstance(integrity_result, list) and not (len(integrity_result) > 0 and integrity_result[0].get('error')):
                    integrity_data = integrity_result

                integrity_ok = any(str(row).lower() == 'ok' for row in integrity_data)

                health_results["checks"].append({
                    "name": "Database Integrity",
                    "status": "pass" if integrity_ok else "fail",
                    "message": "Database integrity is OK" if integrity_ok else "Database integrity issues found",
                    "score": 25 if integrity_ok else 0
                })

        except Exception as integrity_error:
            health_results["checks"].append({
                "name": "Database Integrity",
                "status": "error",
                "message": f"Could not check integrity: {str(integrity_error)}",
                "score": 0
            })

        # 2. Foreign key check
        try:
            fk_check_query = "PRAGMA foreign_key_check"
            fk_check_result = web_explorer.execute_query(fk_check_query)

            fk_violations = 0
            if fk_check_result:
                if isinstance(fk_check_result, dict) and fk_check_result.get('success') and fk_check_result.get('data'):
                    fk_violations = len(fk_check_result['data'])
                elif isinstance(fk_check_result, list) and not (len(fk_check_result) > 0 and fk_check_result[0].get('error')):
                    fk_violations = len(fk_check_result)

            health_results["checks"].append({
                "name": "Foreign Key Constraints",
                "status": "pass" if fk_violations == 0 else "warning",
                "message": f"No foreign key violations" if fk_violations == 0 else f"{fk_violations} foreign key violations found",
                "score": 20 if fk_violations == 0 else 10 if fk_violations < 5 else 0
            })

        except Exception as fk_error:
            health_results["checks"].append({
                "name": "Foreign Key Constraints",
                "status": "error",
                "message": f"Could not check foreign keys: {str(fk_error)}",
                "score": 0
            })

        # 3. Schema completeness check
        try:
            # Check for tables without primary keys
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            tables_result = web_explorer.execute_query(tables_query)

            table_names = []
            if tables_result:
                if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                    table_names = [row['name'] for row in tables_result['data']]
                elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                    table_names = [row['name'] for row in tables_result if 'name' in row]

            tables_without_pk = 0
            for table_name in table_names:
                try:
                    columns_query = f"PRAGMA table_info(`{table_name}`)"
                    columns_result = web_explorer.execute_query(columns_query)

                    has_pk = False
                    if columns_result:
                        columns = []
                        if isinstance(columns_result, dict) and columns_result.get('success') and columns_result.get('data'):
                            columns = columns_result['data']
                        elif isinstance(columns_result, list) and not (len(columns_result) > 0 and columns_result[0].get('error')):
                            columns = columns_result

                        has_pk = any(col.get('pk', 0) == 1 for col in columns)

                    if not has_pk:
                        tables_without_pk += 1

                except Exception:
                    continue

            pk_score = max(0, 20 - (tables_without_pk * 5))
            health_results["checks"].append({
                "name": "Schema Completeness",
                "status": "pass" if tables_without_pk == 0 else "warning",
                "message": f"All tables have primary keys" if tables_without_pk == 0 else f"{tables_without_pk} tables missing primary keys",
                "score": pk_score
            })

        except Exception as schema_error:
            health_results["checks"].append({
                "name": "Schema Completeness",
                "status": "error",
                "message": f"Could not check schema: {str(schema_error)}",
                "score": 0
            })

        # 4. Performance indicators
        try:
            # Check for tables with many rows but no indexes
            performance_score = 20  # Start with full score
            large_unindexed_tables = 0

            for table_name in table_names:
                try:
                    # Get row count
                    count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
                    count_result = web_explorer.execute_query(count_query)

                    row_count = 0
                    if count_result:
                        if isinstance(count_result, dict) and count_result.get('success') and count_result.get('data'):
                            row_count = count_result['data'][0].get('row_count', 0)
                        elif isinstance(count_result, list) and not (len(count_result) > 0 and count_result[0].get('error')):
                            row_count = count_result[0].get('row_count', 0)

                    if row_count > 1000:  # Large table
                        # Check for indexes
                        idx_query = f"PRAGMA index_list(`{table_name}`)"
                        idx_result = web_explorer.execute_query(idx_query)

                        has_user_indexes = False
                        if idx_result:
                            indexes = []
                            if isinstance(idx_result, dict) and idx_result.get('success') and idx_result.get('data'):
                                indexes = idx_result['data']
                            elif isinstance(idx_result, list) and not (len(idx_result) > 0 and idx_result[0].get('error')):
                                indexes = idx_result

                            has_user_indexes = any(not idx.get('name', '').startswith('sqlite_') for idx in indexes)

                        if not has_user_indexes:
                            large_unindexed_tables += 1
                            performance_score -= 5

                except Exception:
                    continue

            performance_score = max(0, performance_score)
            health_results["checks"].append({
                "name": "Performance Indicators",
                "status": "pass" if large_unindexed_tables == 0 else "warning",
                "message": f"Good indexing strategy" if large_unindexed_tables == 0 else f"{large_unindexed_tables} large tables without indexes",
                "score": performance_score
            })

        except Exception as perf_error:
            health_results["checks"].append({
                "name": "Performance Indicators",
                "status": "error",
                "message": f"Could not check performance: {str(perf_error)}",
                "score": 0
            })

        # 5. File system check
        try:
            file_score = 15
            file_status = "pass"
            file_message = "Database file is accessible"

            if web_explorer.db_path and os.path.exists(web_explorer.db_path):
                file_size = os.path.getsize(web_explorer.db_path)
                file_size_mb = file_size / (1024 * 1024)

                if file_size_mb > 500:  # Very large database
                    file_score = 10
                    file_status = "warning"
                    file_message = f"Large database file ({file_size_mb:.1f}MB) - consider optimization"
                elif file_size_mb > 100:  # Large database
                    file_score = 12
                    file_message = f"Database file size: {file_size_mb:.1f}MB"
                else:
                    file_message = f"Database file size: {file_size_mb:.1f}MB"

            health_results["checks"].append({
                "name": "File System",
                "status": file_status,
                "message": file_message,
                "score": file_score
            })

        except Exception as file_error:
            health_results["checks"].append({
                "name": "File System",
                "status": "error",
                "message": f"Could not check file system: {str(file_error)}",
                "score": 0
            })

        # Calculate overall score
        total_score = sum(check["score"] for check in health_results["checks"])
        health_results["overall_score"] = total_score

        # Determine overall status
        if total_score >= 90:
            health_results["status"] = "excellent"
        elif total_score >= 70:
            health_results["status"] = "good"
        elif total_score >= 50:
            health_results["status"] = "fair"
        else:
            health_results["status"] = "poor"

        # Generate recommendations
        failed_checks = [check for check in health_results["checks"] if check["status"] in ["fail", "error"]]
        warning_checks = [check for check in health_results["checks"] if check["status"] == "warning"]

        if failed_checks:
            health_results["recommendations"].append("Address critical database issues immediately")
        if warning_checks:
            health_results["recommendations"].append("Consider optimization to resolve warnings")
        if total_score < 70:
            health_results["recommendations"].append("Run maintenance operations (VACUUM, ANALYZE)")
        if not health_results["recommendations"]:
            health_results["recommendations"].append("Database is in good health - continue regular monitoring")

        return jsonify({
            "success": True,
            "health_check": health_results
        })

    except Exception as e:
        print(f"Error in run_health_check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

# Complete the debug foreign key detection endpoint from paste.txt

@app.route('/api/debug/foreign-key-detection')
def debug_foreign_key_detection():
    """Comprehensive debug for foreign key detection"""
    if not web_explorer.connected:
        return jsonify({"error": "No database connected"})

    try:
        debug_results = {
            "foreign_keys_enabled": None,
            "tables_analyzed": [],
            "parsing_results": [],
            "pragma_results": [],
            "final_foreign_keys": []
        }

        # Check if foreign keys are enabled
        try:
            enable_result = web_explorer.execute_query("PRAGMA foreign_keys = ON")
            status_result = web_explorer.execute_query("PRAGMA foreign_keys")
            debug_results["foreign_keys_enabled"] = {
                "enable_result": enable_result,
                "status": status_result
            }
        except Exception as e:
            debug_results["foreign_keys_enabled"] = {"error": str(e)}

        # Get all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = web_explorer.execute_query(tables_query)

        table_names = []
        if tables_result:
            if isinstance(tables_result, dict) and tables_result.get('success') and tables_result.get('data'):
                table_names = [row['name'] for row in tables_result['data']]
            elif isinstance(tables_result, list) and not (len(tables_result) > 0 and tables_result[0].get('error')):
                table_names = [row['name'] for row in tables_result if 'name' in row]

        debug_results["tables_analyzed"] = table_names

        # For each table, try both methods
        all_foreign_keys = []

        for table_name in table_names:
            table_debug = {
                "table": table_name,
                "create_sql": None,
                "sql_parsing_result": [],
                "pragma_result": [],
                "final_fks": []
            }

            # Method 1: Get CREATE TABLE SQL and parse it
            try:
                create_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                create_result = web_explorer.execute_query(create_query)

                create_sql = ""
                if create_result:
                    if isinstance(create_result, dict) and create_result.get('success') and create_result.get('data'):
                        if len(create_result['data']) > 0:
                            create_sql = create_result['data'][0].get('sql', '')
                    elif isinstance(create_result, list) and not (len(create_result) > 0 and create_result[0].get('error')):
                        if len(create_result) > 0:
                            create_sql = create_result[0].get('sql', '')

                table_debug["create_sql"] = create_sql

                # Parse foreign keys from SQL
                if create_sql:
                    import re

                    # Check for various FK patterns
                    patterns = [
                        # Pattern 1: FOREIGN KEY (column) REFERENCES table (column)
                        r'FOREIGN KEY\s*\(\s*([^)]+)\s*\)\s*REFERENCES\s+([^(\s]+)\s*\(\s*([^)]+)\s*\)',
                        # Pattern 2: column REFERENCES table(column)
                        r'(\w+)\s+[^,\n]*?\s+REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)',
                        # Pattern 3: More flexible REFERENCES pattern
                        r'([`"]?\w+[`"]?)\s+.*?REFERENCES\s+([`"]?\w+[`"]?)\s*\(\s*([`"]?\w+[`"]?)\s*\)'
                    ]

                    for i, pattern in enumerate(patterns):
                        matches = re.findall(pattern, create_sql, re.IGNORECASE | re.MULTILINE)
                        if matches:
                            table_debug["sql_parsing_result"].append({
                                "pattern": i + 1,
                                "matches": matches,
                                "pattern_regex": pattern
                            })

                            for match in matches:
                                if len(match) >= 3:
                                    from_column = match[0].strip().strip('`"\'')
                                    to_table = match[1].strip().strip('`"\'')
                                    to_column = match[2].strip().strip('`"\'')

                                    fk_info = {
                                        "table_name": table_name,
                                        "column": from_column,
                                        "referenced_table": to_table,
                                        "referenced_column": to_column,
                                        "detection_method": f"sql_pattern_{i+1}"
                                    }

                                    table_debug["final_fks"].append(fk_info)
                                    all_foreign_keys.append(fk_info)

            except Exception as sql_error:
                table_debug["sql_parsing_error"] = str(sql_error)

            # Method 2: Try PRAGMA foreign_key_list
            try:
                pragma_query = f"PRAGMA foreign_key_list('{table_name}')"
                pragma_result = web_explorer.execute_query(pragma_query)

                table_debug["pragma_result"] = pragma_result

                if pragma_result:
                    fk_data = []
                    if isinstance(pragma_result, dict) and pragma_result.get('success') and pragma_result.get('data'):
                        fk_data = pragma_result['data']
                    elif isinstance(pragma_result, list) and not (len(pragma_result) > 0 and pragma_result[0].get('error')):
                        fk_data = pragma_result

                    for fk in fk_data:
                        fk_info = {
                            "table_name": table_name,
                            "column": fk.get('from', ''),
                            "referenced_table": fk.get('table', ''),
                            "referenced_column": fk.get('to', ''),
                            "detection_method": "pragma"
                        }

                        # Check for duplicates
                        duplicate = any(
                            existing['table_name'] == fk_info['table_name'] and
                            existing['column'] == fk_info['column'] and
                            existing['referenced_table'] == fk_info['referenced_table']
                            for existing in table_debug["final_fks"]
                        )

                        if not duplicate:
                            table_debug["final_fks"].append(fk_info)
                            all_foreign_keys.append(fk_info)

            except Exception as pragma_error:
                table_debug["pragma_error"] = str(pragma_error)

            debug_results["parsing_results"].append(table_debug)

        debug_results["final_foreign_keys"] = all_foreign_keys

        return jsonify({
            "success": True,
            "debug_results": debug_results,
            "summary": {
                "total_tables": len(table_names),
                "total_foreign_keys_found": len(all_foreign_keys),
                "tables_with_fks": len([t for t in debug_results["parsing_results"] if len(t["final_fks"]) > 0])
            }
        })

    except Exception as e:
        print(f"Error in debug_foreign_key_detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 100MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

# Context processor to make functions available in templates
@app.context_processor
def utility_processor():
    return dict(
        enumerate=enumerate,
        len=len,
        str=str,
        datetime=datetime
    )