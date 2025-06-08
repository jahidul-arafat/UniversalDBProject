#!/usr/bin/env python3
"""
Universal Database Explorer - Interactive CLI Tool
Usage: python db_explorer.py <database_file>
Example: python db_explorer.py student_database.db
"""

import sys
import os
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduced noise for CLI
logger = logging.getLogger(__name__)

# Optional imports for different database types
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

@dataclass
class TableInfo:
    """Information about a database table"""
    name: str
    columns: List[Dict[str, Any]]
    row_count: int
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[Dict[str, Any]]
    triggers: List[Dict[str, str]]

@dataclass
class DatabaseSchema:
    """Complete database schema information"""
    database_type: str
    database_name: str
    tables: List[TableInfo]
    views: List[str]
    triggers: List[Dict[str, str]]
    indexes: List[Dict[str, Any]]

class DatabaseConnector(ABC):
    """Abstract base class for database connectors"""

    @abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_schema(self) -> DatabaseSchema:
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_table_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def close(self):
        pass

class SQLiteConnector(DatabaseConnector):
    """SQLite database connector with advanced operations"""

    def __init__(self):
        self.connection = None
        self.database_path = None

    def connect(self, connection_params: Dict[str, Any]) -> bool:
        try:
            self.database_path = connection_params.get('database', ':memory:')
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to SQLite: {e}")
            return False

    def get_schema(self) -> DatabaseSchema:
        tables = []
        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]

        for table_name in table_names:
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            primary_keys = []

            for col_info in cursor.fetchall():
                col_dict = {
                    'name': col_info[1],
                    'type': col_info[2],
                    'nullable': not col_info[3],
                    'default': col_info[4]
                }
                columns.append(col_dict)
                if col_info[5]:  # Primary key
                    primary_keys.append(col_info[1])

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = []
            for fk_info in cursor.fetchall():
                foreign_keys.append({
                    'column': fk_info[3],
                    'referenced_table': fk_info[2],
                    'referenced_column': fk_info[4]
                })

            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = []
            for idx_info in cursor.fetchall():
                cursor.execute(f"PRAGMA index_info({idx_info[1]})")
                columns_in_index = [col[2] for col in cursor.fetchall()]
                indexes.append({
                    'name': idx_info[1],
                    'unique': bool(idx_info[2]),
                    'columns': columns_in_index
                })

            # Get triggers for this table
            cursor.execute("""
                           SELECT name, sql FROM sqlite_master
                           WHERE type='trigger' AND tbl_name=?
                           """, (table_name,))
            triggers = []
            for trigger_info in cursor.fetchall():
                triggers.append({
                    'name': trigger_info[0],
                    'sql': trigger_info[1]
                })

            table_info = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                indexes=indexes,
                triggers=triggers
            )
            tables.append(table_info)

        # Get views
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = [row[0] for row in cursor.fetchall()]

        # Get all triggers
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
        all_triggers = []
        for trigger_info in cursor.fetchall():
            all_triggers.append({
                'name': trigger_info[0],
                'sql': trigger_info[1]
            })

        # Get all indexes
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
        all_indexes = []
        for idx_info in cursor.fetchall():
            all_indexes.append({
                'name': idx_info[0],
                'sql': idx_info[1]
            })

        return DatabaseSchema(
            database_type='SQLite',
            database_name=self.database_path,
            tables=tables,
            views=views,
            triggers=all_triggers,
            indexes=all_indexes
        )

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if query.strip().upper().startswith('SELECT'):
                columns = [description[0] for description in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            else:
                self.connection.commit()
                return [{'affected_rows': cursor.rowcount, 'status': 'success'}]
        except Exception as e:
            return [{'error': str(e), 'status': 'failed'}]

    def get_table_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        query = f"SELECT * FROM {table_name} LIMIT ?"
        return self.execute_query(query, (limit,))

    def close(self):
        if self.connection:
            self.connection.close()

class UniversalDatabaseExplorer:
    """Complete universal database exploration and operations toolkit"""

    def __init__(self):
        self.connector = None
        self.schema = None
        self.supported_databases = {
            'sqlite': SQLiteConnector,
        }

    def connect(self, db_type: str, connection_params: Dict[str, Any]) -> bool:
        """Connect to a database"""
        db_type = db_type.lower()

        if db_type not in self.supported_databases:
            print(f"❌ Unsupported database type: {db_type}")
            return False

        connector_class = self.supported_databases[db_type]
        self.connector = connector_class()

        if self.connector.connect(connection_params):
            self.schema = self.connector.get_schema()
            return True
        return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get complete database information"""
        if not self.schema:
            return {}

        return {
            'database_type': self.schema.database_type,
            'database_name': self.schema.database_name,
            'total_tables': len(self.schema.tables),
            'total_views': len(self.schema.views),
            'total_triggers': len(self.schema.triggers),
            'total_indexes': len(self.schema.indexes),
            'tables': [
                {
                    'name': table.name,
                    'columns': len(table.columns),
                    'rows': table.row_count,
                    'primary_keys': table.primary_keys,
                    'foreign_keys': len(table.foreign_keys),
                    'indexes': len(table.indexes),
                    'triggers': len(table.triggers)
                }
                for table in self.schema.tables
            ]
        }

    def get_table_structure(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed structure of a specific table"""
        if not self.schema:
            return None

        for table in self.schema.tables:
            if table.name == table_name:
                return {
                    'name': table.name,
                    'columns': table.columns,
                    'row_count': table.row_count,
                    'primary_keys': table.primary_keys,
                    'foreign_keys': table.foreign_keys,
                    'indexes': table.indexes,
                    'triggers': table.triggers
                }
        return None

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute any SQL query"""
        if not self.connector:
            return []
        return self.connector.execute_query(query, params)

    def get_sample_data(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample data from a table"""
        if not self.connector:
            return []
        return self.connector.get_table_data(table_name, limit)

    def inner_join(self, table1: str, table2: str, join_condition: str,
                   columns: str = "*", where_clause: str = "") -> List[Dict[str, Any]]:
        """Perform INNER JOIN between two tables"""
        query = f"""
            SELECT {columns}
            FROM {table1}
            INNER JOIN {table2} ON {join_condition}
            {where_clause}
        """
        return self.execute_query(query)

    def search_tables(self, keyword: str) -> List[str]:
        """Search for tables containing a keyword"""
        if not self.schema:
            return []

        matching_tables = []
        keyword = keyword.lower()

        for table in self.schema.tables:
            if keyword in table.name.lower():
                matching_tables.append(table.name)
            else:
                for column in table.columns:
                    if keyword in column['name'].lower():
                        matching_tables.append(table.name)
                        break

        return matching_tables

    def get_related_tables(self, table_name: str) -> List[str]:
        """Find tables related through foreign keys"""
        if not self.schema:
            return []

        related = []
        target_table = None

        for table in self.schema.tables:
            if table.name == table_name:
                target_table = table
                break

        if not target_table:
            return []

        # Find tables referenced by foreign keys
        for fk in target_table.foreign_keys:
            if fk['referenced_table'] not in related:
                related.append(fk['referenced_table'])

        # Find tables that reference this table
        for table in self.schema.tables:
            for fk in table.foreign_keys:
                if fk['referenced_table'] == table_name and table.name not in related:
                    related.append(table.name)

        return related

    def close(self):
        """Close database connection"""
        if self.connector:
            self.connector.close()

class DatabaseCLI:
    """Interactive CLI for database exploration"""

    def __init__(self, db_file: str):
        self.db_file = db_file
        self.explorer = UniversalDatabaseExplorer()
        self.connected = False

    def connect_to_database(self) -> bool:
        """Connect to the database file"""
        if not os.path.exists(self.db_file):
            print(f"❌ Database file '{self.db_file}' not found!")
            return False

        print(f"🔗 Connecting to database: {self.db_file}")

        # Determine database type based on file extension
        if self.db_file.endswith(('.db', '.sqlite', '.sqlite3')):
            db_type = 'sqlite'
        else:
            db_type = 'sqlite'  # Default to SQLite

        if self.explorer.connect(db_type, {'database': self.db_file}):
            print("✅ Successfully connected!")
            self.connected = True
            return True
        else:
            print("❌ Failed to connect to database")
            return False

    def display_database_overview(self):
        """Display initial database characteristics"""
        if not self.connected:
            return

        print("\n" + "="*60)
        print("📊 DATABASE OVERVIEW")
        print("="*60)

        db_info = self.explorer.get_database_info()

        print(f"📁 Database File: {self.db_file}")
        print(f"🗃️  Database Type: {db_info.get('database_type', 'Unknown')}")
        print(f"📋 Total Tables: {db_info.get('total_tables', 0)}")
        print(f"👁️  Total Views: {db_info.get('total_views', 0)}")
        print(f"⚡ Total Triggers: {db_info.get('total_triggers', 0)}")
        print(f"🔍 Total Indexes: {db_info.get('total_indexes', 0)}")

        # Display table summary
        tables = db_info.get('tables', [])
        if tables:
            print(f"\n📊 TABLES SUMMARY:")
            print("-" * 60)
            print(f"{'Table Name':<25} {'Columns':<10} {'Rows':<10} {'PKs':<5} {'FKs':<5}")
            print("-" * 60)

            for table in tables:
                pk_count = len(table.get('primary_keys', []))
                fk_count = table.get('foreign_keys', 0)
                print(f"{table['name']:<25} {table['columns']:<10} {table['rows']:<10} {pk_count:<5} {fk_count:<5}")

        # Show file size
        try:
            file_size = os.path.getsize(self.db_file)
            size_mb = file_size / (1024 * 1024)
            print(f"\n💾 File Size: {size_mb:.2f} MB ({file_size:,} bytes)")
        except:
            pass

    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🛠️  DATABASE OPERATIONS MENU")
        print("="*60)

        menu_options = [
            ("1", "📋 View Database Overview", "Show database characteristics"),
            ("2", "📊 List All Tables", "Display all tables with details"),
            ("3", "🔍 Explore Specific Table", "View table structure and sample data"),
            ("4", "🔎 Search Tables", "Search tables by keyword"),
            ("5", "🔗 Show Table Relationships", "Display foreign key relationships"),
            ("6", "📝 Execute Custom Query", "Run your own SQL query"),
            ("7", "🔄 JOIN Operations", "Perform table joins"),
            ("8", "📈 Data Analysis", "Aggregate and analyze data"),
            ("9", "🏗️  Schema Operations", "Create/modify tables and indexes"),
            ("10", "⚡ Trigger Management", "View and manage triggers"),
            ("11", "🔍 Performance Analysis", "Analyze table performance"),
            ("12", "📤 Export Operations", "Export data and schema"),
            ("13", "🧪 Data Validation", "Check data integrity"),
            ("0", "🚪 Exit", "Close the application"),
        ]

        for option, title, description in menu_options:
            print(f"{option:>2}. {title:<30} - {description}")

        print("-" * 60)

    def handle_menu_choice(self, choice: str):
        """Handle user menu selection"""
        try:
            if choice == "1":
                self.display_database_overview()
            elif choice == "2":
                self.list_all_tables()
            elif choice == "3":
                self.explore_table()
            elif choice == "4":
                self.search_tables()
            elif choice == "5":
                self.show_relationships()
            elif choice == "6":
                self.execute_custom_query()
            elif choice == "7":
                self.join_operations()
            elif choice == "8":
                self.data_analysis()
            elif choice == "9":
                self.schema_operations()
            elif choice == "10":
                self.trigger_management()
            elif choice == "11":
                self.performance_analysis()
            elif choice == "12":
                self.export_operations()
            elif choice == "13":
                self.data_validation()
            elif choice == "0":
                print("👋 Goodbye! Closing database connection...")
                self.explorer.close()
                return False
            else:
                print("❌ Invalid choice. Please try again.")
        except Exception as e:
            print(f"❌ Error: {e}")

        return True

    def list_all_tables(self):
        """List all tables with detailed information"""
        print("\n📋 ALL TABLES")
        print("-" * 80)

        db_info = self.explorer.get_database_info()
        tables = db_info.get('tables', [])

        if not tables:
            print("No tables found in the database.")
            return

        for i, table in enumerate(tables, 1):
            print(f"\n{i}. Table: {table['name']}")
            print(f"   📊 Columns: {table['columns']}")
            print(f"   📄 Rows: {table['rows']}")
            print(f"   🔑 Primary Keys: {', '.join(table.get('primary_keys', []))}")
            print(f"   🔗 Foreign Keys: {table.get('foreign_keys', 0)}")
            print(f"   🔍 Indexes: {table.get('indexes', 0)}")
            print(f"   ⚡ Triggers: {table.get('triggers', 0)}")

    def explore_table(self):
        """Explore a specific table in detail"""
        table_name = input("\n🔍 Enter table name to explore: ").strip()

        if not table_name:
            print("❌ Table name cannot be empty.")
            return

        structure = self.explorer.get_table_structure(table_name)
        if not structure:
            print(f"❌ Table '{table_name}' not found.")
            return

        print(f"\n📊 TABLE STRUCTURE: {table_name}")
        print("-" * 60)

        # Column details
        print("📋 COLUMNS:")
        for col in structure['columns']:
            nullable = "NULL" if col['nullable'] else "NOT NULL"
            default = f" DEFAULT {col['default']}" if col['default'] else ""
            print(f"  • {col['name']:<20} {col['type']:<15} {nullable}{default}")

        # Primary keys
        if structure['primary_keys']:
            print(f"\n🔑 PRIMARY KEYS: {', '.join(structure['primary_keys'])}")

        # Foreign keys
        if structure['foreign_keys']:
            print("\n🔗 FOREIGN KEYS:")
            for fk in structure['foreign_keys']:
                print(f"  • {fk['column']} → {fk['referenced_table']}.{fk['referenced_column']}")

        # Sample data
        print(f"\n📄 SAMPLE DATA (first 5 rows):")
        sample_data = self.explorer.get_sample_data(table_name, 5)
        if sample_data:
            # Display as simple table
            if sample_data:
                headers = list(sample_data[0].keys())
                print("  " + " | ".join(f"{h:<15}" for h in headers))
                print("  " + "-" * (len(headers) * 18))
                for row in sample_data:
                    values = [str(row.get(h, ''))[:15] for h in headers]
                    print("  " + " | ".join(f"{v:<15}" for v in values))
        else:
            print("  No data found in table.")

    def search_tables(self):
        """Search tables by keyword"""
        keyword = input("\n🔎 Enter search keyword: ").strip()

        if not keyword:
            print("❌ Keyword cannot be empty.")
            return

        matching_tables = self.explorer.search_tables(keyword)

        print(f"\n🔍 SEARCH RESULTS for '{keyword}':")
        if matching_tables:
            for table in matching_tables:
                print(f"  • {table}")
        else:
            print("  No matching tables found.")

    def show_relationships(self):
        """Show table relationships"""
        print("\n🔗 TABLE RELATIONSHIPS")
        print("-" * 50)

        db_info = self.explorer.get_database_info()
        tables = db_info.get('tables', [])

        for table in tables:
            related = self.explorer.get_related_tables(table['name'])
            if related:
                print(f"\n📊 {table['name']} is related to:")
                for rel_table in related:
                    print(f"  → {rel_table}")

    def execute_custom_query(self):
        """Execute a custom SQL query"""
        print("\n📝 CUSTOM QUERY EXECUTION")
        print("Enter your SQL query (type 'DONE' on a new line to execute):")

        query_lines = []
        while True:
            line = input("SQL> ").strip()
            if line.upper() == 'DONE':
                break
            query_lines.append(line)

        query = " ".join(query_lines).strip()

        if not query:
            print("❌ No query entered.")
            return

        print(f"\n🚀 Executing query: {query}")

        try:
            results = self.explorer.execute_query(query)

            if results:
                if 'error' in results[0]:
                    print(f"❌ Query failed: {results[0]['error']}")
                elif query.upper().startswith('SELECT'):
                    print(f"\n✅ Query executed successfully! Found {len(results)} rows:")

                    # Display results in table format
                    if results and len(results) > 0:
                        headers = list(results[0].keys())
                        print("\n" + " | ".join(f"{h:<15}" for h in headers))
                        print("-" * (len(headers) * 18))

                        for row in results[:10]:  # Show first 10 rows
                            values = [str(row.get(h, ''))[:15] for h in headers]
                            print(" | ".join(f"{v:<15}" for v in values))

                        if len(results) > 10:
                            print(f"... and {len(results) - 10} more rows")
                else:
                    print(f"✅ Query executed successfully! Affected rows: {results[0].get('affected_rows', 0)}")
            else:
                print("✅ Query executed successfully! No results returned.")

        except Exception as e:
            print(f"❌ Error executing query: {e}")

    def join_operations(self):
        """Perform table join operations"""
        print("\n🔄 JOIN OPERATIONS")

        # List available tables
        db_info = self.explorer.get_database_info()
        tables = [table['name'] for table in db_info.get('tables', [])]

        if len(tables) < 2:
            print("❌ Need at least 2 tables to perform joins.")
            return

        print("Available tables:", ", ".join(tables))

        table1 = input("Enter first table name: ").strip()
        table2 = input("Enter second table name: ").strip()
        join_condition = input("Enter join condition (e.g., table1.id = table2.user_id): ").strip()

        if not all([table1, table2, join_condition]):
            print("❌ All fields are required.")
            return

        try:
            results = self.explorer.inner_join(table1, table2, join_condition)

            if results:
                print(f"\n✅ JOIN executed successfully! Found {len(results)} rows:")

                if results and len(results) > 0:
                    headers = list(results[0].keys())
                    print("\n" + " | ".join(f"{h:<15}" for h in headers))
                    print("-" * (len(headers) * 18))

                    for row in results[:5]:  # Show first 5 rows
                        values = [str(row.get(h, ''))[:15] for h in headers]
                        print(" | ".join(f"{v:<15}" for v in values))
            else:
                print("No results found for the join operation.")

        except Exception as e:
            print(f"❌ Error executing join: {e}")

    def data_analysis(self):
        """Perform basic data analysis"""
        print("\n📈 DATA ANALYSIS")

        table_name = input("Enter table name for analysis: ").strip()
        if not table_name:
            return

        try:
            # Basic statistics
            count_result = self.explorer.execute_query(f"SELECT COUNT(*) as total_rows FROM {table_name}")
            if count_result:
                print(f"\n📊 Total rows in {table_name}: {count_result[0]['total_rows']}")

            # Show sample aggregation
            structure = self.explorer.get_table_structure(table_name)
            if structure:
                numeric_columns = [col['name'] for col in structure['columns']
                                   if 'INT' in col['type'].upper() or 'REAL' in col['type'].upper()
                                   or 'NUMERIC' in col['type'].upper()]

                if numeric_columns:
                    print(f"\n📈 Numeric column analysis:")
                    for col in numeric_columns[:3]:  # First 3 numeric columns
                        stats = self.explorer.execute_query(f"""
                            SELECT 
                                MIN({col}) as min_val,
                                MAX({col}) as max_val,
                                AVG({col}) as avg_val
                            FROM {table_name}
                            WHERE {col} IS NOT NULL
                        """)
                        if stats and stats[0]:
                            print(f"  {col}: MIN={stats[0]['min_val']}, MAX={stats[0]['max_val']}, AVG={stats[0]['avg_val']:.2f}")

        except Exception as e:
            print(f"❌ Error in analysis: {e}")

    def schema_operations(self):
        """Schema management operations"""
        print("\n🏗️  SCHEMA OPERATIONS")
        print("1. Create new table")
        print("2. Create index")
        print("3. Create view")

        choice = input("Choose operation (1-3): ").strip()

        if choice == "1":
            self.create_table_interactive()
        elif choice == "2":
            self.create_index_interactive()
        elif choice == "3":
            self.create_view_interactive()

    def create_table_interactive(self):
        """Interactive table creation"""
        table_name = input("Enter new table name: ").strip()
        if not table_name:
            return

        print("Enter column definitions (name type, e.g., 'id INTEGER PRIMARY KEY'):")
        print("Type 'DONE' when finished:")

        columns = {}
        while True:
            col_def = input("Column> ").strip()
            if col_def.upper() == 'DONE':
                break

            if ' ' in col_def:
                name, type_def = col_def.split(' ', 1)
                columns[name] = type_def

        if columns:
            try:
                # Simple table creation
                col_defs = [f"{name} {type_def}" for name, type_def in columns.items()]
                query = f"CREATE TABLE {table_name} ({', '.join(col_defs)})"
                result = self.explorer.execute_query(query)

                if result and result[0].get('status') == 'success':
                    print(f"✅ Table '{table_name}' created successfully!")
                else:
                    print(f"❌ Failed to create table: {result[0].get('error', 'Unknown error')}")

            except Exception as e:
                print(f"❌ Error creating table: {e}")

    def create_index_interactive(self):
        """Interactive index creation"""
        index_name = input("Enter index name: ").strip()
        table_name = input("Enter table name: ").strip()
        columns = input("Enter columns (comma-separated): ").strip()

        if not all([index_name, table_name, columns]):
            print("❌ All fields are required.")
            return

        try:
            query = f"CREATE INDEX {index_name} ON {table_name} ({columns})"
            result = self.explorer.execute_query(query)

            if result and result[0].get('status') == 'success':
                print(f"✅ Index '{index_name}' created successfully!")
            else:
                print(f"❌ Failed to create index: {result[0].get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error creating index: {e}")

    def create_view_interactive(self):
        """Interactive view creation"""
        view_name = input("Enter view name: ").strip()

        print("Enter SELECT query for the view:")
        print("Type 'DONE' on a new line when finished:")

        query_lines = []
        while True:
            line = input("SQL> ").strip()
            if line.upper() == 'DONE':
                break
            query_lines.append(line)

        select_query = " ".join(query_lines).strip()

        if not all([view_name, select_query]):
            print("❌ Both view name and query are required.")
            return

        try:
            query = f"CREATE VIEW {view_name} AS {select_query}"
            result = self.explorer.execute_query(query)

            if result and result[0].get('status') == 'success':
                print(f"✅ View '{view_name}' created successfully!")
            else:
                print(f"❌ Failed to create view: {result[0].get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error creating view: {e}")

    def trigger_management(self):
        """Manage database triggers"""
        print("\n⚡ TRIGGER MANAGEMENT")

        # Show existing triggers
        db_info = self.explorer.get_database_info()
        if db_info.get('total_triggers', 0) > 0:
            print("\n📋 EXISTING TRIGGERS:")

            # Get triggers from schema
            if hasattr(self.explorer, 'schema') and self.explorer.schema:
                for trigger in self.explorer.schema.triggers:
                    print(f"  • {trigger['name']}")
                    if trigger.get('sql'):
                        # Show first line of trigger SQL
                        first_line = trigger['sql'].split('\n')[0]
                        print(f"    SQL: {first_line}...")
        else:
            print("No triggers found in database.")

        print("\nTrigger Operations:")
        print("1. Create new trigger")
        print("2. View trigger details")
        print("3. Drop trigger")

        choice = input("Choose operation (1-3): ").strip()

        if choice == "1":
            self.create_trigger_interactive()
        elif choice == "2":
            self.view_trigger_details()
        elif choice == "3":
            self.drop_trigger_interactive()

    def create_trigger_interactive(self):
        """Interactive trigger creation"""
        print("\n🔧 CREATE TRIGGER")

        trigger_name = input("Enter trigger name: ").strip()
        table_name = input("Enter table name: ").strip()

        print("Select trigger event:")
        print("1. BEFORE INSERT")
        print("2. AFTER INSERT")
        print("3. BEFORE UPDATE")
        print("4. AFTER UPDATE")
        print("5. BEFORE DELETE")
        print("6. AFTER DELETE")

        event_choice = input("Choose event (1-6): ").strip()

        events = {
            "1": "BEFORE INSERT",
            "2": "AFTER INSERT",
            "3": "BEFORE UPDATE",
            "4": "AFTER UPDATE",
            "5": "BEFORE DELETE",
            "6": "AFTER DELETE"
        }

        trigger_event = events.get(event_choice)
        if not trigger_event:
            print("❌ Invalid event choice.")
            return

        print(f"\nEnter trigger body for {trigger_event} on {table_name}:")
        print("Example: UPDATE table_name SET modified_at = CURRENT_TIMESTAMP WHERE id = NEW.id;")
        print("Type 'DONE' when finished:")

        body_lines = []
        while True:
            line = input("SQL> ").strip()
            if line.upper() == 'DONE':
                break
            body_lines.append(line)

        trigger_body = " ".join(body_lines).strip()

        if not all([trigger_name, table_name, trigger_body]):
            print("❌ All fields are required.")
            return

        try:
            # SQLite trigger syntax
            query = f"""
                CREATE TRIGGER {trigger_name}
                {trigger_event} ON {table_name}
                BEGIN
                    {trigger_body}
                END
            """

            result = self.explorer.execute_query(query)

            if result and result[0].get('status') == 'success':
                print(f"✅ Trigger '{trigger_name}' created successfully!")
                # Refresh schema
                self.explorer.schema = self.explorer.connector.get_schema()
            else:
                print(f"❌ Failed to create trigger: {result[0].get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error creating trigger: {e}")

    def view_trigger_details(self):
        """View detailed trigger information"""
        trigger_name = input("Enter trigger name to view: ").strip()

        if not trigger_name:
            print("❌ Trigger name cannot be empty.")
            return

        # Find trigger in schema
        if hasattr(self.explorer, 'schema') and self.explorer.schema:
            for trigger in self.explorer.schema.triggers:
                if trigger['name'] == trigger_name:
                    print(f"\n⚡ TRIGGER DETAILS: {trigger_name}")
                    print("-" * 50)
                    print("SQL Definition:")
                    print(trigger.get('sql', 'No SQL definition available'))
                    return

        print(f"❌ Trigger '{trigger_name}' not found.")

    def drop_trigger_interactive(self):
        """Interactive trigger deletion"""
        trigger_name = input("Enter trigger name to drop: ").strip()

        if not trigger_name:
            print("❌ Trigger name cannot be empty.")
            return

        confirm = input(f"⚠️  Are you sure you want to drop trigger '{trigger_name}'? (y/N): ").strip().lower()

        if confirm != 'y':
            print("Operation cancelled.")
            return

        try:
            query = f"DROP TRIGGER {trigger_name}"
            result = self.explorer.execute_query(query)

            if result and result[0].get('status') == 'success':
                print(f"✅ Trigger '{trigger_name}' dropped successfully!")
                # Refresh schema
                self.explorer.schema = self.explorer.connector.get_schema()
            else:
                print(f"❌ Failed to drop trigger: {result[0].get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error dropping trigger: {e}")

    def performance_analysis(self):
        """Analyze database performance"""
        print("\n📈 PERFORMANCE ANALYSIS")

        table_name = input("Enter table name to analyze (or press Enter for all tables): ").strip()

        if table_name:
            self.analyze_single_table_performance(table_name)
        else:
            self.analyze_all_tables_performance()

    def analyze_single_table_performance(self, table_name: str):
        """Analyze performance for a single table"""
        structure = self.explorer.get_table_structure(table_name)
        if not structure:
            print(f"❌ Table '{table_name}' not found.")
            return

        print(f"\n📊 PERFORMANCE ANALYSIS: {table_name}")
        print("-" * 50)

        # Basic stats
        print(f"📄 Total Rows: {structure['row_count']:,}")
        print(f"📋 Total Columns: {len(structure['columns'])}")
        print(f"🔑 Primary Keys: {len(structure['primary_keys'])}")
        print(f"🔗 Foreign Keys: {len(structure['foreign_keys'])}")
        print(f"🔍 Indexes: {len(structure['indexes'])}")

        # Estimate memory usage (rough calculation)
        estimated_size = structure['row_count'] * len(structure['columns']) * 50  # bytes
        print(f"💾 Estimated Size: {estimated_size / (1024*1024):.2f} MB")

        # Index analysis
        if structure['indexes']:
            print(f"\n🔍 INDEX ANALYSIS:")
            for idx in structure['indexes']:
                print(f"  • {idx['name']}: {', '.join(idx.get('columns', []))}")
                if idx.get('unique'):
                    print(f"    Type: UNIQUE")

        # Suggest optimizations
        print(f"\n💡 OPTIMIZATION SUGGESTIONS:")

        # Check for missing indexes on foreign keys
        unindexed_fks = []
        for fk in structure['foreign_keys']:
            fk_col = fk['column']
            # Check if there's an index on this foreign key column
            has_index = any(fk_col in idx.get('columns', []) for idx in structure['indexes'])
            if not has_index:
                unindexed_fks.append(fk_col)

        if unindexed_fks:
            print(f"  • Consider adding indexes on foreign key columns: {', '.join(unindexed_fks)}")

        if structure['row_count'] > 10000 and len(structure['indexes']) < 2:
            print(f"  • Large table with few indexes - consider adding more indexes")

        if len(structure['columns']) > 20:
            print(f"  • Wide table ({len(structure['columns'])} columns) - consider normalization")

    def analyze_all_tables_performance(self):
        """Analyze performance for all tables"""
        print("\n📊 DATABASE-WIDE PERFORMANCE ANALYSIS")
        print("-" * 60)

        db_info = self.explorer.get_database_info()
        tables = db_info.get('tables', [])

        if not tables:
            print("No tables found in database.")
            return

        total_rows = sum(table['rows'] for table in tables)
        total_columns = sum(table['columns'] for table in tables)

        print(f"📈 Overall Statistics:")
        print(f"  • Total Tables: {len(tables)}")
        print(f"  • Total Rows: {total_rows:,}")
        print(f"  • Total Columns: {total_columns}")
        print(f"  • Average Rows per Table: {total_rows / len(tables):.0f}")

        print(f"\n📊 Tables by Size:")
        sorted_tables = sorted(tables, key=lambda x: x['rows'], reverse=True)

        for i, table in enumerate(sorted_tables[:10], 1):  # Top 10 largest tables
            print(f"  {i:2}. {table['name']:<25} {table['rows']:>10,} rows")

        # Find tables that might need attention
        print(f"\n⚠️  Tables Needing Attention:")

        for table in tables:
            issues = []

            # Large table with no indexes
            if table['rows'] > 5000 and table['indexes'] == 0:
                issues.append("no indexes")

            # Tables with many foreign keys but few indexes
            if table['foreign_keys'] > 2 and table['indexes'] < table['foreign_keys']:
                issues.append("unindexed foreign keys")

            # Very wide tables
            if table['columns'] > 15:
                issues.append(f"wide table ({table['columns']} columns)")

            if issues:
                print(f"  • {table['name']}: {', '.join(issues)}")

    def export_operations(self):
        """Handle export operations"""
        print("\n📤 EXPORT OPERATIONS")
        print("1. Export table data to SQL")
        print("2. Export table structure only")
        print("3. Export entire database schema")
        print("4. Export table to CSV format")

        choice = input("Choose export option (1-4): ").strip()

        if choice == "1":
            self.export_table_data()
        elif choice == "2":
            self.export_table_structure()
        elif choice == "3":
            self.export_database_schema()
        elif choice == "4":
            self.export_table_csv()

    def export_table_data(self):
        """Export table data as SQL INSERT statements"""
        table_name = input("Enter table name to export: ").strip()

        if not table_name:
            print("❌ Table name cannot be empty.")
            return

        try:
            # Get all data from table
            data = self.explorer.get_sample_data(table_name, limit=1000000)  # Large limit

            if not data:
                print(f"❌ No data found in table '{table_name}'.")
                return

            output_file = f"{table_name}_data.sql"

            with open(output_file, 'w') as f:
                f.write(f"-- Data export for table: {table_name}\n")
                f.write(f"-- Generated by Universal Database Explorer\n\n")

                for row in data:
                    columns = list(row.keys())
                    values = []

                    for value in row.values():
                        if value is None:
                            values.append('NULL')
                        elif isinstance(value, str):
                            # Escape single quotes
                            escaped_value = value.replace("'", "''")
                            values.append(f"'{escaped_value}'")
                        else:
                            values.append(str(value))

                    insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});\n"
                    f.write(insert_stmt)

            print(f"✅ Table data exported to: {output_file}")
            print(f"📊 Exported {len(data)} rows")

        except Exception as e:
            print(f"❌ Error exporting table data: {e}")

    def export_table_structure(self):
        """Export table structure as CREATE TABLE statement"""
        table_name = input("Enter table name to export structure: ").strip()

        if not table_name:
            print("❌ Table name cannot be empty.")
            return

        structure = self.explorer.get_table_structure(table_name)
        if not structure:
            print(f"❌ Table '{table_name}' not found.")
            return

        try:
            output_file = f"{table_name}_structure.sql"

            with open(output_file, 'w') as f:
                f.write(f"-- Table structure for: {table_name}\n")
                f.write(f"-- Generated by Universal Database Explorer\n\n")

                # CREATE TABLE statement
                f.write(f"CREATE TABLE {table_name} (\n")

                # Columns
                column_defs = []
                for col in structure['columns']:
                    col_def = f"    {col['name']} {col['type']}"
                    if not col['nullable']:
                        col_def += " NOT NULL"
                    if col['default']:
                        col_def += f" DEFAULT {col['default']}"
                    column_defs.append(col_def)

                # Primary key
                if structure['primary_keys']:
                    pk_def = f"    PRIMARY KEY ({', '.join(structure['primary_keys'])})"
                    column_defs.append(pk_def)

                # Foreign keys
                for fk in structure['foreign_keys']:
                    fk_def = f"    FOREIGN KEY ({fk['column']}) REFERENCES {fk['referenced_table']}({fk['referenced_column']})"
                    column_defs.append(fk_def)

                f.write(',\n'.join(column_defs))
                f.write("\n);\n\n")

                # Indexes
                for idx in structure['indexes']:
                    if idx['name'].startswith('sqlite_'):  # Skip system indexes
                        continue
                    unique_keyword = "UNIQUE " if idx.get('unique') else ""
                    columns_str = ', '.join(idx.get('columns', []))
                    f.write(f"CREATE {unique_keyword}INDEX {idx['name']} ON {table_name} ({columns_str});\n")

                # Triggers
                for trigger in structure['triggers']:
                    f.write(f"\n-- Trigger: {trigger['name']}\n")
                    f.write(f"{trigger.get('sql', '')};\n")

            print(f"✅ Table structure exported to: {output_file}")

        except Exception as e:
            print(f"❌ Error exporting table structure: {e}")

    def export_database_schema(self):
        """Export complete database schema"""
        try:
            output_file = f"{os.path.splitext(os.path.basename(self.db_file))[0]}_schema.sql"

            with open(output_file, 'w') as f:
                f.write(f"-- Complete database schema export\n")
                f.write(f"-- Source: {self.db_file}\n")
                f.write(f"-- Generated by Universal Database Explorer\n\n")

                db_info = self.explorer.get_database_info()
                tables = db_info.get('tables', [])

                # Export each table
                for table_info in tables:
                    table_name = table_info['name']
                    structure = self.explorer.get_table_structure(table_name)

                    if structure:
                        f.write(f"-- Table: {table_name}\n")
                        f.write(f"CREATE TABLE {table_name} (\n")

                        # Columns
                        column_defs = []
                        for col in structure['columns']:
                            col_def = f"    {col['name']} {col['type']}"
                            if not col['nullable']:
                                col_def += " NOT NULL"
                            if col['default']:
                                col_def += f" DEFAULT {col['default']}"
                            column_defs.append(col_def)

                        # Primary key
                        if structure['primary_keys']:
                            pk_def = f"    PRIMARY KEY ({', '.join(structure['primary_keys'])})"
                            column_defs.append(pk_def)

                        f.write(',\n'.join(column_defs))
                        f.write("\n);\n\n")

                # Export views
                if hasattr(self.explorer, 'schema') and self.explorer.schema.views:
                    f.write("-- Views\n")
                    for view in self.explorer.schema.views:
                        # Get view definition
                        view_def = self.explorer.execute_query(f"SELECT sql FROM sqlite_master WHERE type='view' AND name='{view}'")
                        if view_def and view_def[0].get('sql'):
                            f.write(f"{view_def[0]['sql']};\n\n")

                # Export triggers
                if hasattr(self.explorer, 'schema') and self.explorer.schema.triggers:
                    f.write("-- Triggers\n")
                    for trigger in self.explorer.schema.triggers:
                        f.write(f"{trigger.get('sql', '')};\n\n")

            print(f"✅ Complete database schema exported to: {output_file}")

        except Exception as e:
            print(f"❌ Error exporting database schema: {e}")

    def export_table_csv(self):
        """Export table data to CSV format"""
        table_name = input("Enter table name to export as CSV: ").strip()

        if not table_name:
            print("❌ Table name cannot be empty.")
            return

        try:
            data = self.explorer.get_sample_data(table_name, limit=1000000)  # Large limit

            if not data:
                print(f"❌ No data found in table '{table_name}'.")
                return

            output_file = f"{table_name}_data.csv"

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if data:
                    # Write header
                    headers = list(data[0].keys())
                    f.write(','.join(headers) + '\n')

                    # Write data rows
                    for row in data:
                        values = []
                        for value in row.values():
                            if value is None:
                                values.append('')
                            elif isinstance(value, str):
                                # Escape commas and quotes
                                if ',' in value or '"' in value or '\n' in value:
                                    escaped_value = value.replace('"', '""')
                                    values.append(f'"{escaped_value}"')
                                else:
                                    values.append(value)
                            else:
                                values.append(str(value))

                        f.write(','.join(values) + '\n')

            print(f"✅ Table data exported to CSV: {output_file}")
            print(f"📊 Exported {len(data)} rows")

        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")

    def data_validation(self):
        """Perform data validation checks"""
        print("\n🧪 DATA VALIDATION")
        print("1. Check referential integrity")
        print("2. Find duplicate records")
        print("3. Validate data types")
        print("4. Check for NULL values")
        print("5. Run all validation checks")

        choice = input("Choose validation option (1-5): ").strip()

        if choice == "1":
            self.check_referential_integrity()
        elif choice == "2":
            self.find_duplicates()
        elif choice == "3":
            self.validate_data_types()
        elif choice == "4":
            self.check_null_values()
        elif choice == "5":
            self.run_all_validations()

    def check_referential_integrity(self):
        """Check referential integrity"""
        print("\n🔗 CHECKING REFERENTIAL INTEGRITY")
        print("-" * 40)

        issues_found = 0

        db_info = self.explorer.get_database_info()
        tables = db_info.get('tables', [])

        for table_info in tables:
            table_name = table_info['name']
            structure = self.explorer.get_table_structure(table_name)

            if structure and structure['foreign_keys']:
                for fk in structure['foreign_keys']:
                    try:
                        # Check for orphaned records
                        query = f"""
                            SELECT COUNT(*) as orphaned_count
                            FROM {table_name} t1
                            LEFT JOIN {fk['referenced_table']} t2 
                            ON t1.{fk['column']} = t2.{fk['referenced_column']}
                            WHERE t1.{fk['column']} IS NOT NULL 
                            AND t2.{fk['referenced_column']} IS NULL
                        """

                        result = self.explorer.execute_query(query)
                        if result and result[0].get('orphaned_count', 0) > 0:
                            issues_found += 1
                            print(f"❌ {table_name}.{fk['column']} → {fk['referenced_table']}.{fk['referenced_column']}")
                            print(f"   Orphaned records: {result[0]['orphaned_count']}")

                    except Exception as e:
                        print(f"⚠️  Could not check FK {table_name}.{fk['column']}: {e}")

        if issues_found == 0:
            print("✅ No referential integrity issues found!")
        else:
            print(f"\n⚠️  Found {issues_found} referential integrity issues.")

    def find_duplicates(self):
        """Find duplicate records in tables"""
        table_name = input("Enter table name to check for duplicates: ").strip()

        if not table_name:
            print("❌ Table name cannot be empty.")
            return

        columns_input = input("Enter columns to check (comma-separated, or press Enter for all): ").strip()

        structure = self.explorer.get_table_structure(table_name)
        if not structure:
            print(f"❌ Table '{table_name}' not found.")
            return

        if columns_input:
            columns = [col.strip() for col in columns_input.split(',')]
        else:
            columns = [col['name'] for col in structure['columns']]

        try:
            columns_str = ", ".join(columns)
            query = f"""
                SELECT {columns_str}, COUNT(*) as duplicate_count
                FROM {table_name}
                GROUP BY {columns_str}
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
            """

            duplicates = self.explorer.execute_query(query)

            if duplicates:
                print(f"\n🔍 DUPLICATE RECORDS FOUND in {table_name}:")
                print("-" * 50)

                # Display header
                headers = list(duplicates[0].keys())
                print(" | ".join(f"{h:<15}" for h in headers))
                print("-" * (len(headers) * 18))

                # Display duplicates
                for row in duplicates:
                    values = [str(row.get(h, ''))[:15] for h in headers]
                    print(" | ".join(f"{v:<15}" for v in values))

                print(f"\n📊 Found {len(duplicates)} sets of duplicate records")
            else:
                print(f"✅ No duplicate records found in {table_name}")

        except Exception as e:
            print(f"❌ Error checking for duplicates: {e}")

    def validate_data_types(self):
        """Validate data types in tables"""
        table_name = input("Enter table name to validate data types: ").strip()

        if not table_name:
            print("❌ Table name cannot be empty.")
            return

        structure = self.explorer.get_table_structure(table_name)
        if not structure:
            print(f"❌ Table '{table_name}' not found.")
            return

        print(f"\n🧪 VALIDATING DATA TYPES in {table_name}")
        print("-" * 50)

        issues_found = 0

        for column in structure['columns']:
            col_name = column['name']
            col_type = column['type'].upper()

            try:
                # Check for specific data type issues
                if 'INT' in col_type:
                    # Check for non-numeric values in integer columns
                    query = f"""
                        SELECT COUNT(*) as invalid_count 
                        FROM {table_name} 
                        WHERE {col_name} IS NOT NULL 
                        AND CAST({col_name} AS TEXT) GLOB '*[!0-9-]*'
                    """

                    result = self.explorer.execute_query(query)
                    if result and result[0].get('invalid_count', 0) > 0:
                        issues_found += 1
                        print(f"❌ {col_name} ({col_type}): {result[0]['invalid_count']} invalid integer values")

                elif 'REAL' in col_type or 'NUMERIC' in col_type:
                    # Check for non-numeric values in numeric columns
                    query = f"""
                        SELECT COUNT(*) as invalid_count 
                        FROM {table_name} 
                        WHERE {col_name} IS NOT NULL 
                        AND CAST({col_name} AS TEXT) GLOB '*[!0-9.-]*'
                    """

                    result = self.explorer.execute_query(query)
                    if result and result[0].get('invalid_count', 0) > 0:
                        issues_found += 1
                        print(f"❌ {col_name} ({col_type}): {result[0]['invalid_count']} invalid numeric values")

                # Check for extremely long text values
                elif 'TEXT' in col_type or 'VARCHAR' in col_type:
                    query = f"""
                        SELECT COUNT(*) as long_text_count 
                        FROM {table_name} 
                        WHERE LENGTH({col_name}) > 1000
                    """

                    result = self.explorer.execute_query(query)
                    if result and result[0].get('long_text_count', 0) > 0:
                        print(f"⚠️  {col_name}: {result[0]['long_text_count']} very long text values (>1000 chars)")

            except Exception as e:
                print(f"⚠️  Could not validate {col_name}: {e}")

        if issues_found == 0:
            print("✅ No data type validation issues found!")
        else:
            print(f"\n⚠️  Found {issues_found} data type validation issues.")

    def check_null_values(self):
        """Check for NULL values in columns"""
        table_name = input("Enter table name to check NULL values: ").strip()

        if not table_name:
            print("❌ Table name cannot be empty.")
            return

        structure = self.explorer.get_table_structure(table_name)
        if not structure:
            print(f"❌ Table '{table_name}' not found.")
            return

        print(f"\n🔍 NULL VALUE ANALYSIS for {table_name}")
        print("-" * 50)

        total_rows = structure['row_count']

        for column in structure['columns']:
            col_name = column['name']

            try:
                query = f"SELECT COUNT(*) as null_count FROM {table_name} WHERE {col_name} IS NULL"
                result = self.explorer.execute_query(query)

                if result:
                    null_count = result[0].get('null_count', 0)
                    null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

                    status = "✅" if null_count == 0 else "⚠️" if null_percentage < 10 else "❌"
                    nullable_text = "NULL" if column['nullable'] else "NOT NULL"

                    print(f"{status} {col_name:<20} {nullable_text:<10} {null_count:>6} ({null_percentage:>5.1f}%)")

            except Exception as e:
                print(f"⚠️  Could not check {col_name}: {e}")

    def run_all_validations(self):
        """Run all validation checks"""
        print("\n🧪 RUNNING ALL VALIDATION CHECKS")
        print("=" * 50)

        print("\n1. Checking referential integrity...")
        self.check_referential_integrity()

        print("\n" + "-" * 50)
        print("2. Checking for common data issues...")

        # Check each table for basic issues
        db_info = self.explorer.get_database_info()
        tables = db_info.get('tables', [])

        for table_info in tables:
            table_name = table_info['name']
            print(f"\n   Checking table: {table_name}")

            try:
                # Check for empty tables
                if table_info['rows'] == 0:
                    print(f"   ⚠️  Empty table: {table_name}")

                # Check for tables with too many NULL values
                structure = self.explorer.get_table_structure(table_name)
                if structure:
                    for column in structure['columns']:
                        if not column['nullable']:  # NOT NULL columns
                            null_check = self.explorer.execute_query(
                                f"SELECT COUNT(*) as null_count FROM {table_name} WHERE {column['name']} IS NULL"
                            )
                            if null_check and null_check[0].get('null_count', 0) > 0:
                                print(f"   ❌ NOT NULL violation in {table_name}.{column['name']}")

            except Exception as e:
                print(f"   ⚠️  Error checking {table_name}: {e}")

        print("\n" + "-" * 50)
        print("3. Performance recommendations...")

        # Basic performance recommendations
        for table_info in tables:
            table_name = table_info['name']

            # Large tables without indexes
            if table_info['rows'] > 1000 and table_info['indexes'] == 0:
                print(f"   💡 Consider adding indexes to large table: {table_name}")

            # Tables with many foreign keys but few indexes
            if table_info['foreign_keys'] > 1 and table_info['indexes'] < table_info['foreign_keys']:
                print(f"   💡 Consider indexing foreign keys in: {table_name}")

        print("\n✅ All validation checks completed!")

    def run_interactive_mode(self):
        """Main interactive mode loop"""
        print("\n🚀 UNIVERSAL DATABASE EXPLORER")
        print("=" * 60)

        # Connect to database
        if not self.connect_to_database():
            return

        # Show initial overview
        self.display_database_overview()

        # Main menu loop
        while True:
            try:
                self.display_menu()
                choice = input("\n🔧 Enter your choice (0-13): ").strip()

                if not self.handle_menu_choice(choice):
                    break

                # Wait for user before showing menu again
                input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Please try again or exit (0).")

def main():
    """Main function to handle command line arguments and start the CLI"""

    # Check command line arguments
    if len(sys.argv) != 2:
        print("🔧 Universal Database Explorer")
        print("=" * 40)
        print("Usage: python db_explorer.py <database_file>")
        print("\nExamples:")
        print("  python db_explorer.py student_database.db")
        print("  python db_explorer.py sales_data.sqlite")
        print("  python db_explorer.py /path/to/company.db")
        print("\nSupported formats:")
        print("  • SQLite databases (.db, .sqlite, .sqlite3)")
        print("  • Any SQLite-compatible database file")
        sys.exit(1)

    db_file = sys.argv[1]

    # Create and run CLI
    cli = DatabaseCLI(db_file)
    cli.run_interactive_mode()

if __name__ == "__main__":
    main()