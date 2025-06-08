#!/usr/bin/env python3
"""
Oracle HR Sample Database Creator
Reads schema and data directly from provided text files
"""

import sqlite3
import os
import re

def read_file_content(filename):
    """Read content from a file"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found!")
        return None
    except Exception as e:
        print(f"❌ Error reading '{filename}': {e}")
        return None

def adapt_schema_for_sqlite(schema_content):
    """Convert MySQL schema to SQLite compatible format"""
    if not schema_content:
        return []

    # Split into individual CREATE TABLE statements
    statements = []
    current_statement = ""

    for line in schema_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('--') or line.startswith('/*'):
            continue

        if line.upper().startswith('CREATE TABLE'):
            if current_statement:
                statements.append(current_statement)
            current_statement = line
        elif current_statement:
            current_statement += " " + line

        if line.endswith(';'):
            if current_statement:
                statements.append(current_statement)
                current_statement = ""

    # Convert MySQL syntax to SQLite
    sqlite_statements = []
    for stmt in statements:
        # Remove MySQL specific syntax
        stmt = re.sub(r'AUTO_INCREMENT', '', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'INT\s*\(\d+\)', 'INTEGER', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'VARCHAR\s*\(\d+\)', 'TEXT', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'CHAR\s*\(\d+\)', 'TEXT', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'DECIMAL\s*\(\d+,\s*\d+\)', 'REAL', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'ON DELETE CASCADE ON UPDATE CASCADE', '', stmt, flags=re.IGNORECASE)

        sqlite_statements.append(stmt)

    return sqlite_statements

def extract_insert_statements(data_content):
    """Extract INSERT statements from data file"""
    if not data_content:
        return []

    insert_statements = []

    for line in data_content.split('\n'):
        line = line.strip()
        if line.upper().startswith('INSERT INTO') and line.endswith(';'):
            # Clean up the statement
            line = line.replace('INSERT INTO', 'INSERT INTO')
            insert_statements.append(line)

    return insert_statements

def create_oracle_hr_database():
    """Create the Oracle HR database from text files"""
    print("🏛️  Creating hr_database.db from text files...")

    # Check if files exist
    schema_file = 'hr_schema.txt'
    data_file = 'hr_data.txt'

    if not os.path.exists(schema_file):
        print(f"❌ Schema file '{schema_file}' not found!")
        print("Make sure you have 'hr_schema.txt' in the current directory.")
        return False

    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        print("Make sure you have 'hr_data.txt' in the current directory.")
        return False

    # Read files
    print("📖 Reading schema file...")
    schema_content = read_file_content(schema_file)

    print("📖 Reading data file...")
    data_content = read_file_content(data_file)

    if not schema_content or not data_content:
        return False

    # Create database
    conn = sqlite3.connect('hr_database.db')
    cursor = conn.cursor()

    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")

    try:
        # Create tables from schema
        print("🏗️  Creating tables...")
        schema_statements = adapt_schema_for_sqlite(schema_content)

        for i, statement in enumerate(schema_statements, 1):
            try:
                print(f"   Creating table {i}/{len(schema_statements)}...")
                cursor.execute(statement)
            except Exception as e:
                print(f"   ⚠️  Warning: {e}")
                print(f"   Statement: {statement[:100]}...")
                continue

        # Insert data
        print("📊 Inserting data...")
        insert_statements = extract_insert_statements(data_content)

        successful_inserts = 0
        for i, statement in enumerate(insert_statements, 1):
            try:
                cursor.execute(statement)
                successful_inserts += 1
                if i % 10 == 0 or i == len(insert_statements):
                    print(f"   Processed {i}/{len(insert_statements)} insert statements...")
            except Exception as e:
                print(f"   ⚠️  Insert error: {e}")
                print(f"   Statement: {statement[:100]}...")
                continue

        print(f"✅ Successfully inserted {successful_inserts}/{len(insert_statements)} records")

        # Create useful indexes
        print("🔍 Creating indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_employees_department_id ON employees(department_id)",
            "CREATE INDEX IF NOT EXISTS idx_employees_manager_id ON employees(manager_id)",
            "CREATE INDEX IF NOT EXISTS idx_employees_job_id ON employees(job_id)",
            "CREATE INDEX IF NOT EXISTS idx_employees_email ON employees(email)",
            "CREATE INDEX IF NOT EXISTS idx_dependents_employee_id ON dependents(employee_id)",
            "CREATE INDEX IF NOT EXISTS idx_departments_location_id ON departments(location_id)",
            "CREATE INDEX IF NOT EXISTS idx_locations_country_id ON locations(country_id)",
            "CREATE INDEX IF NOT EXISTS idx_countries_region_id ON countries(region_id)"
        ]

        for idx in indexes:
            try:
                cursor.execute(idx)
            except Exception as e:
                print(f"   ⚠️  Index warning: {e}")

        # Create useful views
        print("👁️  Creating views...")

        # Employee details view
        try:
            cursor.execute('''
                           CREATE VIEW IF NOT EXISTS employee_details AS
                           SELECT
                               e.employee_id,
                               e.first_name,
                               e.last_name,
                               e.first_name || ' ' || e.last_name AS full_name,
                               e.email,
                               e.phone_number,
                               e.hire_date,
                               e.salary,
                               j.job_title,
                               d.department_name,
                               m.first_name || ' ' || m.last_name AS manager_name,
                               l.city,
                               l.state_province,
                               c.country_name,
                               r.region_name
                           FROM employees e
                                    LEFT JOIN jobs j ON e.job_id = j.job_id
                                    LEFT JOIN departments d ON e.department_id = d.department_id
                                    LEFT JOIN employees m ON e.manager_id = m.employee_id
                                    LEFT JOIN locations l ON d.location_id = l.location_id
                                    LEFT JOIN countries c ON l.country_id = c.country_id
                                    LEFT JOIN regions r ON c.region_id = r.region_id
                           ''')
        except Exception as e:
            print(f"   ⚠️  View warning: {e}")

        # Department summary view
        try:
            cursor.execute('''
                           CREATE VIEW IF NOT EXISTS department_summary AS
                           SELECT
                               d.department_id,
                               d.department_name,
                               COUNT(e.employee_id) as employee_count,
                               AVG(e.salary) as avg_salary,
                               MIN(e.salary) as min_salary,
                               MAX(e.salary) as max_salary,
                               l.city,
                               c.country_name
                           FROM departments d
                                    LEFT JOIN employees e ON d.department_id = e.department_id
                                    LEFT JOIN locations l ON d.location_id = l.location_id
                                    LEFT JOIN countries c ON l.country_id = c.country_id
                           GROUP BY d.department_id, d.department_name, l.city, c.country_name
                           ''')
        except Exception as e:
            print(f"   ⚠️  View warning: {e}")

        # Salary analysis view
        try:
            cursor.execute('''
                           CREATE VIEW IF NOT EXISTS salary_analysis AS
                           SELECT
                               j.job_title,
                               COUNT(e.employee_id) as employee_count,
                               AVG(e.salary) as avg_salary,
                               MIN(e.salary) as min_salary,
                               MAX(e.salary) as max_salary,
                               j.min_salary as job_min_salary,
                               j.max_salary as job_max_salary
                           FROM jobs j
                                    LEFT JOIN employees e ON j.job_id = e.job_id
                           GROUP BY j.job_id, j.job_title, j.min_salary, j.max_salary
                           ORDER BY avg_salary DESC
                           ''')
        except Exception as e:
            print(f"   ⚠️  View warning: {e}")

        # Commit all changes
        conn.commit()

        # Get database statistics
        print("📈 Database statistics:")

        tables = ['regions', 'countries', 'locations', 'jobs', 'departments', 'employees', 'dependents']
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count} records")
            except Exception as e:
                print(f"   {table}: Error getting count - {e}")

        # Get file size
        conn.close()

        if os.path.exists('hr_database.db'):
            file_size = os.path.getsize('hr_database.db')
            size_mb = file_size / (1024 * 1024)
            print(f"   Database file size: {size_mb:.2f} MB")

        print("✅ hr_database.db created successfully!")
        print("\n🎯 This is the famous Oracle HR sample database used worldwide for:")
        print("   • Learning SQL and database concepts")
        print("   • Testing database applications")
        print("   • Training and education")
        print("   • Demonstration purposes")

        print(f"\n🛠️  Usage:")
        print(f"   python db_explorer.py hr_database.db")

        return True

    except Exception as e:
        print(f"❌ Error creating database: {e}")
        conn.rollback()
        conn.close()
        return False

def main():
    """Main function"""
    print("🏛️  Oracle HR Database Creator")
    print("=" * 50)

    if create_oracle_hr_database():
        print("\n🎉 Success! The famous Oracle HR database is ready to explore!")
    else:
        print("\n💥 Failed to create database. Please check the error messages above.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())