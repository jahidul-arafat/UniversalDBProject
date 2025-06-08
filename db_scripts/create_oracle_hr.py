#!/usr/bin/env python3
"""
Enhanced Oracle HR Sample Database Creator
Creates a comprehensive database with all foreign keys, triggers, views, indexes, and constraints
"""

import sqlite3
import os
import re
from datetime import datetime

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
    """Convert MySQL schema to SQLite compatible format with enhanced features"""
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

    # Convert MySQL syntax to SQLite with enhanced constraints
    sqlite_statements = []
    for stmt in statements:
        # Remove MySQL specific syntax but preserve constraints
        stmt = re.sub(r'AUTO_INCREMENT', '', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'INT\s*\(\d+\)', 'INTEGER', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'VARCHAR\s*\(\d+\)', 'TEXT', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'CHAR\s*\(\d+\)', 'TEXT', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'DECIMAL\s*\(\d+,\s*\d+\)', 'REAL', stmt, flags=re.IGNORECASE)

        # Keep foreign key constraints
        # stmt = re.sub(r'ON DELETE CASCADE ON UPDATE CASCADE', '', stmt, flags=re.IGNORECASE)

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

def create_additional_constraints(cursor):
    """Create additional constraints and check constraints"""
    print("🔒 Creating additional constraints...")

    constraints = [
        # Email format constraint
        '''CREATE TRIGGER check_email_format
            BEFORE INSERT ON employees
            FOR EACH ROW
            WHEN NEW.email NOT LIKE '%@%'
        BEGIN
            SELECT RAISE(ABORT, 'Invalid email format');
        END''',

        # Salary range constraints
        '''CREATE TRIGGER check_salary_range
            BEFORE INSERT ON employees
            FOR EACH ROW
            WHEN NEW.salary <= 0
        BEGIN
            SELECT RAISE(ABORT, 'Salary must be positive');
        END''',

        # Hire date constraint
        '''CREATE TRIGGER check_hire_date
            BEFORE INSERT ON employees
            FOR EACH ROW
            WHEN NEW.hire_date > date('now')
        BEGIN
        SELECT RAISE(ABORT, 'Hire date cannot be in the future');
        END''',

        # Manager hierarchy constraint (prevent self-reference)
        '''CREATE TRIGGER check_manager_hierarchy
            BEFORE INSERT ON employees
            FOR EACH ROW
            WHEN NEW.manager_id = NEW.employee_id
        BEGIN
            SELECT RAISE(ABORT, 'Employee cannot be their own manager');
        END''',

        # Department location constraint
        '''CREATE TRIGGER check_department_location
            BEFORE INSERT ON departments
            FOR EACH ROW
            WHEN NEW.location_id IS NOT NULL AND
                 NOT EXISTS (SELECT 1 FROM locations WHERE location_id = NEW.location_id)
        BEGIN
            SELECT RAISE(ABORT, 'Invalid location_id for department');
        END''',
    ]

    for constraint in constraints:
        try:
            cursor.execute(constraint)
        except Exception as e:
            print(f"   ⚠️  Constraint warning: {e}")

def create_audit_triggers(cursor):
    """Create audit triggers for tracking changes"""
    print("📝 Creating audit triggers...")

    # Create audit table
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS audit_log (
                                                            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                            table_name TEXT NOT NULL,
                                                            operation TEXT NOT NULL,
                                                            old_values TEXT,
                                                            new_values TEXT,
                                                            changed_by TEXT DEFAULT 'system',
                                                            changed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                   )
                   ''')

    # Employee audit triggers
    audit_triggers = [
        '''CREATE TRIGGER employee_audit_insert
            AFTER INSERT ON employees
            FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (table_name, operation, new_values)
            VALUES ('employees', 'INSERT',
                    'employee_id:' || NEW.employee_id ||
                    ',name:' || NEW.first_name || ' ' || NEW.last_name ||
                    ',salary:' || NEW.salary);
        END''',

        '''CREATE TRIGGER employee_audit_update
            AFTER UPDATE ON employees
            FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (table_name, operation, old_values, new_values)
            VALUES ('employees', 'UPDATE',
                    'salary:' || OLD.salary || ',department:' || OLD.department_id,
                    'salary:' || NEW.salary || ',department:' || NEW.department_id);
        END''',

        '''CREATE TRIGGER employee_audit_delete
            AFTER DELETE ON employees
            FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (table_name, operation, old_values)
            VALUES ('employees', 'DELETE',
                    'employee_id:' || OLD.employee_id ||
                    ',name:' || OLD.first_name || ' ' || OLD.last_name);
        END''',

        # Department audit triggers
        '''CREATE TRIGGER department_audit_insert
            AFTER INSERT ON departments
            FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (table_name, operation, new_values)
            VALUES ('departments', 'INSERT',
                    'department_id:' || NEW.department_id ||
                    ',name:' || NEW.department_name);
        END''',

        '''CREATE TRIGGER department_audit_update
            AFTER UPDATE ON departments
            FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (table_name, operation, old_values, new_values)
            VALUES ('departments', 'UPDATE',
                    'name:' || OLD.department_name || ',location:' || OLD.location_id,
                    'name:' || NEW.department_name || ',location:' || NEW.location_id);
        END''',
    ]

    for trigger in audit_triggers:
        try:
            cursor.execute(trigger)
        except Exception as e:
            print(f"   ⚠️  Audit trigger warning: {e}")

def create_business_triggers(cursor):
    """Create business logic triggers"""
    print("💼 Creating business logic triggers...")

    business_triggers = [
        # Auto-update employee count when employees are added/removed
        '''CREATE TRIGGER update_dept_employee_count_insert
            AFTER INSERT ON employees
            FOR EACH ROW
            WHEN NEW.department_id IS NOT NULL
        BEGIN
            UPDATE departments
            SET employee_count = (
                SELECT COUNT(*)
                FROM employees
                WHERE department_id = NEW.department_id
            )
            WHERE department_id = NEW.department_id;
        END''',

        '''CREATE TRIGGER update_dept_employee_count_delete
            AFTER DELETE ON employees
            FOR EACH ROW
            WHEN OLD.department_id IS NOT NULL
        BEGIN
            UPDATE departments
            SET employee_count = (
                SELECT COUNT(*)
                FROM employees
                WHERE department_id = OLD.department_id
            )
            WHERE department_id = OLD.department_id;
        END''',

        # Validate salary against job salary range
        '''CREATE TRIGGER validate_employee_salary
            BEFORE INSERT ON employees
            FOR EACH ROW
        BEGIN
            SELECT CASE
                       WHEN NEW.salary < (SELECT min_salary FROM jobs WHERE job_id = NEW.job_id) OR
                            NEW.salary > (SELECT max_salary FROM jobs WHERE job_id = NEW.job_id)
                           THEN RAISE(ABORT, 'Salary outside job range')
                       END;
        END''',

        # Auto-generate employee email if not provided
        '''CREATE TRIGGER generate_employee_email
            BEFORE INSERT ON employees
            FOR EACH ROW
            WHEN NEW.email IS NULL OR NEW.email = ''
        BEGIN
            UPDATE employees SET email =
                                     lower(NEW.first_name || '.' || NEW.last_name || '@company.com')
            WHERE employee_id = NEW.employee_id;
        END''',
    ]

    # First add employee_count column to departments if it doesn't exist
    try:
        cursor.execute('ALTER TABLE departments ADD COLUMN employee_count INTEGER DEFAULT 0')
    except:
        pass  # Column might already exist

    for trigger in business_triggers:
        try:
            cursor.execute(trigger)
        except Exception as e:
            print(f"   ⚠️  Business trigger warning: {e}")

def create_comprehensive_indexes(cursor):
    """Create comprehensive indexes for performance"""
    print("🔍 Creating comprehensive indexes...")

    indexes = [
        # Primary relationship indexes
        "CREATE INDEX IF NOT EXISTS idx_employees_department_id ON employees(department_id)",
        "CREATE INDEX IF NOT EXISTS idx_employees_manager_id ON employees(manager_id)",
        "CREATE INDEX IF NOT EXISTS idx_employees_job_id ON employees(job_id)",
        "CREATE INDEX IF NOT EXISTS idx_dependents_employee_id ON dependents(employee_id)",
        "CREATE INDEX IF NOT EXISTS idx_departments_location_id ON departments(location_id)",
        "CREATE INDEX IF NOT EXISTS idx_locations_country_id ON locations(country_id)",
        "CREATE INDEX IF NOT EXISTS idx_countries_region_id ON countries(region_id)",

        # Performance indexes for common queries
        "CREATE INDEX IF NOT EXISTS idx_employees_email ON employees(email)",
        "CREATE INDEX IF NOT EXISTS idx_employees_last_name ON employees(last_name)",
        "CREATE INDEX IF NOT EXISTS idx_employees_hire_date ON employees(hire_date)",
        "CREATE INDEX IF NOT EXISTS idx_employees_salary ON employees(salary)",
        "CREATE INDEX IF NOT EXISTS idx_employees_full_name ON employees(first_name, last_name)",

        # Composite indexes for common query patterns
        "CREATE INDEX IF NOT EXISTS idx_emp_dept_job ON employees(department_id, job_id)",
        "CREATE INDEX IF NOT EXISTS idx_emp_manager_dept ON employees(manager_id, department_id)",
        "CREATE INDEX IF NOT EXISTS idx_emp_hire_salary ON employees(hire_date, salary)",

        # Location and geography indexes
        "CREATE INDEX IF NOT EXISTS idx_locations_city ON locations(city)",
        "CREATE INDEX IF NOT EXISTS idx_locations_country_city ON locations(country_id, city)",
        "CREATE INDEX IF NOT EXISTS idx_countries_region_name ON countries(region_id, country_name)",

        # Job and salary analysis indexes
        "CREATE INDEX IF NOT EXISTS idx_jobs_salary_range ON jobs(min_salary, max_salary)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(job_title)",

        # Department analysis indexes
        "CREATE INDEX IF NOT EXISTS idx_departments_name ON departments(department_name)",
        "CREATE INDEX IF NOT EXISTS idx_dept_location_name ON departments(location_id, department_name)",

        # Dependent analysis indexes
        "CREATE INDEX IF NOT EXISTS idx_dependents_relationship ON dependents(relationship)",
        "CREATE INDEX IF NOT EXISTS idx_dependents_name ON dependents(first_name, last_name)",

        # Audit table indexes
        "CREATE INDEX IF NOT EXISTS idx_audit_table_operation ON audit_log(table_name, operation)",
        "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(changed_at)",

        # Unique constraints as indexes
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_employees_email_unique ON employees(email)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_regions_name_unique ON regions(region_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_countries_name_unique ON countries(country_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_departments_name_unique ON departments(department_name)",
    ]

    successful_indexes = 0
    for idx in indexes:
        try:
            cursor.execute(idx)
            successful_indexes += 1
        except Exception as e:
            print(f"   ⚠️  Index warning: {e}")

    print(f"   ✅ Created {successful_indexes}/{len(indexes)} indexes")

def create_enhanced_views(cursor):
    """Create comprehensive views for analysis"""
    print("👁️  Creating enhanced views...")

    views = [
        # Enhanced employee details view
        ('''employee_details''', '''
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
                                     j.min_salary as job_min_salary,
                                     j.max_salary as job_max_salary,
                                     CASE
                                         WHEN e.salary < j.min_salary THEN 'Below Range'
                                         WHEN e.salary > j.max_salary THEN 'Above Range'
                                         ELSE 'Within Range'
                                         END as salary_status,
                                     d.department_name,
                                     m.first_name || ' ' || m.last_name AS manager_name,
                                     l.city,
                                     l.state_province,
                                     l.street_address,
                                     c.country_name,
                                     r.region_name,
                                     (julianday('now') - julianday(e.hire_date)) as days_employed,
                                     CASE
                                         WHEN (julianday('now') - julianday(e.hire_date)) > 3650 THEN 'Senior'
                                         WHEN (julianday('now') - julianday(e.hire_date)) > 1825 THEN 'Experienced'
                                         ELSE 'Junior'
                                         END as seniority_level
                                 FROM employees e
                                          LEFT JOIN jobs j ON e.job_id = j.job_id
                                          LEFT JOIN departments d ON e.department_id = d.department_id
                                          LEFT JOIN employees m ON e.manager_id = m.employee_id
                                          LEFT JOIN locations l ON d.location_id = l.location_id
                                          LEFT JOIN countries c ON l.country_id = c.country_id
                                          LEFT JOIN regions r ON c.region_id = r.region_id
                                 '''),

        # Department analytics view
        ('''department_analytics''', '''
                                     CREATE VIEW IF NOT EXISTS department_analytics AS
                                     SELECT
                                         d.department_id,
                                         d.department_name,
                                         COUNT(e.employee_id) as employee_count,
                                         AVG(e.salary) as avg_salary,
                                         MIN(e.salary) as min_salary,
                                         MAX(e.salary) as max_salary,
                                         SUM(e.salary) as total_payroll,
                                         COUNT(DISTINCT e.job_id) as unique_jobs,
                                         COUNT(CASE WHEN e.manager_id IS NULL THEN 1 END) as managers_count,
                                         l.city,
                                         l.state_province,
                                         c.country_name,
                                         r.region_name,
                                         CASE
                                             WHEN COUNT(e.employee_id) > 10 THEN 'Large'
                                             WHEN COUNT(e.employee_id) > 5 THEN 'Medium'
                                             ELSE 'Small'
                                             END as department_size
                                     FROM departments d
                                              LEFT JOIN employees e ON d.department_id = e.department_id
                                              LEFT JOIN locations l ON d.location_id = l.location_id
                                              LEFT JOIN countries c ON l.country_id = c.country_id
                                              LEFT JOIN regions r ON c.region_id = r.region_id
                                     GROUP BY d.department_id, d.department_name, l.city, l.state_province, c.country_name, r.region_name
                                     '''),

        # Salary analysis view
        ('''salary_analysis''', '''
                                CREATE VIEW IF NOT EXISTS salary_analysis AS
                                SELECT
                                    j.job_title,
                                    COUNT(e.employee_id) as employee_count,
                                    AVG(e.salary) as avg_salary,
                                    MIN(e.salary) as min_actual_salary,
                                    MAX(e.salary) as max_actual_salary,
                                    j.min_salary as job_min_salary,
                                    j.max_salary as job_max_salary,
                                    ROUND((AVG(e.salary) - j.min_salary) * 100.0 / (j.max_salary - j.min_salary), 2) as salary_percentile,
                                    COUNT(CASE WHEN e.salary < j.min_salary THEN 1 END) as below_range_count,
                                    COUNT(CASE WHEN e.salary > j.max_salary THEN 1 END) as above_range_count
                                FROM jobs j
                                         LEFT JOIN employees e ON j.job_id = e.job_id
                                GROUP BY j.job_id, j.job_title, j.min_salary, j.max_salary
                                ORDER BY avg_salary DESC
                                '''),

        # Management hierarchy view
        ('''management_hierarchy''', '''
                                     CREATE VIEW IF NOT EXISTS management_hierarchy AS
                                     SELECT
                                         e.employee_id,
                                         e.first_name || ' ' || e.last_name as employee_name,
                                         e.job_title as employee_job,
                                         m.employee_id as manager_id,
                                         m.first_name || ' ' || m.last_name as manager_name,
                                         m.job_title as manager_job,
                                         COUNT(s.employee_id) as direct_reports,
                                         e.department_name,
                                         e.salary as employee_salary,
                                         m.salary as manager_salary
                                     FROM employee_details e
                                              LEFT JOIN employee_details m ON e.manager_id = m.employee_id
                                              LEFT JOIN employee_details s ON e.employee_id = s.manager_id
                                     GROUP BY e.employee_id, e.employee_name, e.job_title, m.employee_id, m.employee_name, m.job_title
                                     '''),

        # Geographic distribution view
        ('''geographic_distribution''', '''
                                        CREATE VIEW IF NOT EXISTS geographic_distribution AS
                                        SELECT
                                            r.region_name,
                                            c.country_name,
                                            l.city,
                                            l.state_province,
                                            COUNT(DISTINCT d.department_id) as departments_count,
                                            COUNT(e.employee_id) as employees_count,
                                            AVG(e.salary) as avg_salary,
                                            GROUP_CONCAT(DISTINCT d.department_name) as departments
                                        FROM regions r
                                                 LEFT JOIN countries c ON r.region_id = c.region_id
                                                 LEFT JOIN locations l ON c.country_id = l.country_id
                                                 LEFT JOIN departments d ON l.location_id = d.location_id
                                                 LEFT JOIN employees e ON d.department_id = e.department_id
                                        GROUP BY r.region_name, c.country_name, l.city, l.state_province
                                        ORDER BY employees_count DESC
                                        '''),

        # Employee tenure analysis
        ('''employee_tenure_analysis''', '''
                                         CREATE VIEW IF NOT EXISTS employee_tenure_analysis AS
                                         SELECT
                                             e.employee_id,
                                             e.full_name,
                                             e.department_name,
                                             e.job_title,
                                             e.hire_date,
                                             e.days_employed,
                                             e.seniority_level,
                                             e.salary,
                                             COUNT(d.dependent_id) as dependents_count,
                                             CASE
                                                 WHEN e.days_employed > 7300 THEN 'Veteran (20+ years)'
                                                 WHEN e.days_employed > 3650 THEN 'Senior (10+ years)'
                                                 WHEN e.days_employed > 1825 THEN 'Experienced (5+ years)'
                                                 WHEN e.days_employed > 365 THEN 'Established (1+ years)'
                                                 ELSE 'New Employee'
                                                 END as tenure_category,
                                             ROUND(e.days_employed / 365.25, 1) as years_employed
                                         FROM employee_details e
                                                  LEFT JOIN dependents d ON e.employee_id = d.employee_id
                                         GROUP BY e.employee_id, e.full_name, e.department_name, e.job_title, e.hire_date, e.days_employed
                                         ORDER BY e.days_employed DESC
                                         '''),

        # Department cost analysis
        ('''department_cost_analysis''', '''
                                         CREATE VIEW IF NOT EXISTS department_cost_analysis AS
                                         SELECT
                                             d.department_name,
                                             d.employee_count,
                                             d.total_payroll,
                                             d.avg_salary,
                                             d.city,
                                             d.country_name,
                                             ROUND(d.total_payroll / NULLIF(d.employee_count, 0), 2) as cost_per_employee,
                                             RANK() OVER (ORDER BY d.total_payroll DESC) as payroll_rank,
                                             RANK() OVER (ORDER BY d.avg_salary DESC) as avg_salary_rank,
                                             ROUND(d.total_payroll * 100.0 / SUM(d.total_payroll) OVER (), 2) as payroll_percentage
                                         FROM department_analytics d
                                         WHERE d.employee_count > 0
                                         ORDER BY d.total_payroll DESC
                                         ''')
    ]

    successful_views = 0
    for view_name, view_sql in views:
        try:
            cursor.execute(view_sql)
            successful_views += 1
            print(f"   ✅ Created view: {view_name}")
        except Exception as e:
            print(f"   ⚠️  View warning for {view_name}: {e}")

    print(f"   ✅ Created {successful_views}/{len(views)} views")

def create_stored_procedures_as_views(cursor):
    """Create complex analytical views that simulate stored procedures"""
    print("📊 Creating analytical views...")

    analytical_views = [
        # Top performers by department
        '''CREATE VIEW IF NOT EXISTS top_performers AS
        SELECT
            e.*,
            RANK() OVER (PARTITION BY e.department_name ORDER BY e.salary DESC) as dept_salary_rank,
            RANK() OVER (ORDER BY e.salary DESC) as overall_salary_rank
        FROM employee_details e
        WHERE e.department_name IS NOT NULL''',

        # Salary gaps analysis
        '''CREATE VIEW IF NOT EXISTS salary_gaps AS
        SELECT
            d.department_name,
            j.job_title,
            MAX(e.salary) - MIN(e.salary) as salary_gap,
            COUNT(e.employee_id) as employee_count,
            AVG(e.salary) as avg_salary
        FROM employees e
                 JOIN departments d ON e.department_id = d.department_id
                 JOIN jobs j ON e.job_id = j.job_id
        GROUP BY d.department_name, j.job_title
        HAVING COUNT(e.employee_id) > 1
        ORDER BY salary_gap DESC''',

        # Promotion candidates (employees below job max salary)
        '''CREATE VIEW IF NOT EXISTS promotion_candidates AS
        SELECT
            e.*,
            e.job_max_salary - e.salary as potential_increase,
            ROUND((e.salary * 100.0 / e.job_max_salary), 2) as salary_utilization
        FROM employee_details e
        WHERE e.salary < e.job_max_salary * 0.9
        ORDER BY e.days_employed DESC, potential_increase DESC'''
    ]

    for view_sql in analytical_views:
        try:
            cursor.execute(view_sql)
        except Exception as e:
            print(f"   ⚠️  Analytical view warning: {e}")

def update_department_employee_counts(cursor):
    """Update employee counts in departments"""
    print("🔄 Updating department employee counts...")

    try:
        cursor.execute('''
                       UPDATE departments
                       SET employee_count = (
                           SELECT COUNT(*)
                           FROM employees
                           WHERE employees.department_id = departments.department_id
                       )
                       ''')
    except Exception as e:
        print(f"   ⚠️  Employee count update warning: {e}")

def create_oracle_hr_database():
    """Create the enhanced Oracle HR database with all features"""
    print("🏛️  Creating enhanced hr_database.db with comprehensive features...")

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
    print("📖 Reading schema and data files...")
    schema_content = read_file_content(schema_file)
    data_content = read_file_content(data_file)

    if not schema_content or not data_content:
        return False

    # Remove existing database
    if os.path.exists('hr_database.db'):
        os.remove('hr_database.db')

    # Create database
    conn = sqlite3.connect('hr_database.db')
    cursor = conn.cursor()

    # Enable foreign key constraints and other pragmas
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.execute("PRAGMA cache_size = 10000")

    try:
        # Create tables from schema
        print("🏗️  Creating tables with foreign key constraints...")
        schema_statements = adapt_schema_for_sqlite(schema_content)

        for i, statement in enumerate(schema_statements, 1):
            try:
                print(f"   Creating table {i}/{len(schema_statements)}...")
                cursor.execute(statement)
            except Exception as e:
                print(f"   ⚠️  Table creation warning: {e}")
                continue

        # Insert data
        print("📊 Inserting sample data...")
        insert_statements = extract_insert_statements(data_content)

        successful_inserts = 0
        for i, statement in enumerate(insert_statements, 1):
            try:
                cursor.execute(statement)
                successful_inserts += 1
                if i % 10 == 0 or i == len(insert_statements):
                    print(f"   Processed {i}/{len(insert_statements)} records...")
            except Exception as e:
                print(f"   ⚠️  Insert error: {e}")
                continue

        print(f"✅ Successfully inserted {successful_inserts}/{len(insert_statements)} records")

        # Create all enhancements
        create_additional_constraints(cursor)
        create_audit_triggers(cursor)
        create_business_triggers(cursor)
        create_comprehensive_indexes(cursor)
        create_enhanced_views(cursor)
        create_stored_procedures_as_views(cursor)
        update_department_employee_counts(cursor)

        # Commit all changes
        conn.commit()

        # Get comprehensive database statistics
        print("📈 Database statistics:")

        # Table statistics
        tables = ['regions', 'countries', 'locations', 'jobs', 'departments', 'employees', 'dependents', 'audit_log']
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   📋 {table}: {count} records")
            except:
                print(f"   📋 {table}: table not found")

        # Views count
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'")
        views_count = cursor.fetchone()[0]
        print(f"   👁️  Views: {views_count}")

        # Indexes count
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
        indexes_count = cursor.fetchone()[0]
        print(f"   🔍 Indexes: {indexes_count}")

        # Triggers count
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'")
        triggers_count = cursor.fetchone()[0]
        print(f"   ⚡ Triggers: {triggers_count}")

        # Foreign keys count
        cursor.execute('''
                       SELECT COUNT(*) FROM sqlite_master
                       WHERE type='table' AND sql LIKE '%FOREIGN KEY%'
                       ''')
        fk_count = cursor.fetchone()[0]
        print(f"   🔗 Tables with FK: {fk_count}")

        conn.close()

        # Get file size
        if os.path.exists('hr_database.db'):
            file_size = os.path.getsize('hr_database.db')
            size_mb = file_size / (1024 * 1024)
            print(f"   💾 File size: {size_mb:.2f} MB")

        print("\n✅ Enhanced hr_database.db created successfully!")
        print("\n🎯 This comprehensive HR database includes:")
        print("   • Complete foreign key relationships")
        print("   • Data validation triggers")
        print("   • Audit trail capabilities")
        print("   • Business logic triggers")
        print("   • Performance-optimized indexes")
        print("   • Rich analytical views")
        print("   • Data integrity constraints")

        print(f"\n🛠️  Usage:")
        print(f"   1. Test with: python db_explorer.py hr_database.db")
        print(f"   2. Explore views: SELECT * FROM employee_details;")
        print(f"   3. Check relationships: SELECT * FROM department_analytics;")
        print(f"   4. Audit trail: SELECT * FROM audit_log;")

        return True

    except Exception as e:
        print(f"❌ Error creating enhanced database: {e}")
        conn.rollback()
        conn.close()
        return False

def create_sample_queries_file():
    """Create a file with sample queries to demonstrate database features"""
    print("📝 Creating sample queries file...")

    sample_queries = """-- Enhanced Oracle HR Database - Sample Queries
-- Generated on: {timestamp}

-- ==============================================
-- BASIC EMPLOYEE QUERIES
-- ==============================================

-- 1. All employees with their complete details
SELECT * FROM employee_details LIMIT 10;

-- 2. Employees by department with salary information
SELECT 
    department_name,
    full_name,
    job_title,
    salary,
    salary_status
FROM employee_details 
ORDER BY department_name, salary DESC;

-- 3. Management hierarchy
SELECT 
    employee_name,
    employee_job,
    manager_name,
    manager_job,
    direct_reports
FROM management_hierarchy
WHERE manager_name IS NOT NULL
ORDER BY direct_reports DESC;

-- ==============================================
-- DEPARTMENT ANALYTICS
-- ==============================================

-- 4. Department cost analysis
SELECT * FROM department_cost_analysis;

-- 5. Department performance metrics
SELECT 
    department_name,
    employee_count,
    avg_salary,
    total_payroll,
    department_size,
    city,
    country_name
FROM department_analytics
ORDER BY total_payroll DESC;

-- ==============================================
-- SALARY ANALYSIS
-- ==============================================

-- 6. Salary analysis by job title
SELECT * FROM salary_analysis ORDER BY avg_salary DESC;

-- 7. Top performers across the company
SELECT 
    full_name,
    department_name,
    job_title,
    salary,
    overall_salary_rank
FROM top_performers
WHERE overall_salary_rank <= 10;

-- 8. Salary gaps within departments and jobs
SELECT * FROM salary_gaps WHERE salary_gap > 2000;

-- 9. Promotion candidates
SELECT 
    full_name,
    department_name,
    job_title,
    salary,
    potential_increase,
    salary_utilization
FROM promotion_candidates
LIMIT 15;

-- ==============================================
-- GEOGRAPHIC ANALYSIS
-- ==============================================

-- 10. Geographic distribution of employees
SELECT * FROM geographic_distribution ORDER BY employees_count DESC;

-- 11. Employees by region and country
SELECT 
    region_name,
    country_name,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary
FROM employee_details
GROUP BY region_name, country_name
ORDER BY employee_count DESC;

-- ==============================================
-- TENURE AND CAREER ANALYSIS
-- ==============================================

-- 12. Employee tenure analysis
SELECT * FROM employee_tenure_analysis ORDER BY years_employed DESC;

-- 13. Employees by seniority level
SELECT 
    seniority_level,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary,
    AVG(years_employed) as avg_tenure
FROM employee_tenure_analysis
GROUP BY seniority_level
ORDER BY avg_tenure DESC;

-- 14. Employees with dependents
SELECT 
    e.full_name,
    e.department_name,
    e.job_title,
    e.salary,
    t.dependents_count
FROM employee_details e
JOIN employee_tenure_analysis t ON e.employee_id = t.employee_id
WHERE t.dependents_count > 0
ORDER BY t.dependents_count DESC;

-- ==============================================
-- FOREIGN KEY RELATIONSHIP QUERIES
-- ==============================================

-- 15. Complete organizational structure
SELECT 
    r.region_name,
    c.country_name,
    l.city,
    d.department_name,
    COUNT(e.employee_id) as employees
FROM regions r
JOIN countries c ON r.region_id = c.region_id
JOIN locations l ON c.country_id = l.country_id
JOIN departments d ON l.location_id = d.location_id
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY r.region_name, c.country_name, l.city, d.department_name
ORDER BY r.region_name, c.country_name, employees DESC;

-- 16. Manager-subordinate relationships
SELECT 
    m.first_name || ' ' || m.last_name as manager,
    m.job_title as manager_title,
    e.first_name || ' ' || e.last_name as employee,
    e.job_title as employee_title,
    d.department_name
FROM employees e
JOIN employees m ON e.manager_id = m.employee_id
JOIN departments d ON e.department_id = d.department_id
JOIN jobs j1 ON e.job_id = j1.job_id
JOIN jobs j2 ON m.job_id = j2.job_id
ORDER BY manager, employee;

-- ==============================================
-- AUDIT AND TRIGGER TESTING
-- ==============================================

-- 17. View audit log
SELECT 
    table_name,
    operation,
    new_values,
    changed_at
FROM audit_log
ORDER BY changed_at DESC
LIMIT 20;

-- 18. Test business rules (these will fail due to triggers)
-- Uncomment to test:
-- INSERT INTO employees (employee_id, first_name, last_name, email, hire_date, job_id, salary, manager_id, department_id)
-- VALUES (999, 'Test', 'User', 'invalid-email', '2025-01-01', 1, -1000, 999, 1);

-- ==============================================
-- PERFORMANCE QUERIES
-- ==============================================

-- 19. Query with multiple indexes (should be fast)
SELECT 
    e.full_name,
    e.department_name,
    e.salary
FROM employee_details e
WHERE e.department_name = 'IT'
  AND e.salary > 5000
  AND e.hire_date > '1995-01-01'
ORDER BY e.salary DESC;

-- 20. Complex analytical query
SELECT 
    d.department_name,
    COUNT(e.employee_id) as total_employees,
    COUNT(CASE WHEN e.salary > j.max_salary * 0.8 THEN 1 END) as high_earners,
    AVG(e.salary) as avg_salary,
    MAX(e.salary) as max_salary,
    COUNT(DISTINCT e.job_id) as unique_jobs
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
LEFT JOIN jobs j ON e.job_id = j.job_id
GROUP BY d.department_name
HAVING COUNT(e.employee_id) > 0
ORDER BY avg_salary DESC;

-- ==============================================
-- CONSTRAINT TESTING
-- ==============================================

-- 21. Check data integrity
SELECT 'Employees without departments' as check_type, COUNT(*) as count
FROM employees WHERE department_id IS NULL
UNION ALL
SELECT 'Departments without locations', COUNT(*)
FROM departments WHERE location_id IS NULL
UNION ALL
SELECT 'Employees without jobs', COUNT(*)
FROM employees WHERE job_id IS NULL
UNION ALL
SELECT 'Employees with invalid managers', COUNT(*)
FROM employees e1 
WHERE manager_id IS NOT NULL 
  AND NOT EXISTS (SELECT 1 FROM employees e2 WHERE e2.employee_id = e1.manager_id);

-- ==============================================
-- ADVANCED ANALYTICS
-- ==============================================

-- 22. Salary distribution by quartiles
WITH salary_quartiles AS (
    SELECT 
        department_name,
        salary,
        NTILE(4) OVER (PARTITION BY department_name ORDER BY salary) as quartile
    FROM employee_details
    WHERE department_name IS NOT NULL
)
SELECT 
    department_name,
    quartile,
    COUNT(*) as employee_count,
    MIN(salary) as min_salary,
    MAX(salary) as max_salary,
    AVG(salary) as avg_salary
FROM salary_quartiles
GROUP BY department_name, quartile
ORDER BY department_name, quartile;

-- 23. Year-over-year hiring trends
SELECT 
    strftime('%Y', hire_date) as hire_year,
    COUNT(*) as hires,
    AVG(salary) as avg_starting_salary
FROM employees
GROUP BY strftime('%Y', hire_date)
ORDER BY hire_year;

-- 24. Department efficiency (revenue per employee simulation)
SELECT 
    d.department_name,
    d.employee_count,
    d.total_payroll,
    d.avg_salary,
    -- Simulated revenue calculation
    ROUND(d.total_payroll * 
        CASE d.department_name
            WHEN 'Sales' THEN 4.5
            WHEN 'IT' THEN 3.2
            WHEN 'Marketing' THEN 2.8
            WHEN 'Finance' THEN 2.5
            ELSE 2.0
        END, 2) as estimated_revenue,
    ROUND((d.total_payroll * 
        CASE d.department_name
            WHEN 'Sales' THEN 4.5
            WHEN 'IT' THEN 3.2
            WHEN 'Marketing' THEN 2.8
            WHEN 'Finance' THEN 2.5
            ELSE 2.0
        END) - d.total_payroll, 2) as estimated_profit
FROM department_analytics d
WHERE d.employee_count > 0
ORDER BY estimated_profit DESC;

-- End of sample queries
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    try:
        with open('hr_sample_queries.sql', 'w', encoding='utf-8') as f:
            f.write(sample_queries)
        print("   ✅ Created hr_sample_queries.sql")
    except Exception as e:
        print(f"   ⚠️  Error creating sample queries file: {e}")

def create_database_documentation():
    """Create comprehensive documentation for the database"""
    print("📚 Creating database documentation...")

    documentation = """# Enhanced Oracle HR Database Documentation

## Overview
This is a comprehensive implementation of the Oracle HR sample database for SQLite, enhanced with:
- Complete foreign key relationships
- Data validation triggers
- Audit trail system
- Business logic triggers
- Performance indexes
- Rich analytical views
- Data integrity constraints

## Database Schema

### Core Tables

#### 1. regions
- **Purpose**: Geographic regions
- **Key Fields**: region_id (PK), region_name
- **Relationships**: One-to-many with countries

#### 2. countries
- **Purpose**: Countries within regions
- **Key Fields**: country_id (PK), country_name, region_id (FK)
- **Relationships**: 
  - Many-to-one with regions
  - One-to-many with locations

#### 3. locations
- **Purpose**: Physical office locations
- **Key Fields**: location_id (PK), street_address, city, country_id (FK)
- **Relationships**: 
  - Many-to-one with countries
  - One-to-many with departments

#### 4. jobs
- **Purpose**: Job definitions and salary ranges
- **Key Fields**: job_id (PK), job_title, min_salary, max_salary
- **Relationships**: One-to-many with employees

#### 5. departments
- **Purpose**: Organizational departments
- **Key Fields**: department_id (PK), department_name, location_id (FK), employee_count
- **Relationships**: 
  - Many-to-one with locations
  - One-to-many with employees

#### 6. employees
- **Purpose**: Employee records (central table)
- **Key Fields**: employee_id (PK), first_name, last_name, email, job_id (FK), manager_id (FK), department_id (FK)
- **Relationships**: 
  - Many-to-one with jobs
  - Many-to-one with departments
  - Self-referencing for manager hierarchy
  - One-to-many with dependents

#### 7. dependents
- **Purpose**: Employee family members
- **Key Fields**: dependent_id (PK), first_name, last_name, relationship, employee_id (FK)
- **Relationships**: Many-to-one with employees

#### 8. audit_log
- **Purpose**: Audit trail for data changes
- **Key Fields**: audit_id (PK), table_name, operation, old_values, new_values, changed_at
- **Relationships**: None (logging table)

## Enhanced Features

### Triggers

#### Data Validation Triggers
1. **check_email_format**: Validates email format
2. **check_salary_range**: Ensures positive salaries
3. **check_hire_date**: Prevents future hire dates
4. **check_manager_hierarchy**: Prevents self-management
5. **validate_employee_salary**: Validates salary against job range

#### Audit Triggers
1. **employee_audit_insert/update/delete**: Tracks employee changes
2. **department_audit_insert/update**: Tracks department changes

#### Business Logic Triggers
1. **update_dept_employee_count**: Maintains employee counts
2. **generate_employee_email**: Auto-generates emails

### Views

#### Core Views
1. **employee_details**: Complete employee information with joins
2. **department_analytics**: Department performance metrics
3. **salary_analysis**: Job-based salary analysis
4. **management_hierarchy**: Manager-subordinate relationships
5. **geographic_distribution**: Location-based statistics

#### Analytical Views
1. **employee_tenure_analysis**: Tenure and seniority analysis
2. **department_cost_analysis**: Financial analysis by department
3. **top_performers**: Salary ranking analysis
4. **salary_gaps**: Compensation gap analysis
5. **promotion_candidates**: Employees eligible for promotion

### Indexes

#### Primary Relationship Indexes
- Foreign key indexes for all relationships
- Unique constraints on key business fields

#### Performance Indexes
- Composite indexes for common query patterns
- Text search indexes for names and emails
- Date range indexes for temporal queries

#### Analytical Indexes
- Salary range indexes
- Geographic indexes
- Audit trail indexes

## Usage Examples

### Basic Queries
```sql
-- Get all employees with details
SELECT * FROM employee_details;

-- Department summary
SELECT * FROM department_analytics;
```

### Analytical Queries
```sql
-- Top earners by department
SELECT department_name, full_name, salary 
FROM employee_details 
ORDER BY department_name, salary DESC;

-- Promotion candidates
SELECT * FROM promotion_candidates;
```

### Audit Queries
```sql
-- Recent changes
SELECT * FROM audit_log 
ORDER BY changed_at DESC 
LIMIT 10;
```

## Data Integrity Features

### Foreign Key Constraints
- All relationships enforced at database level
- Cascade operations where appropriate
- Referential integrity maintained

### Check Constraints (via Triggers)
- Email format validation
- Salary range validation
- Date consistency checks
- Business rule enforcement

### Unique Constraints
- Unique employee emails
- Unique region/country/department names
- Prevent duplicate key data

## Performance Optimizations

### Indexing Strategy
- All foreign keys indexed
- Composite indexes for common queries
- Unique indexes for business constraints

### Query Optimization
- Views pre-calculate common joins
- Indexed columns for WHERE clauses
- Efficient sorting and grouping

## Maintenance Procedures

### Regular Maintenance
```sql
-- Update statistics
ANALYZE;

-- Check integrity
PRAGMA integrity_check;

-- Vacuum database
VACUUM;
```

### Monitoring Queries
```sql
-- Check constraint violations
SELECT * FROM audit_log WHERE operation = 'ERROR';

-- Monitor department employee counts
SELECT department_name, employee_count 
FROM departments 
WHERE employee_count != (
    SELECT COUNT(*) FROM employees 
    WHERE department_id = departments.department_id
);
```

## Testing the Database

### Constraint Testing
```sql
-- Test email validation (should fail)
INSERT INTO employees (..., email, ...) VALUES (..., 'invalid-email', ...);

-- Test salary validation (should fail)
INSERT INTO employees (..., salary, ...) VALUES (..., -1000, ...);
```

### Performance Testing
```sql
-- Test index usage
EXPLAIN QUERY PLAN 
SELECT * FROM employee_details 
WHERE department_name = 'IT' AND salary > 5000;
```

## Extension Points

### Adding New Features
1. **Additional Triggers**: Add business rules as needed
2. **New Views**: Create specialized analytical views
3. **Custom Functions**: Add calculated fields
4. **Additional Audit**: Extend audit coverage

### Integration Points
1. **ETL Processes**: Use views for data extraction
2. **Reporting Tools**: Connect to analytical views
3. **Applications**: Use employee_details view for UI
4. **APIs**: Expose views as REST endpoints

---
Generated by Enhanced Oracle HR Database Creator
"""

    try:
        with open('hr_database_documentation.md', 'w', encoding='utf-8') as f:
            f.write(documentation)
        print("   ✅ Created hr_database_documentation.md")
    except Exception as e:
        print(f"   ⚠️  Error creating documentation: {e}")

def main():
    """Main function"""
    print("🏛️  Enhanced Oracle HR Database Creator")
    print("=" * 60)

    if create_oracle_hr_database():
        create_sample_queries_file()
        create_database_documentation()

        print("\n🎉 SUCCESS! Enhanced HR database created with:")
        print("   📁 hr_database.db - The complete database")
        print("   📄 hr_sample_queries.sql - 24 sample queries")
        print("   📚 hr_database_documentation.md - Complete documentation")

        print("\n🚀 Next Steps:")
        print("   1. Test relationships: python db_explorer.py hr_database.db")
        print("   2. Run sample queries from hr_sample_queries.sql")
        print("   3. Explore the relationship visualization tool")
        print("   4. Read the documentation for advanced features")

        print("\n💡 Pro Tips:")
        print("   • Use employee_details view for most queries")
        print("   • Check audit_log for data change tracking")
        print("   • Use department_analytics for business insights")
        print("   • Explore promotion_candidates for HR planning")

        return 0
    else:
        print("\n💥 Failed to create enhanced database.")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())