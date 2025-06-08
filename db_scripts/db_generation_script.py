#!/usr/bin/env python3
"""
Sample Database Creator
Creates realistic sample databases for testing the Universal Database Explorer

Usage: python create_sample_databases.py
This will create multiple .db files with sample data
"""

import sqlite3
import random
from datetime import datetime, timedelta
import os

def create_student_database():
    """Create a comprehensive student management database"""
    print("📚 Creating student_database.db...")

    conn = sqlite3.connect('student_database.db')
    cursor = conn.cursor()

    # Create tables with proper relationships and constraints

    # Departments table
    cursor.execute('''
                   CREATE TABLE departments (
                                                dept_id INTEGER PRIMARY KEY,
                                                dept_name TEXT NOT NULL UNIQUE,
                                                dept_code TEXT NOT NULL UNIQUE,
                                                building TEXT,
                                                budget REAL DEFAULT 0,
                                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   )
                   ''')

    # Courses table
    cursor.execute('''
                   CREATE TABLE courses (
                                            course_id INTEGER PRIMARY KEY,
                                            course_code TEXT NOT NULL UNIQUE,
                                            course_name TEXT NOT NULL,
                                            credits INTEGER NOT NULL CHECK (credits > 0),
                                            dept_id INTEGER,
                                            description TEXT,
                                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
                   )
                   ''')

    # Students table
    cursor.execute('''
                   CREATE TABLE students (
                                             student_id INTEGER PRIMARY KEY,
                                             first_name TEXT NOT NULL,
                                             last_name TEXT NOT NULL,
                                             email TEXT UNIQUE NOT NULL,
                                             phone TEXT,
                                             date_of_birth DATE,
                                             enrollment_date DATE NOT NULL,
                                             major_dept_id INTEGER,
                                             gpa REAL CHECK (gpa >= 0 AND gpa <= 4.0),
                                             status TEXT DEFAULT 'active' CHECK (status IN ('active', 'graduated', 'suspended', 'transferred')),
                                             FOREIGN KEY (major_dept_id) REFERENCES departments(dept_id)
                   )
                   ''')

    # Instructors table
    cursor.execute('''
                   CREATE TABLE instructors (
                                                instructor_id INTEGER PRIMARY KEY,
                                                first_name TEXT NOT NULL,
                                                last_name TEXT NOT NULL,
                                                email TEXT UNIQUE NOT NULL,
                                                phone TEXT,
                                                dept_id INTEGER,
                                                hire_date DATE,
                                                salary REAL,
                                                office_number TEXT,
                                                FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
                   )
                   ''')

    # Enrollments table (many-to-many relationship)
    cursor.execute('''
                   CREATE TABLE enrollments (
                                                enrollment_id INTEGER PRIMARY KEY,
                                                student_id INTEGER,
                                                course_id INTEGER,
                                                instructor_id INTEGER,
                                                semester TEXT NOT NULL,
                                                year INTEGER NOT NULL,
                                                grade TEXT CHECK (grade IN ('A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F', 'W', 'I')),
                                                enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                FOREIGN KEY (student_id) REFERENCES students(student_id),
                                                FOREIGN KEY (course_id) REFERENCES courses(course_id),
                                                FOREIGN KEY (instructor_id) REFERENCES instructors(instructor_id),
                                                UNIQUE(student_id, course_id, semester, year)
                   )
                   ''')

    # Insert sample data

    # Departments
    departments = [
        (1, 'Computer Science', 'CS', 'Tech Building', 2500000),
        (2, 'Mathematics', 'MATH', 'Science Hall', 1800000),
        (3, 'Physics', 'PHYS', 'Science Hall', 2200000),
        (4, 'Business Administration', 'BUS', 'Business Center', 3000000),
        (5, 'English Literature', 'ENG', 'Liberal Arts', 1500000),
        (6, 'Psychology', 'PSY', 'Social Sciences', 1700000)
    ]

    cursor.executemany('''
                       INSERT INTO departments (dept_id, dept_name, dept_code, building, budget)
                       VALUES (?, ?, ?, ?, ?)
                       ''', departments)

    # Courses
    courses = [
        (1, 'CS101', 'Introduction to Programming', 3, 1, 'Basic programming concepts using Python'),
        (2, 'CS201', 'Data Structures', 4, 1, 'Arrays, lists, trees, graphs, and algorithms'),
        (3, 'CS301', 'Database Systems', 3, 1, 'Relational databases, SQL, and database design'),
        (4, 'CS401', 'Software Engineering', 4, 1, 'Software development lifecycle and methodologies'),
        (5, 'MATH101', 'Calculus I', 4, 2, 'Differential calculus and applications'),
        (6, 'MATH201', 'Linear Algebra', 3, 2, 'Vectors, matrices, and linear transformations'),
        (7, 'PHYS101', 'General Physics I', 4, 3, 'Mechanics, waves, and thermodynamics'),
        (8, 'BUS101', 'Introduction to Business', 3, 4, 'Fundamentals of business operations'),
        (9, 'ENG101', 'English Composition', 3, 5, 'Writing and communication skills'),
        (10, 'PSY101', 'Introduction to Psychology', 3, 6, 'Basic psychological principles')
    ]

    cursor.executemany('''
                       INSERT INTO courses (course_id, course_code, course_name, credits, dept_id, description)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', courses)

    # Students
    first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Emily', 'Chris', 'Lisa', 'Tom', 'Anna',
                   'Robert', 'Maria', 'James', 'Jennifer', 'William', 'Elizabeth', 'Michael', 'Jessica']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson']

    students_data = []
    for i in range(1, 151):  # 150 students
        first = random.choice(first_names)
        last = random.choice(last_names)
        email = f"{first.lower()}.{last.lower()}{i}@university.edu"
        phone = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

        # Random birth date (18-25 years old)
        birth_year = random.randint(1998, 2005)
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)
        birth_date = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"

        # Random enrollment date (within last 4 years)
        enroll_year = random.randint(2020, 2024)
        enroll_month = random.choice([1, 8])  # January or August enrollment
        enroll_date = f"{enroll_year}-{enroll_month:02d}-15"

        major_dept = random.randint(1, 6)
        gpa = round(random.uniform(2.0, 4.0), 2)
        status = random.choices(['active', 'graduated', 'suspended'], weights=[85, 12, 3])[0]

        students_data.append((i, first, last, email, phone, birth_date, enroll_date, major_dept, gpa, status))

    cursor.executemany('''
                       INSERT INTO students (student_id, first_name, last_name, email, phone, date_of_birth,
                                             enrollment_date, major_dept_id, gpa, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', students_data)

    # Instructors
    instructor_names = [
        ('Dr. Alan', 'Turing', 1, 95000, 'CS-101'),
        ('Prof. Ada', 'Lovelace', 1, 105000, 'CS-102'),
        ('Dr. Donald', 'Knuth', 1, 120000, 'CS-103'),
        ('Prof. Marie', 'Curie', 3, 110000, 'PHYS-201'),
        ('Dr. Albert', 'Einstein', 3, 125000, 'PHYS-202'),
        ('Prof. Isaac', 'Newton', 2, 100000, 'MATH-301'),
        ('Dr. Katherine', 'Johnson', 2, 95000, 'MATH-302'),
        ('Prof. Warren', 'Buffett', 4, 130000, 'BUS-401'),
        ('Dr. Maya', 'Angelou', 5, 85000, 'ENG-501'),
        ('Prof. Sigmund', 'Freud', 6, 90000, 'PSY-601')
    ]

    instructors_data = []
    for i, (first, last, dept, salary, office) in enumerate(instructor_names, 1):
        email = f"{first.lower().replace('dr. ', '').replace('prof. ', '')}.{last.lower()}@university.edu"
        phone = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        hire_year = random.randint(2010, 2023)
        hire_date = f"{hire_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        instructors_data.append((i, first, last, email, phone, dept, hire_date, salary, office))

    cursor.executemany('''
                       INSERT INTO instructors (instructor_id, first_name, last_name, email, phone, dept_id,
                                                hire_date, salary, office_number)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', instructors_data)

    # Enrollments
    enrollments_data = []
    enrollment_id = 1

    for student_id in range(1, 151):
        # Each student enrolls in 3-6 courses
        num_courses = random.randint(3, 6)
        student_courses = random.sample(range(1, 11), num_courses)

        for course_id in student_courses:
            instructor_id = random.randint(1, 10)
            semester = random.choice(['Fall', 'Spring', 'Summer'])
            year = random.randint(2022, 2024)
            grade = random.choices(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F'],
                                   weights=[5, 15, 12, 18, 15, 10, 8, 7, 5, 3, 1, 1])[0]

            enrollments_data.append((enrollment_id, student_id, course_id, instructor_id, semester, year, grade))
            enrollment_id += 1

    cursor.executemany('''
                       INSERT INTO enrollments (enrollment_id, student_id, course_id, instructor_id, semester, year, grade)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ''', enrollments_data)

    # Create indexes for better performance
    cursor.execute('CREATE INDEX idx_students_email ON students(email)')
    cursor.execute('CREATE INDEX idx_students_major ON students(major_dept_id)')
    cursor.execute('CREATE INDEX idx_enrollments_student ON enrollments(student_id)')
    cursor.execute('CREATE INDEX idx_enrollments_course ON enrollments(course_id)')
    cursor.execute('CREATE INDEX idx_courses_dept ON courses(dept_id)')

    # Create a view for student information
    cursor.execute('''
                   CREATE VIEW student_info AS
                   SELECT
                       s.student_id,
                       s.first_name || ' ' || s.last_name AS full_name,
                       s.email,
                       d.dept_name AS major,
                       s.gpa,
                       s.status
                   FROM students s
                            LEFT JOIN departments d ON s.major_dept_id = d.dept_id
                   ''')

    # Create a trigger to update GPA when grades change
    cursor.execute('''
                   CREATE TRIGGER update_gpa_after_grade_change
                       AFTER UPDATE OF grade ON enrollments
                       WHEN NEW.grade IS NOT NULL AND OLD.grade != NEW.grade
                   BEGIN
                   UPDATE students
                   SET gpa = (
                       SELECT ROUND(AVG(
                                            CASE grade
                                                WHEN 'A+' THEN 4.0 WHEN 'A' THEN 4.0 WHEN 'A-' THEN 3.7
                                                WHEN 'B+' THEN 3.3 WHEN 'B' THEN 3.0 WHEN 'B-' THEN 2.7
                                                WHEN 'C+' THEN 2.3 WHEN 'C' THEN 2.0 WHEN 'C-' THEN 1.7
                                                WHEN 'D+' THEN 1.3 WHEN 'D' THEN 1.0 WHEN 'F' THEN 0.0
                                                ELSE NULL END
                                    ), 2)
                       FROM enrollments
                       WHERE student_id = NEW.student_id
                         AND grade IS NOT NULL
                         AND grade NOT IN ('W', 'I')
                   )
                   WHERE student_id = NEW.student_id;
                   END
                   ''')

    conn.commit()
    conn.close()
    print("✅ student_database.db created with 150 students, 10 courses, 6 departments, and enrollment data")

def create_sales_database():
    """Create a comprehensive sales and inventory database"""
    print("💰 Creating sales_database.db...")

    conn = sqlite3.connect('sales_database.db')
    cursor = conn.cursor()

    # Customers table
    cursor.execute('''
                   CREATE TABLE customers (
                                              customer_id INTEGER PRIMARY KEY,
                                              first_name TEXT NOT NULL,
                                              last_name TEXT NOT NULL,
                                              email TEXT UNIQUE,
                                              phone TEXT,
                                              address TEXT,
                                              city TEXT,
                                              state TEXT,
                                              zip_code TEXT,
                                              registration_date DATE DEFAULT CURRENT_DATE,
                                              customer_type TEXT DEFAULT 'regular' CHECK (customer_type IN ('regular', 'premium', 'vip'))
                   )
                   ''')

    # Categories table
    cursor.execute('''
                   CREATE TABLE categories (
                                               category_id INTEGER PRIMARY KEY,
                                               category_name TEXT NOT NULL UNIQUE,
                                               description TEXT,
                                               parent_category_id INTEGER,
                                               FOREIGN KEY (parent_category_id) REFERENCES categories(category_id)
                   )
                   ''')

    # Products table
    cursor.execute('''
                   CREATE TABLE products (
                                             product_id INTEGER PRIMARY KEY,
                                             product_name TEXT NOT NULL,
                                             category_id INTEGER,
                                             price REAL NOT NULL CHECK (price >= 0),
                                             cost REAL NOT NULL CHECK (cost >= 0),
                                             stock_quantity INTEGER DEFAULT 0 CHECK (stock_quantity >= 0),
                                             reorder_level INTEGER DEFAULT 10,
                                             supplier TEXT,
                                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                             FOREIGN KEY (category_id) REFERENCES categories(category_id)
                   )
                   ''')

    # Orders table
    cursor.execute('''
                   CREATE TABLE orders (
                                           order_id INTEGER PRIMARY KEY,
                                           customer_id INTEGER,
                                           order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                           total_amount REAL DEFAULT 0,
                                           tax_amount REAL DEFAULT 0,
                                           shipping_cost REAL DEFAULT 0,
                                           status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')),
                                           payment_method TEXT CHECK (payment_method IN ('cash', 'credit_card', 'debit_card', 'paypal', 'check')),
                                           FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                   )
                   ''')

    # Order items table
    cursor.execute('''
                   CREATE TABLE order_items (
                                                item_id INTEGER PRIMARY KEY,
                                                order_id INTEGER,
                                                product_id INTEGER,
                                                quantity INTEGER NOT NULL CHECK (quantity > 0),
                                                unit_price REAL NOT NULL CHECK (unit_price >= 0),
                                                line_total REAL GENERATED ALWAYS AS (quantity * unit_price) STORED,
                                                FOREIGN KEY (order_id) REFERENCES orders(order_id),
                                                FOREIGN KEY (product_id) REFERENCES products(product_id)
                   )
                   ''')

    # Insert sample data

    # Categories
    categories = [
        (1, 'Electronics', 'Electronic devices and accessories', None),
        (2, 'Computers', 'Desktop and laptop computers', 1),
        (3, 'Smartphones', 'Mobile phones and accessories', 1),
        (4, 'Books', 'Physical and digital books', None),
        (5, 'Fiction', 'Fiction books and novels', 4),
        (6, 'Non-Fiction', 'Educational and reference books', 4),
        (7, 'Clothing', 'Apparel and accessories', None),
        (8, 'Home & Garden', 'Home improvement and gardening', None),
        (9, 'Sports', 'Sports equipment and gear', None),
        (10, 'Toys & Games', 'Children toys and board games', None)
    ]

    cursor.executemany('''
                       INSERT INTO categories (category_id, category_name, description, parent_category_id)
                       VALUES (?, ?, ?, ?)
                       ''', categories)

    # Products
    products = [
        ('Laptop Pro 15"', 2, 1299.99, 800.00, 45, 'TechCorp'),
        ('Gaming Desktop', 2, 1899.99, 1200.00, 23, 'GameTech'),
        ('iPhone 13', 3, 999.99, 600.00, 67, 'Apple'),
        ('Samsung Galaxy S21', 3, 849.99, 520.00, 52, 'Samsung'),
        ('Wireless Earbuds', 3, 199.99, 80.00, 156, 'AudioTech'),
        ('Python Programming Guide', 6, 49.99, 20.00, 89, 'TechBooks'),
        ('Mystery Novel Collection', 5, 29.99, 12.00, 134, 'BookHouse'),
        ('Running Shoes', 9, 129.99, 60.00, 78, 'SportsBrand'),
        ('Coffee Maker', 8, 89.99, 45.00, 34, 'KitchenPro'),
        ('Board Game Deluxe', 10, 59.99, 25.00, 67, 'GameWorks'),
        ('Bluetooth Speaker', 1, 79.99, 35.00, 92, 'SoundTech'),
        ('Fitness Tracker', 1, 149.99, 70.00, 76, 'HealthTech'),
        ('Cookbook Collection', 6, 39.99, 15.00, 45, 'CookBooks'),
        ('Winter Jacket', 7, 179.99, 80.00, 29, 'ClothingCo'),
        ('Garden Tools Set', 8, 119.99, 55.00, 38, 'GardenPro')
    ]

    products_data = []
    for i, (name, cat_id, price, cost, stock, supplier) in enumerate(products, 1):
        products_data.append((i, name, cat_id, price, cost, stock, 10, supplier))

    cursor.executemany('''
                       INSERT INTO products (product_id, product_name, category_id, price, cost, stock_quantity, reorder_level, supplier)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', products_data)

    # Customers
    customer_data = []
    first_names = ['Alex', 'Sarah', 'Mike', 'Jennifer', 'David', 'Lisa', 'Chris', 'Emily', 'James', 'Anna']
    last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA']

    for i in range(1, 201):  # 200 customers
        first = random.choice(first_names)
        last = random.choice(last_names)
        email = f"{first.lower()}.{last.lower()}{i}@email.com"
        phone = f"({random.randint(200, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"

        city_idx = random.randint(0, len(cities)-1)
        city = cities[city_idx]
        state = states[city_idx]

        address = f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Park Rd', 'First St', 'Second Ave'])}"
        zip_code = f"{random.randint(10000, 99999)}"

        reg_year = random.randint(2020, 2024)
        reg_month = random.randint(1, 12)
        reg_day = random.randint(1, 28)
        reg_date = f"{reg_year}-{reg_month:02d}-{reg_day:02d}"

        customer_type = random.choices(['regular', 'premium', 'vip'], weights=[70, 25, 5])[0]

        customer_data.append((i, first, last, email, phone, address, city, state, zip_code, reg_date, customer_type))

    cursor.executemany('''
                       INSERT INTO customers (customer_id, first_name, last_name, email, phone, address, city, state, zip_code, registration_date, customer_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', customer_data)

    # Orders and Order Items
    order_data = []
    order_items_data = []
    item_id = 1

    for order_id in range(1, 501):  # 500 orders
        customer_id = random.randint(1, 200)

        # Random order date within last 2 years
        days_ago = random.randint(0, 730)
        order_date = datetime.now() - timedelta(days=days_ago)
        order_date_str = order_date.strftime('%Y-%m-%d %H:%M:%S')

        status = random.choices(['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
                                weights=[5, 10, 15, 65, 5])[0]
        payment_method = random.choice(['cash', 'credit_card', 'debit_card', 'paypal'])

        # Generate order items
        num_items = random.randint(1, 5)
        order_total = 0

        order_items = random.sample(range(1, 16), num_items)  # Random products

        for product_id in order_items:
            quantity = random.randint(1, 3)
            # Get product price (simplified - using index)
            unit_price = products[product_id - 1][2]  # price from products list

            order_items_data.append((item_id, order_id, product_id, quantity, unit_price))
            order_total += quantity * unit_price
            item_id += 1

        tax_amount = round(order_total * 0.08, 2)  # 8% tax
        shipping_cost = 0 if order_total > 50 else 5.99

        order_data.append((order_id, customer_id, order_date_str, order_total, tax_amount, shipping_cost, status, payment_method))

    cursor.executemany('''
                       INSERT INTO orders (order_id, customer_id, order_date, total_amount, tax_amount, shipping_cost, status, payment_method)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', order_data)

    cursor.executemany('''
                       INSERT INTO order_items (item_id, order_id, product_id, quantity, unit_price)
                       VALUES (?, ?, ?, ?, ?)
                       ''', order_items_data)

    # Create indexes
    cursor.execute('CREATE INDEX idx_customers_email ON customers(email)')
    cursor.execute('CREATE INDEX idx_orders_customer ON orders(customer_id)')
    cursor.execute('CREATE INDEX idx_orders_date ON orders(order_date)')
    cursor.execute('CREATE INDEX idx_products_category ON products(category_id)')
    cursor.execute('CREATE INDEX idx_order_items_order ON order_items(order_id)')
    cursor.execute('CREATE INDEX idx_order_items_product ON order_items(product_id)')

    # Create views
    cursor.execute('''
                   CREATE VIEW sales_summary AS
                   SELECT
                           DATE(o.order_date) as sale_date,
                           COUNT(DISTINCT o.order_id) as orders_count,
                           SUM(o.total_amount) as daily_revenue,
                           AVG(o.total_amount) as avg_order_value
                           FROM orders o
                           WHERE o.status != 'cancelled'
                           GROUP BY DATE(o.order_date)
                   ''')

    cursor.execute('''
                   CREATE VIEW top_products AS
                   SELECT
                       p.product_name,
                       SUM(oi.quantity) as total_sold,
                       SUM(oi.line_total) as total_revenue
                   FROM products p
                            JOIN order_items oi ON p.product_id = oi.product_id
                            JOIN orders o ON oi.order_id = o.order_id
                   WHERE o.status != 'cancelled'
        GROUP BY p.product_id, p.product_name
        ORDER BY total_sold DESC
                   ''')

    # Create trigger to update stock quantity
    cursor.execute('''
                   CREATE TRIGGER update_stock_after_order
                       AFTER INSERT ON order_items
                   BEGIN
                       UPDATE products
                       SET stock_quantity = stock_quantity - NEW.quantity
                       WHERE product_id = NEW.product_id;
                   END
                   ''')

    conn.commit()
    conn.close()
    print("✅ sales_database.db created with 200 customers, 15 products, 500 orders, and sales data")

def create_company_database():
    """Create a company HR and project management database"""
    print("🏢 Creating company_database.db...")

    conn = sqlite3.connect('company_database.db')
    cursor = conn.cursor()

    # Departments table
    cursor.execute('''
                   CREATE TABLE departments (
                                                dept_id INTEGER PRIMARY KEY,
                                                dept_name TEXT NOT NULL UNIQUE,
                                                manager_id INTEGER,
                                                budget REAL DEFAULT 0,
                                                location TEXT,
                                                phone TEXT
                   )
                   ''')

    # Employees table
    cursor.execute('''
                   CREATE TABLE employees (
                                              employee_id INTEGER PRIMARY KEY,
                                              first_name TEXT NOT NULL,
                                              last_name TEXT NOT NULL,
                                              email TEXT UNIQUE NOT NULL,
                                              phone TEXT,
                                              hire_date DATE NOT NULL,
                                              job_title TEXT NOT NULL,
                                              dept_id INTEGER,
                                              salary REAL CHECK (salary > 0),
                                              manager_id INTEGER,
                                              status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'terminated')),
                                              FOREIGN KEY (dept_id) REFERENCES departments(dept_id),
                                              FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
                   )
                   ''')

    # Projects table
    cursor.execute('''
                   CREATE TABLE projects (
                                             project_id INTEGER PRIMARY KEY,
                                             project_name TEXT NOT NULL,
                                             description TEXT,
                                             start_date DATE,
                                             end_date DATE,
                                             budget REAL,
                                             status TEXT DEFAULT 'planning' CHECK (status IN ('planning', 'active', 'completed', 'cancelled')),
                                             manager_id INTEGER,
                                             FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
                   )
                   ''')

    # Project assignments table
    cursor.execute('''
                   CREATE TABLE project_assignments (
                                                        assignment_id INTEGER PRIMARY KEY,
                                                        project_id INTEGER,
                                                        employee_id INTEGER,
                                                        role TEXT,
                                                        hours_allocated REAL DEFAULT 0,
                                                        start_date DATE,
                                                        end_date DATE,
                                                        FOREIGN KEY (project_id) REFERENCES projects(project_id),
                                                        FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
                   )
                   ''')

    # Time tracking table
    cursor.execute('''
                   CREATE TABLE time_tracking (
                                                  time_id INTEGER PRIMARY KEY,
                                                  employee_id INTEGER,
                                                  project_id INTEGER,
                                                  work_date DATE,
                                                  hours_worked REAL CHECK (hours_worked >= 0 AND hours_worked <= 24),
                                                  description TEXT,
                                                  FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
                                                  FOREIGN KEY (project_id) REFERENCES projects(project_id)
                   )
                   ''')

    # Insert sample data

    # Departments
    departments = [
        (1, 'Human Resources', None, 500000, 'Building A, Floor 2', '555-0100'),
        (2, 'Engineering', None, 2000000, 'Building B, Floor 3-5', '555-0200'),
        (3, 'Marketing', None, 800000, 'Building A, Floor 3', '555-0300'),
        (4, 'Sales', None, 1200000, 'Building C, Floor 1-2', '555-0400'),
        (5, 'Finance', None, 600000, 'Building A, Floor 1', '555-0500'),
        (6, 'IT Support', None, 400000, 'Building B, Floor 1', '555-0600')
    ]

    cursor.executemany('''
                       INSERT INTO departments (dept_id, dept_name, manager_id, budget, location, phone)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', departments)

    # Employees
    job_titles = {
        1: ['HR Manager', 'HR Specialist', 'Recruiter', 'Benefits Coordinator'],
        2: ['Software Engineer', 'Senior Developer', 'Tech Lead', 'DevOps Engineer', 'QA Engineer'],
        3: ['Marketing Manager', 'Marketing Specialist', 'Content Creator', 'Social Media Manager'],
        4: ['Sales Manager', 'Sales Representative', 'Account Manager', 'Business Development'],
        5: ['Finance Manager', 'Accountant', 'Financial Analyst', 'Budget Analyst'],
        6: ['IT Manager', 'System Administrator', 'Help Desk Technician', 'Network Engineer']
    }

    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Chris', 'Lisa', 'Robert', 'Anna',
                   'James', 'Jennifer', 'William', 'Elizabeth', 'Daniel', 'Jessica', 'Matthew', 'Ashley']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson']

    employees_data = []
    employee_id = 1

    # Create managers first
    managers = []
    for dept_id in range(1, 7):
        first = random.choice(first_names)
        last = random.choice(last_names)
        email = f"{first.lower()}.{last.lower()}@company.com"
        phone = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

        hire_year = random.randint(2015, 2020)
        hire_date = f"{hire_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        job_title = job_titles[dept_id][0]  # First title is manager
        salary = random.randint(80000, 120000)

        employees_data.append((employee_id, first, last, email, phone, hire_date, job_title, dept_id, salary, None, 'active'))
        managers.append(employee_id)
        employee_id += 1

    # Create regular employees
    for dept_id in range(1, 7):
        num_employees = random.randint(8, 15)  # 8-15 employees per department
        manager_id = managers[dept_id - 1]

        for _ in range(num_employees):
            first = random.choice(first_names)
            last = random.choice(last_names)
            email = f"{first.lower()}.{last.lower()}{employee_id}@company.com"
            phone = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

            hire_year = random.randint(2018, 2024)
            hire_date = f"{hire_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

            job_title = random.choice(job_titles[dept_id][1:])  # Exclude manager title

            # Salary based on department and seniority
            base_salary = {1: 55000, 2: 75000, 3: 60000, 4: 65000, 5: 70000, 6: 60000}[dept_id]
            years_experience = 2024 - hire_year
            salary = base_salary + (years_experience * 3000) + random.randint(-5000, 10000)

            status = random.choices(['active', 'inactive'], weights=[95, 5])[0]

            employees_data.append((employee_id, first, last, email, phone, hire_date, job_title, dept_id, salary, manager_id, status))
            employee_id += 1

    cursor.executemany('''
                       INSERT INTO employees (employee_id, first_name, last_name, email, phone, hire_date, job_title, dept_id, salary, manager_id, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', employees_data)

    # Update departments with manager IDs
    for i, manager_id in enumerate(managers, 1):
        cursor.execute('UPDATE departments SET manager_id = ? WHERE dept_id = ?', (manager_id, i))

    # Projects
    project_names = [
        'Customer Portal Redesign', 'Mobile App Development', 'Data Analytics Platform',
        'Marketing Campaign 2024', 'HR System Upgrade', 'Sales CRM Integration',
        'Website Optimization', 'Security Audit Project', 'Cloud Migration',
        'Product Launch Campaign', 'Employee Training Platform', 'Financial Reporting System'
    ]

    projects_data = []
    for i, project_name in enumerate(project_names, 1):
        description = f"Strategic project for {project_name.lower()} to improve business operations"

        start_year = random.randint(2023, 2024)
        start_month = random.randint(1, 12)
        start_date = f"{start_year}-{start_month:02d}-01"

        # End date 3-12 months after start
        end_month = start_month + random.randint(3, 12)
        end_year = start_year
        if end_month > 12:
            end_month -= 12
            end_year += 1
        end_date = f"{end_year}-{end_month:02d}-{random.randint(15, 28):02d}"

        budget = random.randint(50000, 500000)
        status = random.choices(['planning', 'active', 'completed'], weights=[20, 60, 20])[0]
        manager_id = random.choice(managers)

        projects_data.append((i, project_name, description, start_date, end_date, budget, status, manager_id))

    cursor.executemany('''
                       INSERT INTO projects (project_id, project_name, description, start_date, end_date, budget, status, manager_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', projects_data)

    # Project assignments
    assignments_data = []
    assignment_id = 1

    for project_id in range(1, 13):
        # Each project has 3-8 team members
        num_assignments = random.randint(3, 8)

        # Select random employees (excluding managers for variety)
        available_employees = [emp[0] for emp in employees_data if emp[0] not in managers]
        assigned_employees = random.sample(available_employees, min(num_assignments, len(available_employees)))

        roles = ['Developer', 'Analyst', 'Designer', 'Tester', 'Coordinator', 'Specialist']

        for emp_id in assigned_employees:
            role = random.choice(roles)
            hours_allocated = random.randint(20, 40)

            # Assignment dates within project timeline
            proj_start = projects_data[project_id - 1][3]  # start_date from projects
            proj_end = projects_data[project_id - 1][4]    # end_date from projects

            assignments_data.append((assignment_id, project_id, emp_id, role, hours_allocated, proj_start, proj_end))
            assignment_id += 1

    cursor.executemany('''
                       INSERT INTO project_assignments (assignment_id, project_id, employee_id, role, hours_allocated, start_date, end_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ''', assignments_data)

    # Time tracking entries
    time_data = []
    time_id = 1

    # Generate time entries for active projects
    for assignment in assignments_data:
        project_id = assignment[1]
        employee_id = assignment[2]

        # Generate 10-30 time entries per assignment
        num_entries = random.randint(10, 30)

        for _ in range(num_entries):
            # Random work date within last 6 months
            days_ago = random.randint(0, 180)
            work_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

            hours_worked = round(random.uniform(1, 8), 1)

            descriptions = [
                'Code development and testing',
                'Meeting with stakeholders',
                'Bug fixes and optimization',
                'Documentation update',
                'Design review and feedback',
                'Research and analysis',
                'Client communication',
                'Team collaboration'
            ]
            description = random.choice(descriptions)

            time_data.append((time_id, employee_id, project_id, work_date, hours_worked, description))
            time_id += 1

    cursor.executemany('''
                       INSERT INTO time_tracking (time_id, employee_id, project_id, work_date, hours_worked, description)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', time_data)

    # Create indexes
    cursor.execute('CREATE INDEX idx_employees_dept ON employees(dept_id)')
    cursor.execute('CREATE INDEX idx_employees_manager ON employees(manager_id)')
    cursor.execute('CREATE INDEX idx_employees_email ON employees(email)')
    cursor.execute('CREATE INDEX idx_projects_manager ON projects(manager_id)')
    cursor.execute('CREATE INDEX idx_assignments_project ON project_assignments(project_id)')
    cursor.execute('CREATE INDEX idx_assignments_employee ON project_assignments(employee_id)')
    cursor.execute('CREATE INDEX idx_time_employee ON time_tracking(employee_id)')
    cursor.execute('CREATE INDEX idx_time_project ON time_tracking(project_id)')
    cursor.execute('CREATE INDEX idx_time_date ON time_tracking(work_date)')

    # Create views
    cursor.execute('''
                   CREATE VIEW employee_summary AS
                   SELECT
                       e.employee_id,
                       e.first_name || ' ' || e.last_name AS full_name,
                       e.job_title,
                       d.dept_name,
                       e.salary,
                       m.first_name || ' ' || m.last_name AS manager_name,
                       e.hire_date,
                       e.status
                   FROM employees e
                            LEFT JOIN departments d ON e.dept_id = d.dept_id
                            LEFT JOIN employees m ON e.manager_id = m.employee_id
                   ''')

    cursor.execute('''
                   CREATE VIEW project_progress AS
                   SELECT
                       p.project_id,
                       p.project_name,
                       p.status,
                       p.budget,
                       COUNT(pa.employee_id) as team_size,
                       SUM(tt.hours_worked) as total_hours_logged,
                       AVG(tt.hours_worked) as avg_daily_hours
                   FROM projects p
                            LEFT JOIN project_assignments pa ON p.project_id = pa.project_id
                            LEFT JOIN time_tracking tt ON p.project_id = tt.project_id
                   GROUP BY p.project_id, p.project_name, p.status, p.budget
                   ''')

    # Create triggers
    cursor.execute('''
                   CREATE TRIGGER update_employee_status
                       AFTER UPDATE OF status ON employees
                       WHEN NEW.status = 'terminated'
                   BEGIN
                       -- Remove from active project assignments
                       UPDATE project_assignments
                       SET end_date = DATE('now')
                       WHERE employee_id = NEW.employee_id
                         AND end_date > DATE('now');
                   END
                   ''')

    cursor.execute('''
                   CREATE TRIGGER validate_time_entry
                       BEFORE INSERT ON time_tracking
                       WHEN NEW.hours_worked > 12
                   BEGIN
                       SELECT RAISE(ABORT, 'Hours worked cannot exceed 12 hours per day');
                   END
                   ''')

    conn.commit()
    conn.close()
    print("✅ company_database.db created with 6 departments, 60+ employees, 12 projects, and time tracking")

def create_library_database():
    """Create a library management database"""
    print("📚 Creating library_database.db...")

    conn = sqlite3.connect('library_database.db')
    cursor = conn.cursor()

    # Authors table
    cursor.execute('''
                   CREATE TABLE authors (
                                            author_id INTEGER PRIMARY KEY,
                                            first_name TEXT NOT NULL,
                                            last_name TEXT NOT NULL,
                                            birth_date DATE,
                                            nationality TEXT,
                                            biography TEXT
                   )
                   ''')

    # Categories table
    cursor.execute('''
                   CREATE TABLE categories (
                                               category_id INTEGER PRIMARY KEY,
                                               category_name TEXT NOT NULL UNIQUE,
                                               description TEXT
                   )
                   ''')

    # Books table
    cursor.execute('''
                   CREATE TABLE books (
                                          book_id INTEGER PRIMARY KEY,
                                          title TEXT NOT NULL,
                                          isbn TEXT UNIQUE,
                                          author_id INTEGER,
                                          category_id INTEGER,
                                          publication_year INTEGER,
                                          publisher TEXT,
                                          pages INTEGER,
                                          language TEXT DEFAULT 'English',
                                          copies_total INTEGER DEFAULT 1,
                                          copies_available INTEGER DEFAULT 1,
                                          FOREIGN KEY (author_id) REFERENCES authors(author_id),
                                          FOREIGN KEY (category_id) REFERENCES categories(category_id)
                   )
                   ''')

    # Members table
    cursor.execute('''
                   CREATE TABLE members (
                                            member_id INTEGER PRIMARY KEY,
                                            first_name TEXT NOT NULL,
                                            last_name TEXT NOT NULL,
                                            email TEXT UNIQUE,
                                            phone TEXT,
                                            address TEXT,
                                            membership_date DATE DEFAULT CURRENT_DATE,
                                            membership_type TEXT DEFAULT 'standard' CHECK (membership_type IN ('standard', 'premium', 'student')),
                                            status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'expired'))
                   )
                   ''')

    # Loans table
    cursor.execute('''
                   CREATE TABLE loans (
                                          loan_id INTEGER PRIMARY KEY,
                                          book_id INTEGER,
                                          member_id INTEGER,
                                          loan_date DATE DEFAULT CURRENT_DATE,
                                          due_date DATE,
                                          return_date DATE,
                                          fine_amount REAL DEFAULT 0,
                                          status TEXT DEFAULT 'active' CHECK (status IN ('active', 'returned', 'overdue')),
                                          FOREIGN KEY (book_id) REFERENCES books(book_id),
                                          FOREIGN KEY (member_id) REFERENCES members(member_id)
                   )
                   ''')

    # Insert sample data

    # Categories
    categories = [
        (1, 'Fiction', 'Novels and fictional stories'),
        (2, 'Non-Fiction', 'Factual and educational books'),
        (3, 'Science', 'Scientific research and discoveries'),
        (4, 'History', 'Historical events and biographies'),
        (5, 'Technology', 'Computer science and technology'),
        (6, 'Biography', 'Life stories of notable people'),
        (7, 'Mystery', 'Mystery and detective stories'),
        (8, 'Romance', 'Romantic fiction'),
        (9, 'Fantasy', 'Fantasy and magical stories'),
        (10, 'Self-Help', 'Personal development and improvement')
    ]

    cursor.executemany('''
                       INSERT INTO categories (category_id, category_name, description)
                       VALUES (?, ?, ?)
                       ''', categories)

    # Authors
    famous_authors = [
        ('Agatha', 'Christie', '1890-09-15', 'British', 'Famous mystery writer'),
        ('Stephen', 'King', '1947-09-21', 'American', 'Horror and suspense novelist'),
        ('J.K.', 'Rowling', '1965-07-31', 'British', 'Author of Harry Potter series'),
        ('George', 'Orwell', '1903-06-25', 'British', 'Author of 1984 and Animal Farm'),
        ('Jane', 'Austen', '1775-12-16', 'British', 'Regency romance novelist'),
        ('Mark', 'Twain', '1835-11-30', 'American', 'Author of Tom Sawyer and Huckleberry Finn'),
        ('Ernest', 'Hemingway', '1899-07-21', 'American', 'Nobel Prize winning author'),
        ('Virginia', 'Woolf', '1882-01-25', 'British', 'Modernist writer'),
        ('Charles', 'Dickens', '1812-02-07', 'British', 'Victorian era novelist'),
        ('Maya', 'Angelou', '1928-04-04', 'American', 'Poet and civil rights activist'),
        ('Isaac', 'Asimov', '1920-01-02', 'American', 'Science fiction writer'),
        ('Toni', 'Morrison', '1931-02-18', 'American', 'Nobel Prize winning novelist'),
        ('Gabriel', 'García Márquez', '1927-03-06', 'Colombian', 'Magical realism author'),
        ('Harper', 'Lee', '1926-04-28', 'American', 'Author of To Kill a Mockingbird'),
        ('F. Scott', 'Fitzgerald', '1896-09-24', 'American', 'Author of The Great Gatsby')
    ]

    authors_data = []
    for i, (first, last, birth, nationality, bio) in enumerate(famous_authors, 1):
        authors_data.append((i, first, last, birth, nationality, bio))

    cursor.executemany('''
                       INSERT INTO authors (author_id, first_name, last_name, birth_date, nationality, biography)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', authors_data)

    # Books
    books_data = [
        ('Murder on the Orient Express', '978-0-00-712279-7', 1, 7, 1934, 'Collins Crime Club', 256),
        ('The Shining', '978-0-385-12167-5', 2, 7, 1977, 'Doubleday', 447),
        ('Harry Potter and the Philosopher\'s Stone', '978-0-7475-3269-9', 3, 9, 1997, 'Bloomsbury', 223),
        ('1984', '978-0-452-28423-4', 4, 1, 1949, 'Secker & Warburg', 328),
        ('Pride and Prejudice', '978-0-14-143951-8', 5, 8, 1813, 'T. Egerton', 432),
        ('The Adventures of Tom Sawyer', '978-0-486-40077-4', 6, 1, 1876, 'American Publishing Company', 274),
        ('The Old Man and the Sea', '978-0-684-80122-3', 7, 1, 1952, 'Charles Scribner\'s Sons', 127),
        ('To the Lighthouse', '978-0-15-690739-6', 8, 1, 1927, 'Hogarth Press', 209),
        ('Great Expectations', '978-0-14-143956-3', 9, 1, 1861, 'Chapman & Hall', 544),
        ('I Know Why the Caged Bird Sings', '978-0-345-51440-8', 10, 6, 1969, 'Random House', 289),
        ('Foundation', '978-0-553-29335-0', 11, 3, 1951, 'Gnome Press', 244),
        ('Beloved', '978-1-4000-3341-6', 12, 1, 1987, 'Alfred A. Knopf', 321),
        ('One Hundred Years of Solitude', '978-0-06-088328-7', 13, 1, 1967, 'Harper & Row', 417),
        ('To Kill a Mockingbird', '978-0-06-112008-4', 14, 1, 1960, 'J. B. Lippincott & Co.', 281),
        ('The Great Gatsby', '978-0-7432-7356-5', 15, 1, 1925, 'Charles Scribner\'s Sons', 180)
    ]

    books_insert_data = []
    for i, (title, isbn, author_id, category_id, year, publisher, pages) in enumerate(books_data, 1):
        copies_total = random.randint(2, 8)
        copies_available = random.randint(0, copies_total)
        books_insert_data.append((i, title, isbn, author_id, category_id, year, publisher, pages, 'English', copies_total, copies_available))

    cursor.executemany('''
                       INSERT INTO books (book_id, title, isbn, author_id, category_id, publication_year, publisher, pages, language, copies_total, copies_available)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', books_insert_data)

    # Members
    member_first_names = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack']
    member_last_names = ['Adams', 'Brown', 'Clark', 'Davis', 'Evans', 'Garcia', 'Harris', 'Johnson', 'King', 'Lewis']

    members_data = []
    for i in range(1, 101):  # 100 members
        first = random.choice(member_first_names)
        last = random.choice(member_last_names)
        email = f"{first.lower()}.{last.lower()}{i}@email.com"
        phone = f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        address = f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Pine Rd', 'Elm St'])}"

        membership_year = random.randint(2020, 2024)
        membership_date = f"{membership_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

        membership_type = random.choices(['standard', 'premium', 'student'], weights=[60, 25, 15])[0]
        status = random.choices(['active', 'suspended', 'expired'], weights=[85, 5, 10])[0]

        members_data.append((i, first, last, email, phone, address, membership_date, membership_type, status))

    cursor.executemany('''
                       INSERT INTO members (member_id, first_name, last_name, email, phone, address, membership_date, membership_type, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', members_data)

    # Loans
    loans_data = []
    for loan_id in range(1, 251):  # 250 loans
        book_id = random.randint(1, 15)
        member_id = random.randint(1, 100)

        # Random loan date within last year
        days_ago = random.randint(0, 365)
        loan_date = datetime.now() - timedelta(days=days_ago)

        # Due date is typically 14 days after loan date
        due_date = loan_date + timedelta(days=14)

        # Some books are returned, some are still out
        is_returned = random.choice([True, False])

        if is_returned:
            # Return date between loan date and now
            return_days = random.randint(1, min(30, days_ago))
            return_date = loan_date + timedelta(days=return_days)

            # Calculate fine for overdue books
            if return_date > due_date:
                overdue_days = (return_date - due_date).days
                fine_amount = overdue_days * 0.50  # $0.50 per day fine
                status = 'returned'
            else:
                fine_amount = 0
                status = 'returned'
        else:
            return_date = None
            # Check if currently overdue
            if datetime.now().date() > due_date.date():
                overdue_days = (datetime.now().date() - due_date.date()).days
                fine_amount = overdue_days * 0.50
                status = 'overdue'
            else:
                fine_amount = 0
                status = 'active'

        loans_data.append((
            loan_id, book_id, member_id,
            loan_date.strftime('%Y-%m-%d'),
            due_date.strftime('%Y-%m-%d'),
            return_date.strftime('%Y-%m-%d') if return_date else None,
            fine_amount, status
        ))

    cursor.executemany('''
                       INSERT INTO loans (loan_id, book_id, member_id, loan_date, due_date, return_date, fine_amount, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', loans_data)

    # Create indexes
    cursor.execute('CREATE INDEX idx_books_author ON books(author_id)')
    cursor.execute('CREATE INDEX idx_books_category ON books(category_id)')
    cursor.execute('CREATE INDEX idx_books_isbn ON books(isbn)')
    cursor.execute('CREATE INDEX idx_members_email ON members(email)')
    cursor.execute('CREATE INDEX idx_loans_book ON loans(book_id)')
    cursor.execute('CREATE INDEX idx_loans_member ON loans(member_id)')
    cursor.execute('CREATE INDEX idx_loans_status ON loans(status)')
    cursor.execute('CREATE INDEX idx_loans_due_date ON loans(due_date)')

    # Create views
    cursor.execute('''
                   CREATE VIEW book_catalog AS
                   SELECT
                       b.book_id,
                       b.title,
                       a.first_name || ' ' || a.last_name AS author_name,
                       c.category_name,
                       b.publication_year,
                       b.publisher,
                       b.copies_total,
                       b.copies_available,
                       CASE WHEN b.copies_available > 0 THEN 'Available' ELSE 'Checked Out' END AS availability
                   FROM books b
                            JOIN authors a ON b.author_id = a.author_id
                            JOIN categories c ON b.category_id = c.category_id
                   ''')

    cursor.execute('''
                   CREATE VIEW overdue_books AS
                   SELECT
                       l.loan_id,
                       b.title,
                       m.first_name || ' ' || m.last_name AS member_name,
                       m.email,
                       l.loan_date,
                       l.due_date,
                       l.fine_amount,
                           DATE('now') AS current_date,
                           julianday('now') - julianday(l.due_date) AS days_overdue
                           FROM loans l
                           JOIN books b ON l.book_id = b.book_id
                           JOIN members m ON l.member_id = m.member_id
                           WHERE l.status = 'overdue'
                   ''')

    # Create triggers
    cursor.execute('''
                   CREATE TRIGGER update_book_availability_on_loan
                       AFTER INSERT ON loans
                   BEGIN
                       UPDATE books
                       SET copies_available = copies_available - 1
                       WHERE book_id = NEW.book_id AND copies_available > 0;
                   END
                   ''')

    cursor.execute('''
                   CREATE TRIGGER update_book_availability_on_return
                       AFTER UPDATE OF return_date ON loans
                       WHEN NEW.return_date IS NOT NULL AND OLD.return_date IS NULL
                   BEGIN
                       UPDATE books
                       SET copies_available = copies_available + 1
                       WHERE book_id = NEW.book_id;

                       UPDATE loans
                       SET status = 'returned'
                       WHERE loan_id = NEW.loan_id;
                   END
                   ''')

    cursor.execute('''
                   CREATE TRIGGER calculate_fine_on_return
                       AFTER UPDATE OF return_date ON loans
                       WHEN NEW.return_date IS NOT NULL AND OLD.return_date IS NULL
                   BEGIN
                       UPDATE loans
                       SET fine_amount = CASE
                                             WHEN julianday(NEW.return_date) > julianday(NEW.due_date)
                                                 THEN (julianday(NEW.return_date) - julianday(NEW.due_date)) * 0.50
                                             ELSE 0
                           END
                       WHERE loan_id = NEW.loan_id;
                   END
                   ''')

    conn.commit()
    conn.close()
    print("✅ library_database.db created with 15 books, 100 members, 250 loans, and library management features")

def create_inventory_database():
    """Create an inventory and warehouse management database"""
    print("📦 Creating inventory_database.db...")

    conn = sqlite3.connect('inventory_database.db')
    cursor = conn.cursor()

    # Suppliers table
    cursor.execute('''
                   CREATE TABLE suppliers (
                                              supplier_id INTEGER PRIMARY KEY,
                                              supplier_name TEXT NOT NULL,
                                              contact_person TEXT,
                                              email TEXT,
                                              phone TEXT,
                                              address TEXT,
                                              city TEXT,
                                              country TEXT,
                                              payment_terms TEXT,
                                              rating INTEGER CHECK (rating >= 1 AND rating <= 5)
                   )
                   ''')

    # Warehouses table
    cursor.execute('''
                   CREATE TABLE warehouses (
                                               warehouse_id INTEGER PRIMARY KEY,
                                               warehouse_name TEXT NOT NULL,
                                               location TEXT,
                                               capacity INTEGER,
                                               manager_name TEXT,
                                               phone TEXT
                   )
                   ''')

    # Product categories
    cursor.execute('''
                   CREATE TABLE product_categories (
                                                       category_id INTEGER PRIMARY KEY,
                                                       category_name TEXT NOT NULL UNIQUE,
                                                       description TEXT,
                                                       parent_category_id INTEGER,
                                                       FOREIGN KEY (parent_category_id) REFERENCES product_categories(category_id)
                   )
                   ''')

    # Products table
    cursor.execute('''
                   CREATE TABLE products (
                                             product_id INTEGER PRIMARY KEY,
                                             product_code TEXT UNIQUE NOT NULL,
                                             product_name TEXT NOT NULL,
                                             description TEXT,
                                             category_id INTEGER,
                                             unit_price REAL CHECK (unit_price >= 0),
                                             cost_price REAL CHECK (cost_price >= 0),
                                             weight REAL,
                                             dimensions TEXT,
                                             supplier_id INTEGER,
                                             reorder_point INTEGER DEFAULT 10,
                                             max_stock_level INTEGER DEFAULT 1000,
                                             is_active BOOLEAN DEFAULT 1,
                                             FOREIGN KEY (category_id) REFERENCES product_categories(category_id),
                                             FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
                   )
                   ''')

    # Inventory table
    cursor.execute('''
                   CREATE TABLE inventory (
                                              inventory_id INTEGER PRIMARY KEY,
                                              product_id INTEGER,
                                              warehouse_id INTEGER,
                                              quantity_on_hand INTEGER DEFAULT 0,
                                              quantity_reserved INTEGER DEFAULT 0,
                                              quantity_available INTEGER GENERATED ALWAYS AS (quantity_on_hand - quantity_reserved) STORED,
                                              last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                              FOREIGN KEY (product_id) REFERENCES products(product_id),
                                              FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
                                              UNIQUE(product_id, warehouse_id)
                   )
                   ''')

    # Stock movements table
    cursor.execute('''
                   CREATE TABLE stock_movements (
                                                    movement_id INTEGER PRIMARY KEY,
                                                    product_id INTEGER,
                                                    warehouse_id INTEGER,
                                                    movement_type TEXT CHECK (movement_type IN ('IN', 'OUT', 'TRANSFER', 'ADJUSTMENT')),
                                                    quantity INTEGER NOT NULL,
                                                    reference_number TEXT,
                                                    notes TEXT,
                                                    movement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                    created_by TEXT,
                                                    FOREIGN KEY (product_id) REFERENCES products(product_id),
                                                    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
                   )
                   ''')

    # Purchase orders table
    cursor.execute('''
                   CREATE TABLE purchase_orders (
                                                    po_id INTEGER PRIMARY KEY,
                                                    po_number TEXT UNIQUE NOT NULL,
                                                    supplier_id INTEGER,
                                                    order_date DATE DEFAULT CURRENT_DATE,
                                                    expected_delivery_date DATE,
                                                    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'sent', 'received', 'cancelled')),
                                                    total_amount REAL DEFAULT 0,
                                                    notes TEXT,
                                                    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
                   )
                   ''')

    # Purchase order items
    cursor.execute('''
                   CREATE TABLE purchase_order_items (
                                                         po_item_id INTEGER PRIMARY KEY,
                                                         po_id INTEGER,
                                                         product_id INTEGER,
                                                         quantity_ordered INTEGER,
                                                         quantity_received INTEGER DEFAULT 0,
                                                         unit_price REAL,
                                                         line_total REAL GENERATED ALWAYS AS (quantity_ordered * unit_price) STORED,
                                                         FOREIGN KEY (po_id) REFERENCES purchase_orders(po_id),
                                                         FOREIGN KEY (product_id) REFERENCES products(product_id)
                   )
                   ''')

    # Insert sample data

    # Suppliers
    suppliers_data = [
        (1, 'TechSupply Corp', 'John Anderson', 'john@techsupply.com', '555-0001', '123 Tech St', 'San Francisco', 'USA', 'Net 30', 4),
        (2, 'Global Electronics', 'Maria Rodriguez', 'maria@globalelec.com', '555-0002', '456 Circuit Ave', 'Austin', 'USA', 'Net 15', 5),
        (3, 'Office Supplies Plus', 'David Kim', 'david@officesupply.com', '555-0003', '789 Business Blvd', 'Chicago', 'USA', 'Net 45', 3),
        (4, 'Industrial Components', 'Sarah Johnson', 'sarah@indcomp.com', '555-0004', '321 Factory Rd', 'Detroit', 'USA', 'Net 30', 4),
        (5, 'International Trading Co', 'Wei Zhang', 'wei@intltrade.com', '555-0005', '654 Import St', 'Los Angeles', 'USA', 'Net 60', 3)
    ]

    cursor.executemany('''
                       INSERT INTO suppliers (supplier_id, supplier_name, contact_person, email, phone, address, city, country, payment_terms, rating)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', suppliers_data)

    # Warehouses
    warehouses_data = [
        (1, 'Main Warehouse', 'Downtown District', 50000, 'Mike Wilson', '555-1001'),
        (2, 'North Distribution Center', 'North Industrial Park', 75000, 'Lisa Chen', '555-1002'),
        (3, 'South Facility', 'South Commerce Zone', 30000, 'Robert Taylor', '555-1003'),
        (4, 'East Coast Hub', 'Port Authority Area', 100000, 'Jennifer Davis', '555-1004')
    ]

    cursor.executemany('''
                       INSERT INTO warehouses (warehouse_id, warehouse_name, location, capacity, manager_name, phone)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', warehouses_data)

    # Product categories
    categories_data = [
        (1, 'Electronics', 'Electronic devices and components', None),
        (2, 'Computers', 'Computer hardware and accessories', 1),
        (3, 'Mobile Devices', 'Smartphones, tablets, and accessories', 1),
        (4, 'Office Supplies', 'General office equipment and supplies', None),
        (5, 'Furniture', 'Office and warehouse furniture', 4),
        (6, 'Stationery', 'Paper, pens, and writing materials', 4),
        (7, 'Industrial', 'Industrial equipment and tools', None),
        (8, 'Safety Equipment', 'Safety gear and protective equipment', 7),
        (9, 'Tools', 'Hand tools and power tools', 7),
        (10, 'Consumables', 'Consumable items and supplies', None)
    ]

    cursor.executemany('''
                       INSERT INTO product_categories (category_id, category_name, description, parent_category_id)
                       VALUES (?, ?, ?, ?)
                       ''', categories_data)

    # Products
    products_data = [
        ('LAPTOP001', 'Business Laptop 15"', 'High-performance laptop for business use', 2, 899.99, 650.00, 2.1, '15x10x0.8 inches', 1, 5, 50),
        ('PHONE001', 'Smartphone Pro', 'Latest smartphone with advanced features', 3, 799.99, 500.00, 0.2, '6x3x0.3 inches', 2, 10, 100),
        ('DESK001', 'Executive Desk', 'Large wooden executive desk', 5, 599.99, 350.00, 45.0, '60x30x30 inches', 3, 2, 20),
        ('CHAIR001', 'Office Chair Deluxe', 'Ergonomic office chair with lumbar support', 5, 299.99, 180.00, 15.5, '26x26x45 inches', 3, 5, 50),
        ('TABLET001', 'Business Tablet', '10-inch tablet for business applications', 3, 399.99, 250.00, 0.5, '10x7x0.3 inches', 2, 8, 75),
        ('PRINTER001', 'Laser Printer', 'High-speed laser printer', 2, 249.99, 150.00, 12.0, '16x14x10 inches', 1, 3, 30),
        ('PAPER001', 'Copy Paper A4', 'White copy paper 500 sheets', 6, 4.99, 2.50, 2.5, '11.7x8.3x2 inches', 3, 100, 5000),
        ('HELMET001', 'Safety Helmet', 'Industrial safety helmet', 8, 29.99, 15.00, 0.4, '12x10x6 inches', 4, 50, 500),
        ('DRILL001', 'Cordless Drill', 'Professional cordless drill with battery', 9, 89.99, 55.00, 1.8, '10x8x3 inches', 4, 10, 100),
        ('CABLE001', 'Ethernet Cable 6ft', 'Cat6 ethernet cable', 2, 9.99, 4.00, 0.2, '72x1x1 inches', 2, 200, 2000),
        ('PEN001', 'Ballpoint Pen Blue', 'Blue ink ballpoint pen', 6, 0.99, 0.30, 0.01, '6x0.5x0.5 inches', 3, 1000, 10000),
        ('MOUSE001', 'Wireless Mouse', 'Optical wireless mouse', 2, 24.99, 12.00, 0.1, '4x2.5x1.5 inches', 1, 50, 500),
        ('MONITOR001', '24" Monitor', '24-inch LED monitor', 2, 179.99, 110.00, 4.2, '21x14x8 inches', 1, 8, 80),
        ('GLOVES001', 'Work Gloves', 'Industrial work gloves pair', 8, 12.99, 6.00, 0.2, '10x4x1 inches', 4, 100, 1000),
        ('BINDER001', '3-Ring Binder', 'Standard 3-ring binder', 6, 3.99, 1.50, 0.8, '11x12x2 inches', 3, 200, 2000)
    ]

    products_insert_data = []
    for i, (code, name, desc, cat_id, price, cost, weight, dims, supp_id, reorder, max_stock) in enumerate(products_data, 1):
        products_insert_data.append((i, code, name, desc, cat_id, price, cost, weight, dims, supp_id, reorder, max_stock, 1))

    cursor.executemany('''
                       INSERT INTO products (product_id, product_code, product_name, description, category_id, unit_price, cost_price, weight, dimensions, supplier_id, reorder_point, max_stock_level, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', products_insert_data)

    # Inventory
    inventory_data = []
    inventory_id = 1

    for product_id in range(1, 16):  # For each product
        for warehouse_id in range(1, 5):  # In each warehouse
            # Random quantities
            on_hand = random.randint(0, 200)
            reserved = random.randint(0, min(on_hand, 50))

            inventory_data.append((inventory_id, product_id, warehouse_id, on_hand, reserved))
            inventory_id += 1

    cursor.executemany('''
                       INSERT INTO inventory (inventory_id, product_id, warehouse_id, quantity_on_hand, quantity_reserved)
                       VALUES (?, ?, ?, ?, ?)
                       ''', inventory_data)

    # Stock movements
    movements_data = []
    movement_types = ['IN', 'OUT', 'TRANSFER', 'ADJUSTMENT']
    users = ['admin', 'warehouse1', 'warehouse2', 'manager']

    for movement_id in range(1, 201):  # 200 movements
        product_id = random.randint(1, 15)
        warehouse_id = random.randint(1, 4)
        movement_type = random.choice(movement_types)

        if movement_type == 'IN':
            quantity = random.randint(10, 100)
        elif movement_type == 'OUT':
            quantity = -random.randint(1, 50)
        elif movement_type == 'TRANSFER':
            quantity = random.choice([random.randint(1, 30), -random.randint(1, 30)])
        else:  # ADJUSTMENT
            quantity = random.randint(-20, 20)

        ref_number = f"REF{movement_id:06d}"
        notes = f"Stock {movement_type.lower()} for product {product_id}"

        # Random date within last 6 months
        days_ago = random.randint(0, 180)
        movement_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')

        created_by = random.choice(users)

        movements_data.append((movement_id, product_id, warehouse_id, movement_type, quantity, ref_number, notes, movement_date, created_by))

    cursor.executemany('''
                       INSERT INTO stock_movements (movement_id, product_id, warehouse_id, movement_type, quantity, reference_number, notes, movement_date, created_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', movements_data)

    # Purchase orders
    po_data = []
    for po_id in range(1, 26):  # 25 purchase orders
        po_number = f"PO{po_id:06d}"
        supplier_id = random.randint(1, 5)

        # Random order date within last year
        days_ago = random.randint(0, 365)
        order_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        # Expected delivery 7-30 days after order
        delivery_days = random.randint(7, 30)
        expected_delivery = (datetime.now() - timedelta(days=days_ago) + timedelta(days=delivery_days)).strftime('%Y-%m-%d')

        status = random.choices(['pending', 'approved', 'sent', 'received', 'cancelled'], weights=[10, 15, 20, 50, 5])[0]
        notes = f"Purchase order for supplier {supplier_id}"

        po_data.append((po_id, po_number, supplier_id, order_date, expected_delivery, status, 0, notes))

    cursor.executemany('''
                       INSERT INTO purchase_orders (po_id, po_number, supplier_id, order_date, expected_delivery_date, status, total_amount, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', po_data)

    # Purchase order items
    po_items_data = []
    po_item_id = 1

    for po_id in range(1, 26):
        # Each PO has 1-5 items
        num_items = random.randint(1, 5)
        po_products = random.sample(range(1, 16), num_items)

        po_total = 0

        for product_id in po_products:
            quantity_ordered = random.randint(10, 100)

            # Get product cost price (simplified - using index)
            unit_price = products_data[product_id - 1][5]  # cost_price from products_data

            # Some items might be partially received
            if random.choice([True, False]):
                quantity_received = random.randint(0, quantity_ordered)
            else:
                quantity_received = 0

            po_items_data.append((po_item_id, po_id, product_id, quantity_ordered, quantity_received, unit_price))
            po_total += quantity_ordered * unit_price
            po_item_id += 1

        # Update PO total
        cursor.execute('UPDATE purchase_orders SET total_amount = ? WHERE po_id = ?', (po_total, po_id))

    cursor.executemany('''
                       INSERT INTO purchase_order_items (po_item_id, po_id, product_id, quantity_ordered, quantity_received, unit_price)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', po_items_data)

    # Create indexes
    cursor.execute('CREATE INDEX idx_products_code ON products(product_code)')
    cursor.execute('CREATE INDEX idx_products_category ON products(category_id)')
    cursor.execute('CREATE INDEX idx_products_supplier ON products(supplier_id)')
    cursor.execute('CREATE INDEX idx_inventory_product ON inventory(product_id)')
    cursor.execute('CREATE INDEX idx_inventory_warehouse ON inventory(warehouse_id)')
    cursor.execute('CREATE INDEX idx_movements_product ON stock_movements(product_id)')
    cursor.execute('CREATE INDEX idx_movements_date ON stock_movements(movement_date)')
    cursor.execute('CREATE INDEX idx_po_supplier ON purchase_orders(supplier_id)')
    cursor.execute('CREATE INDEX idx_po_items_po ON purchase_order_items(po_id)')
    cursor.execute('CREATE INDEX idx_po_items_product ON purchase_order_items(product_id)')

    # Create views
    cursor.execute('''
                   CREATE VIEW inventory_summary AS
                   SELECT
                       p.product_code,
                       p.product_name,
                       w.warehouse_name,
                       i.quantity_on_hand,
                       i.quantity_reserved,
                       i.quantity_available,
                       p.reorder_point,
                       CASE
                           WHEN i.quantity_available <= p.reorder_point THEN 'Low Stock'
                           WHEN i.quantity_available = 0 THEN 'Out of Stock'
                           ELSE 'In Stock'
                           END as stock_status
                   FROM inventory i
                            JOIN products p ON i.product_id = p.product_id
                            JOIN warehouses w ON i.warehouse_id = w.warehouse_id
                   ''')

    cursor.execute('''
                   CREATE VIEW low_stock_alert AS
                   SELECT
                       p.product_code,
                       p.product_name,
                       SUM(i.quantity_available) as total_available,
                       p.reorder_point,
                       s.supplier_name
                   FROM products p
                            JOIN inventory i ON p.product_id = i.product_id
                            JOIN suppliers s ON p.supplier_id = s.supplier_id
                   GROUP BY p.product_id, p.product_code, p.product_name, p.reorder_point, s.supplier_name
                   HAVING SUM(i.quantity_available) <= p.reorder_point
                   ''')

    # Create triggers
    cursor.execute('''
                   CREATE TRIGGER update_inventory_on_movement
                       AFTER INSERT ON stock_movements
                   BEGIN
                       UPDATE inventory
                       SET quantity_on_hand = quantity_on_hand + NEW.quantity,
                           last_updated = CURRENT_TIMESTAMP
                       WHERE product_id = NEW.product_id AND warehouse_id = NEW.warehouse_id;
                   END
                   ''')

    cursor.execute('''
                   CREATE TRIGGER validate_stock_movement
                       BEFORE INSERT ON stock_movements
                       WHEN NEW.movement_type = 'OUT' AND NEW.quantity < 0
                   BEGIN
                       SELECT CASE
                                  WHEN (SELECT quantity_available FROM inventory
                                        WHERE product_id = NEW.product_id AND warehouse_id = NEW.warehouse_id) < ABS(NEW.quantity)
                                      THEN RAISE(ABORT, 'Insufficient stock for withdrawal')
                                  END;
                   END
                   ''')

    conn.commit()
    conn.close()
    print("✅ inventory_database.db created with 15 products, 4 warehouses, inventory tracking, and purchase orders")

def main():
    """Create all sample databases"""
    print("🚀 Creating Sample Databases for Universal Database Explorer")
    print("=" * 70)

    try:
        # Create all databases
        create_student_database()
        create_sales_database()
        create_company_database()
        create_library_database()
        create_inventory_database()

        print("\n" + "=" * 70)
        print("✅ ALL DATABASES CREATED SUCCESSFULLY!")
        print("=" * 70)

        print("\n📁 Created Database Files:")
        databases = [
            ('student_database.db', '📚 University management with students, courses, enrollments'),
            ('sales_database.db', '💰 E-commerce with customers, products, orders'),
            ('company_database.db', '🏢 HR and project management system'),
            ('library_database.db', '📖 Library management with books, members, loans'),
            ('inventory_database.db', '📦 Warehouse and inventory management')
        ]

        for db_file, description in databases:
            if os.path.exists(db_file):
                size_mb = os.path.getsize(db_file) / (1024 * 1024)
                print(f"  • {db_file:<25} ({size_mb:.2f} MB) - {description}")

        print(f"\n🛠️  Usage Examples:")
        print(f"  python db_explorer.py student_database.db")
        print(f"  python db_explorer.py sales_database.db")
        print(f"  python db_explorer.py company_database.db")
        print(f"  python db_explorer.py library_database.db")
        print(f"  python db_explorer.py inventory_database.db")

        print(f"\n💡 Each database contains:")
        print(f"  • Realistic sample data")
        print(f"  • Complex relationships and foreign keys")
        print(f"  • Indexes for performance")
        print(f"  • Views for common queries")
        print(f"  • Triggers for business logic")
        print(f"  • Various data types and constraints")

    except Exception as e:
        print(f"❌ Error creating databases: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())