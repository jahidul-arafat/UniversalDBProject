-- Enhanced Oracle HR Database - Sample Queries
-- Generated on: 2025-06-07 15:36:11

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
