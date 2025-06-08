// Database Explorer Frontend - JavaScript Implementation
// This module provides all client-side functionality calling the backend APIs

// Global variables
let isConnected = false;
let currentQueryResults = [];
let chatHistory = [];
let currentTables = [];

// API wrapper class that calls all backend endpoints
class DatabaseExplorerAPI {
    static async request(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    // Database connection APIs
    static async connectDatabase(formData) {
        const response = await fetch('/api/connect', {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }

    static async getDatabaseInfo() {
        return await this.request('/api/database-info');
    }

    static async getDatabaseStats() {
        return await this.request('/api/database-stats');
    }

    static async getApiStatus() {
        return await this.request('/api/status');
    }

    // Table management APIs
    static async getTables() {
        return await this.request('/api/tables');
    }

    static async getTableStructure(tableName) {
        return await this.request(`/api/table/${tableName}/structure`);
    }

    static async getTableData(tableName, limit = 100) {
        return await this.request(`/api/table/${tableName}/data?limit=${limit}`);
    }

    static async getTableRelationships(tableName) {
        return await this.request(`/api/table/${tableName}/relationships`);
    }

    static async analyzeTable(tableName, includeStats = true, includeDistribution = false) {
        return await this.request(`/api/table/${tableName}/analyze?include_stats=${includeStats}&include_distribution=${includeDistribution}`);
    }

    static async findDuplicates(tableName, columns = [], limit = 100) {
        return await this.request(`/api/table/${tableName}/duplicates`, {
            method: 'POST',
            body: JSON.stringify({ columns, limit })
        });
    }

    static async getNullAnalysis(tableName) {
        return await this.request(`/api/table/${tableName}/null-analysis`);
    }

    static async generateTableSQL(tableName) {
        return await this.request(`/api/table/${tableName}/generate-sql`);
    }

    // Query execution APIs
    static async executeQuery(query) {
        return await this.request('/api/execute-query', {
            method: 'POST',
            body: JSON.stringify({ query })
        });
    }

    static async searchTables(keyword) {
        return await this.request(`/api/search-tables?keyword=${encodeURIComponent(keyword)}`);
    }

    static async getSampleQueries() {
        return await this.request('/api/sample-queries');
    }

    // Join operations
    static async joinTables(table1, table2, joinCondition, columns = '*', whereClause = '', limit = 100, joinType = 'INNER') {
        return await this.request('/api/join-tables', {
            method: 'POST',
            body: JSON.stringify({
                table1,
                table2,
                join_condition: joinCondition,
                columns,
                where_clause: whereClause,
                limit,
                join_type: joinType
            })
        });
    }

    // Visualization APIs
    static async createVisualization(data, chartType, xColumn, yColumn, title) {
        return await this.request('/api/visualize', {
            method: 'POST',
            body: JSON.stringify({
                data,
                chart_type: chartType,
                x_column: xColumn,
                y_column: yColumn,
                title
            })
        });
    }

    // Export APIs
    static async exportCSV(data, filename) {
        const response = await fetch('/api/export-csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data, filename })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            return { success: true };
        } else {
            return await response.json();
        }
    }

    static async exportSchema() {
        const response = await fetch('/api/export-schema');
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'database_schema.sql';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            return { success: true };
        } else {
            return await response.json();
        }
    }

    // Validation and analysis APIs
    static async validateData() {
        return await this.request('/api/validate-data');
    }

    static async performanceAnalysis() {
        return await this.request('/api/performance-analysis');
    }

    // Schema management APIs
    static async createTable(tableName, columns) {
        return await this.request('/api/schema/create-table', {
            method: 'POST',
            body: JSON.stringify({
                table_name: tableName,
                columns
            })
        });
    }

    static async createIndex(indexName, tableName, columns, unique = false) {
        return await this.request('/api/schema/create-index', {
            method: 'POST',
            body: JSON.stringify({
                index_name: indexName,
                table_name: tableName,
                columns,
                unique
            })
        });
    }

    static async createView(viewName, selectQuery) {
        return await this.request('/api/schema/create-view', {
            method: 'POST',
            body: JSON.stringify({
                view_name: viewName,
                select_query: selectQuery
            })
        });
    }

    static async createTrigger(triggerName, tableName, event, triggerBody, whenCondition = '') {
        return await this.request('/api/schema/create-trigger', {
            method: 'POST',
            body: JSON.stringify({
                trigger_name: triggerName,
                table_name: tableName,
                event,
                trigger_body: triggerBody,
                when_condition: whenCondition
            })
        });
    }

    // Trigger management
    static async getTriggers() {
        return await this.request('/api/triggers');
    }

    static async getTriggerDetails(triggerName) {
        return await this.request(`/api/triggers/${triggerName}`);
    }

    static async dropTrigger(triggerName) {
        return await this.request(`/api/triggers/${triggerName}`, {
            method: 'DELETE'
        });
    }

    // AI Chat APIs
    static async getAIModels() {
        return await this.request('/api/ai-chat/models');
    }

    static async sendAIQuery(question, modelId, executeSQL = false, lmStudioUrl = null) {
        const payload = {
            question,
            model_id: modelId,
            execute_sql: executeSQL
        };

        if (lmStudioUrl) {
            payload.lm_studio_url = lmStudioUrl;
        }

        return await this.request('/api/ai-chat/query', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    static async sendConversationMessage(message, history, modelId, lmStudioUrl = null) {
        const payload = {
            message,
            history,
            model_id: modelId
        };

        if (lmStudioUrl) {
            payload.lm_studio_url = lmStudioUrl;
        }

        return await this.request('/api/ai-chat/conversation', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    static async validateAIConnection(lmStudioUrl) {
        return await this.request('/api/ai-chat/validate-connection', {
            method: 'POST',
            body: JSON.stringify({ lm_studio_url: lmStudioUrl })
        });
    }

    static async getSchemaExplanation(focus = 'overview', modelId, lmStudioUrl = null) {
        const payload = {
            focus,
            model_id: modelId
        };

        if (lmStudioUrl) {
            payload.lm_studio_url = lmStudioUrl;
        }

        return await this.request('/api/ai-chat/explain-schema', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    static async getSuggestedQueries(type = 'analysis', modelId, lmStudioUrl = null) {
        const payload = {
            type,
            model_id: modelId
        };

        if (lmStudioUrl) {
            payload.lm_studio_url = lmStudioUrl;
        }

        return await this.request('/api/ai-chat/suggest-queries', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    static async testAIModel(modelId, lmStudioUrl = null) {
        const payload = {
            model_id: modelId
        };

        if (lmStudioUrl) {
            payload.lm_studio_url = lmStudioUrl;
        }

        return await this.request('/api/ai-chat/test-model', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    static async debugLMStudio(lmStudioUrl = null, modelId = null) {
        const payload = {};

        if (lmStudioUrl) {
            payload.lm_studio_url = lmStudioUrl;
        }

        if (modelId) {
            payload.model_id = modelId;
        }

        return await this.request('/api/ai-chat/debug-lm-studio', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }
}

// Database connection functions
async function connectDatabase() {
    const fileInput = document.getElementById('databaseFile');
    const file = fileInput.files[0];

    if (!file) {
        showAlert('warning', 'Please select a database file');
        return;
    }

    const formData = new FormData();
    formData.append('database_file', file);

    try {
        showLoading('Connecting to database...');
        const result = await DatabaseExplorerAPI.connectDatabase(formData);

        if (result.success) {
            isConnected = true;
            updateConnectionStatus(true, file.name);
            await loadDatabaseInfo();
            await loadTables();
            showAlert('success', result.message);
        } else {
            showAlert('danger', 'Connection failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error connecting to database: ' + error.message);
    } finally {
        hideLoading();
    }
}

function updateConnectionStatus(connected, dbName = '') {
    const statusEl = document.getElementById('connectionStatus');
    const textEl = document.getElementById('connectionText');
    const indicator = statusEl.querySelector('.status-indicator');
    const dbInfoCard = document.getElementById('dbInfoCard');

    if (connected) {
        statusEl.className = 'connection-status connection-connected';
        indicator.className = 'status-indicator status-connected';
        textEl.textContent = 'Connected';

        if (dbName) {
            document.getElementById('dbName').textContent = dbName;
            dbInfoCard.style.display = 'block';
        }
    } else {
        statusEl.className = 'connection-status connection-disconnected';
        indicator.className = 'status-indicator status-disconnected';
        textEl.textContent = 'Not Connected';
        dbInfoCard.style.display = 'none';
    }
}

async function loadDatabaseInfo() {
    try {
        const data = await DatabaseExplorerAPI.getDatabaseInfo();

        if (data.error) {
            showAlert('danger', data.error);
            return;
        }

        // Update metrics
        document.getElementById('totalTables').textContent = data.total_tables;
        document.getElementById('totalRows').textContent = data.total_rows.toLocaleString();
        document.getElementById('totalSize').textContent = data.file_size_mb;
        document.getElementById('totalIndexes').textContent = data.total_indexes || 0;

        // Update database details
        const detailsEl = document.getElementById('databaseDetails');
        detailsEl.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Database Information</h6>
                    <p><strong>Name:</strong> ${data.database_name}</p>
                    <p><strong>Type:</strong> ${data.database_type}</p>
                    <p><strong>Size:</strong> ${data.file_size_mb} MB</p>
                </div>
                <div class="col-md-6">
                    <h6>Contents</h6>
                    <p><strong>Tables:</strong> ${data.total_tables}</p>
                    <p><strong>Views:</strong> ${data.total_views}</p>
                    <p><strong>Total Rows:</strong> ${data.total_rows.toLocaleString()}</p>
                </div>
            </div>
        `;

        // Update top tables
        if (data.tables && data.tables.length > 0) {
            const topTablesEl = document.getElementById('topTables');
            let tablesHtml = '<div class="table-responsive"><table class="table table-sm">';
            tablesHtml += '<thead><tr><th>Table</th><th>Rows</th><th>Columns</th><th>Actions</th></tr></thead><tbody>';

            data.tables.slice(0, 5).forEach(table => {
                tablesHtml += `
                    <tr>
                        <td><strong>${table.name}</strong></td>
                        <td>${table.rows.toLocaleString()}</td>
                        <td>${table.columns}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary" onclick="showTableDetails('${table.name}')">
                                <i class="fas fa-eye"></i> View
                            </button>
                        </td>
                    </tr>
                `;
            });

            tablesHtml += '</tbody></table></div>';
            topTablesEl.innerHTML = tablesHtml;
        }

        // Update sidebar info
        document.getElementById('dbSize').textContent = `${data.file_size_mb} MB`;
        document.getElementById('dbTables').textContent = `${data.total_tables} tables`;

        // Update health status
        updateHealthStatus(data);

    } catch (error) {
        showAlert('danger', 'Error loading database info: ' + error.message);
    }
}

function updateHealthStatus(dbInfo) {
    const healthEl = document.getElementById('healthStatus');
    let html = '';

    // Check for potential issues
    const issues = [];
    const recommendations = [];

    if (dbInfo.total_tables === 0) {
        issues.push('No tables found');
    }

    if (dbInfo.total_rows === 0) {
        issues.push('No data found');
    }

    if (dbInfo.total_indexes === 0 && dbInfo.total_tables > 0) {
        recommendations.push('Consider adding indexes for better performance');
    }

    if (issues.length === 0) {
        html = `
            <div class="text-success">
                <i class="fas fa-check-circle"></i> Database looks healthy
            </div>
        `;
    } else {
        html = `
            <div class="text-warning">
                <i class="fas fa-exclamation-triangle"></i> ${issues.join(', ')}
            </div>
        `;
    }

    if (recommendations.length > 0) {
        html += `
            <div class="mt-2 small text-muted">
                ${recommendations.join(', ')}
            </div>
        `;
    }

    healthEl.innerHTML = html;
}

async function loadTables() {
    try {
        const data = await DatabaseExplorerAPI.getTables();

        if (data.error) {
            showAlert('danger', data.error);
            return;
        }

        currentTables = data.tables || [];

        // Update sidebar tables list
        const tablesListEl = document.getElementById('tablesList');
        if (currentTables.length > 0) {
            let html = '';
            currentTables.forEach(table => {
                html += `
                    <div class="table-item p-2 mb-1" onclick="showTableDetails('${table.name}')">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="fas fa-table text-light"></i>
                                <span class="text-light ms-2">${table.name}</span>
                                <br>
                                <small class="text-light opacity-75">${table.rows.toLocaleString()} rows</small>
                            </div>
                            <i class="fas fa-chevron-right text-light opacity-50"></i>
                        </div>
                    </div>
                `;
            });
            tablesListEl.innerHTML = html;
        } else {
            tablesListEl.innerHTML = '<div class="text-light opacity-50 small">No tables found</div>';
        }

        // Update tables tab
        updateTablesGrid(currentTables);

    } catch (error) {
        showAlert('danger', 'Error loading tables: ' + error.message);
    }
}

function updateTablesGrid(tables) {
    const gridEl = document.getElementById('tablesGrid');

    if (!tables || tables.length === 0) {
        gridEl.innerHTML = `
            <div class="text-muted text-center py-4">
                <i class="fas fa-table fa-3x mb-3"></i>
                <p>No tables to display</p>
            </div>
        `;
        return;
    }

    let html = '<div class="row">';

    tables.forEach(table => {
        html += `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title">
                            <i class="fas fa-table text-primary"></i>
                            ${table.name}
                        </h5>
                        <p class="card-text">
                            <small class="text-muted">
                                ${table.rows.toLocaleString()} rows • ${table.columns} columns
                            </small>
                        </p>
                        <div class="btn-group w-100" role="group">
                            <button class="btn btn-outline-primary btn-sm" onclick="showTableDetails('${table.name}')">
                                <i class="fas fa-eye"></i> View
                            </button>
                            <button class="btn btn-outline-success btn-sm" onclick="queryTable('${table.name}')">
                                <i class="fas fa-code"></i> Query
                            </button>
                            <button class="btn btn-outline-info btn-sm" onclick="analyzeTableData('${table.name}')">
                                <i class="fas fa-chart-line"></i> Analyze
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    gridEl.innerHTML = html;
}

// SQL Query functions
async function executeQuery() {
    const query = document.getElementById('sqlQuery').value.trim();

    if (!query) {
        showAlert('warning', 'Please enter a SQL query');
        return;
    }

    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    try {
        showLoading('Executing query...');
        const result = await DatabaseExplorerAPI.executeQuery(query);

        if (result.success) {
            if (result.type === 'select') {
                currentQueryResults = result.data;
                displayQueryResults(result.data, result.columns);
                updateVisualizationColumns(result.columns);
                document.getElementById('exportBtn').style.display = 'inline-block';
                document.getElementById('visualizeBtn').style.display = 'inline-block';
                document.getElementById('resultCount').textContent = `${result.row_count} rows`;
                document.getElementById('resultCount').style.display = 'inline-block';
                showAlert('success', `Query executed successfully. ${result.row_count} rows returned.`);
            } else {
                document.getElementById('queryResults').innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle"></i> ${result.message}
                    </div>
                `;
                showAlert('success', result.message);
            }
        } else {
            document.getElementById('queryResults').innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle"></i> ${result.error}
                </div>
            `;
            showAlert('danger', 'Query failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error executing query: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayQueryResults(data, columns) {
    const resultsEl = document.getElementById('queryResults');

    if (!data || data.length === 0) {
        resultsEl.innerHTML = '<p class="text-muted">No results returned</p>';
        return;
    }

    let html = '<div class="table-responsive"><table class="table table-striped table-sm">';

    // Headers
    html += '<thead><tr>';
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Data rows (limit display for performance)
    const displayData = data.slice(0, 1000);
    displayData.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            const displayValue = value === null ? '<em class="text-muted">NULL</em>' : String(value);
            html += `<td>${displayValue}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';

    if (data.length > 1000) {
        html += `<p class="text-muted mt-2">Showing first 1000 rows of ${data.length} total results</p>`;
    }

    resultsEl.innerHTML = html;
}

function updateVisualizationColumns(columns) {
    const xSelect = document.getElementById('xColumn');
    const ySelect = document.getElementById('yColumn');

    xSelect.innerHTML = '<option value="">Select column...</option>';
    ySelect.innerHTML = '<option value="">Select column...</option>';

    columns.forEach(col => {
        xSelect.innerHTML += `<option value="${col}">${col}</option>`;
        ySelect.innerHTML += `<option value="${col}">${col}</option>`;
    });
}

// Visualization functions
async function generateChart() {
    if (!currentQueryResults || currentQueryResults.length === 0) {
        showAlert('warning', 'No data available for visualization. Execute a SELECT query first.');
        return;
    }

    const chartType = document.getElementById('chartType').value;
    const xColumn = document.getElementById('xColumn').value;
    const yColumn = document.getElementById('yColumn').value;

    if (!xColumn || !yColumn) {
        showAlert('warning', 'Please select columns for X and Y axes');
        return;
    }

    try {
        showLoading('Generating chart...');
        const title = `${chartType.charAt(0).toUpperCase() + chartType.slice(1)} Chart: ${yColumn} by ${xColumn}`;
        const result = await DatabaseExplorerAPI.createVisualization(
            currentQueryResults, chartType, xColumn, yColumn, title
        );

        if (result.success) {
            document.getElementById('chartContainer').innerHTML = `
                <img src="${result.image}" class="img-fluid" alt="Generated Chart">
            `;
            showAlert('success', 'Chart generated successfully');
        } else {
            showAlert('danger', 'Failed to generate chart: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error generating chart: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Export functions
async function exportResults() {
    if (!currentQueryResults || currentQueryResults.length === 0) {
        showAlert('warning', 'No data to export');
        return;
    }

    try {
        const result = await DatabaseExplorerAPI.exportCSV(currentQueryResults, 'query_results.csv');
        if (result.success) {
            showAlert('success', 'Results exported successfully');
        } else {
            showAlert('danger', 'Export failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error exporting results: ' + error.message);
    }
}

async function exportSchema() {
    try {
        const result = await DatabaseExplorerAPI.exportSchema();
        if (result.success) {
            showAlert('success', 'Schema exported successfully');
        } else {
            showAlert('danger', 'Export failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error exporting schema: ' + error.message);
    }
}

// Tools functions
async function runValidation() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    try {
        showLoading('Running data validation...');
        const result = await DatabaseExplorerAPI.validateData();

        const resultsEl = document.getElementById('validationResults');

        if (result.success) {
            let html = '<h6>Validation Results:</h6>';
            result.results.forEach(item => {
                const alertClass = item.type === 'error' ? 'alert-danger' :
                    item.type === 'warning' ? 'alert-warning' :
                        item.type === 'success' ? 'alert-success' : 'alert-info';

                html += `
                    <div class="alert ${alertClass} alert-sm">
                        <strong>${item.title}:</strong> ${item.message}
                    </div>
                `;
            });
            resultsEl.innerHTML = html;
        } else {
            showAlert('danger', 'Validation failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error running validation: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function runPerformanceAnalysis() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    try {
        showLoading('Analyzing performance...');
        const result = await DatabaseExplorerAPI.performanceAnalysis();

        const resultsEl = document.getElementById('performanceResults');

        if (result.success) {
            const analysis = result.analysis;

            let html = '<h6>Performance Analysis:</h6>';

            // Table sizes
            if (analysis.table_sizes && analysis.table_sizes.length > 0) {
                html += '<h6 class="mt-3">Largest Tables:</h6>';
                html += '<div class="table-responsive"><table class="table table-sm">';
                html += '<thead><tr><th>Table</th><th>Rows</th><th>Est. Size</th></tr></thead><tbody>';

                analysis.table_sizes.slice(0, 5).forEach(table => {
                    html += `
                        <tr>
                            <td><strong>${table.name}</strong></td>
                            <td>${table.rows.toLocaleString()}</td>
                            <td>${table.estimated_size_mb} MB</td>
                        </tr>
                    `;
                });

                html += '</tbody></table></div>';
            }

            // Recommendations
            if (analysis.recommendations && analysis.recommendations.length > 0) {
                html += '<h6 class="mt-3">Recommendations:</h6>';
                analysis.recommendations.forEach(rec => {
                    html += `<div class="alert alert-info alert-sm">${rec}</div>`;
                });
            }

            resultsEl.innerHTML = html;
        } else {
            showAlert('danger', 'Performance analysis failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error running performance analysis: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Continuation of getDatabaseStats function and remaining frontend functionality

async function getDatabaseStats() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    try {
        showLoading('Getting database statistics...');
        const result = await DatabaseExplorerAPI.getDatabaseStats();

        const resultsEl = document.getElementById('statsResults');

        if (result.success) {
            const stats = result.stats;

            let html = '<h6>Database Statistics:</h6>';

            // Overview
            html += `
                <div class="row mb-3">
                    <div class="col-md-6">
                        <strong>Database:</strong> ${stats.database_overview.database_name}<br>
                        <strong>File Size:</strong> ${stats.database_overview.file_size_mb} MB<br>
                        <strong>Total Tables:</strong> ${stats.database_overview.total_tables}<br>
                        <strong>Total Rows:</strong> ${stats.database_overview.total_rows.toLocaleString()}
                    </div>
                    <div class="col-md-6">
                        <strong>Total Columns:</strong> ${stats.database_overview.total_columns}<br>
                        <strong>Views:</strong> ${stats.database_overview.total_views}<br>
                        <strong>Triggers:</strong> ${stats.database_overview.total_triggers}<br>
                        <strong>Schema Version:</strong> ${stats.database_overview.schema_version}
                    </div>
                </div>
            `;

            // Column analysis
            if (stats.column_analysis) {
                html += '<h6>Column Analysis:</h6>';
                html += `
                    <p><strong>Most Common Type:</strong> ${stats.column_analysis.most_common_type || 'N/A'}</p>
                    <p><strong>Average Columns per Table:</strong> ${stats.column_analysis.avg_columns_per_table}</p>
                `;
            }

            // Index statistics
            if (stats.index_stats) {
                html += '<h6>Index Statistics:</h6>';
                html += `
                    <p><strong>Total Indexes:</strong> ${stats.index_stats.total_indexes}</p>
                    <p><strong>Index Coverage:</strong> ${stats.index_stats.index_coverage_percentage}%</p>
                    <p><strong>Strategy:</strong> ${stats.index_stats.indexing_strategy}</p>
                `;
            }

            // Data quality
            if (stats.data_quality) {
                html += '<h6>Data Quality:</h6>';
                html += `
                    <p><strong>Quality Score:</strong> ${stats.data_quality.data_quality_score}/100</p>
                    <p><strong>Tables with NULL Issues:</strong> ${stats.data_quality.tables_with_null_issues}</p>
                `;
            }

            resultsEl.innerHTML = html;
        } else {
            showAlert('danger', 'Failed to get database statistics: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error getting database statistics: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Table analysis functions
async function showTableDetails(tableName) {
    try {
        showLoading('Loading table details...');

        // Get table structure
        const structureResult = await DatabaseExplorerAPI.getTableStructure(tableName);

        // Get table data
        const dataResult = await DatabaseExplorerAPI.getTableData(tableName, 50);

        // Show modal with table details
        showTableModal(tableName, structureResult, dataResult);

    } catch (error) {
        showAlert('danger', 'Error loading table details: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showTableModal(tableName, structure, data) {
    // Create modal HTML
    const modalHtml = `
        <div class="modal fade" id="tableModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-table"></i> Table: ${tableName}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <ul class="nav nav-tabs" id="tableDetailTabs">
                            <li class="nav-item">
                                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#structure">
                                    <i class="fas fa-sitemap"></i> Structure
                                </button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#data">
                                    <i class="fas fa-table"></i> Data
                                </button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#relationships">
                                    <i class="fas fa-link"></i> Relationships
                                </button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#analysis">
                                    <i class="fas fa-chart-line"></i> Analysis
                                </button>
                            </li>
                        </ul>
                        
                        <div class="tab-content mt-3">
                            <div class="tab-pane fade show active" id="structure">
                                ${generateStructureHTML(structure)}
                            </div>
                            <div class="tab-pane fade" id="data">
                                ${generateDataHTML(data)}
                            </div>
                            <div class="tab-pane fade" id="relationships">
                                <div class="text-center py-3">
                                    <button class="btn btn-primary" onclick="loadTableRelationships('${tableName}')">
                                        <i class="fas fa-search"></i> Load Relationships
                                    </button>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="analysis">
                                <div class="text-center py-3">
                                    <button class="btn btn-primary" onclick="loadTableAnalysis('${tableName}')">
                                        <i class="fas fa-chart-line"></i> Analyze Table
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-outline-secondary" onclick="queryTable('${tableName}')" data-bs-dismiss="modal">
                            <i class="fas fa-code"></i> Query Table
                        </button>
                        <button class="btn btn-outline-info" onclick="generateTableSQL('${tableName}')">
                            <i class="fas fa-file-code"></i> Generate SQL
                        </button>
                        <button class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal
    const existingModal = document.getElementById('tableModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('tableModal'));
    modal.show();
}

function generateStructureHTML(structure) {
    if (!structure || !structure.success) {
        return '<p class="text-muted">Could not load table structure</p>';
    }

    const tableStructure = structure.structure;
    let html = '';

    // Columns
    html += '<h6>Columns</h6>';
    html += '<div class="table-responsive"><table class="table table-sm">';
    html += '<thead><tr><th>Name</th><th>Type</th><th>Nullable</th><th>Default</th><th>Primary Key</th></tr></thead><tbody>';

    if (tableStructure.columns) {
        tableStructure.columns.forEach(col => {
            const isPK = tableStructure.primary_keys && tableStructure.primary_keys.includes(col.name);
            html += `
                <tr>
                    <td><strong>${col.name}</strong></td>
                    <td><span class="badge bg-secondary">${col.type}</span></td>
                    <td>${col.nullable ? '✓' : '✗'}</td>
                    <td>${col.default || '-'}</td>
                    <td>${isPK ? '<i class="fas fa-key text-warning"></i>' : ''}</td>
                </tr>
            `;
        });
    }

    html += '</tbody></table></div>';

    // Foreign Keys
    if (tableStructure.foreign_keys && tableStructure.foreign_keys.length > 0) {
        html += '<h6 class="mt-4">Foreign Keys</h6>';
        html += '<div class="table-responsive"><table class="table table-sm">';
        html += '<thead><tr><th>Column</th><th>References</th></tr></thead><tbody>';

        tableStructure.foreign_keys.forEach(fk => {
            html += `
                <tr>
                    <td><strong>${fk.column}</strong></td>
                    <td>${fk.referenced_table}.${fk.referenced_column}</td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
    }

    // Indexes
    if (tableStructure.indexes && tableStructure.indexes.length > 0) {
        html += '<h6 class="mt-4">Indexes</h6>';
        html += '<div class="table-responsive"><table class="table table-sm">';
        html += '<thead><tr><th>Name</th><th>Columns</th><th>Unique</th></tr></thead><tbody>';

        tableStructure.indexes.forEach(idx => {
            const columns = idx.columns ? idx.columns.join(', ') : 'N/A';
            const unique = idx.unique ? '<i class="fas fa-check text-success"></i>' : '';

            html += `
                <tr>
                    <td><strong>${idx.name}</strong></td>
                    <td>${columns}</td>
                    <td>${unique}</td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
    }

    return html;
}

function generateDataHTML(data) {
    if (!data || !data.success) {
        return '<p class="text-muted">Could not load table data</p>';
    }

    if (!data.data || data.data.length === 0) {
        return '<p class="text-muted">No data found in table</p>';
    }

    let html = `<p class="text-muted">Showing first 50 rows (${data.row_count} total)</p>`;
    html += '<div class="table-responsive"><table class="table table-sm table-striped">';

    // Headers
    html += '<thead><tr>';
    data.columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Data rows
    data.data.forEach(row => {
        html += '<tr>';
        data.columns.forEach(col => {
            const value = row[col];
            const displayValue = value === null ? '<em class="text-muted">NULL</em>' : String(value);
            html += `<td>${displayValue}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';

    return html;
}

async function loadTableRelationships(tableName) {
    try {
        const result = await DatabaseExplorerAPI.getTableRelationships(tableName);

        let html = '';

        if (result.success) {
            if (result.relationships.length === 0) {
                html = '<p class="text-muted">No relationships found for this table</p>';
            } else {
                html = '<h6>Table Relationships</h6>';

                result.relationships.forEach(rel => {
                    const icon = rel.type === 'references' ? 'fas fa-arrow-right' : 'fas fa-arrow-left';
                    const color = rel.type === 'references' ? 'text-primary' : 'text-success';

                    html += `
                        <div class="border rounded p-3 mb-2">
                            <div class="d-flex align-items-center">
                                <i class="${icon} ${color} me-2"></i>
                                <strong>${rel.table}</strong>
                                <span class="badge bg-secondary ms-2">${rel.relationship}</span>
                            </div>
                            <small class="text-muted">
                                ${rel.foreign_key.column} → ${rel.foreign_key.referenced_table}.${rel.foreign_key.referenced_column}
                            </small>
                        </div>
                    `;
                });
            }
        } else {
            html = `<p class="text-danger">Error loading relationships: ${result.error}</p>`;
        }

        document.getElementById('relationships').innerHTML = html;

    } catch (error) {
        document.getElementById('relationships').innerHTML =
            `<p class="text-danger">Error loading relationships: ${error.message}</p>`;
    }
}

async function loadTableAnalysis(tableName) {
    try {
        showLoading('Analyzing table...');

        const result = await DatabaseExplorerAPI.analyzeTable(tableName, true, true);

        let html = '';

        if (result.success) {
            const analysis = result.analysis;

            html += '<h6>Table Analysis</h6>';

            // Basic stats
            html += `
                <div class="row mb-3">
                    <div class="col-md-6">
                        <strong>Total Rows:</strong> ${analysis.basic_stats.total_rows.toLocaleString()}<br>
                        <strong>Total Columns:</strong> ${analysis.basic_stats.total_columns}
                    </div>
                </div>
            `;

            // Numeric analysis
            if (analysis.numeric_analysis && analysis.numeric_analysis.length > 0) {
                html += '<h6>Numeric Columns</h6>';
                html += '<div class="table-responsive"><table class="table table-sm">';
                html += '<thead><tr><th>Column</th><th>Min</th><th>Max</th><th>Average</th><th>Nulls</th></tr></thead><tbody>';

                analysis.numeric_analysis.forEach(col => {
                    html += `
                        <tr>
                            <td><strong>${col.column}</strong></td>
                            <td>${col.min_value}</td>
                            <td>${col.max_value}</td>
                            <td>${col.avg_value}</td>
                            <td>${col.null_count}</td>
                        </tr>
                    `;
                });

                html += '</tbody></table></div>';
            }

            // Text analysis
            if (analysis.text_analysis && analysis.text_analysis.length > 0) {
                html += '<h6>Text Columns</h6>';
                html += '<div class="table-responsive"><table class="table table-sm">';
                html += '<thead><tr><th>Column</th><th>Min Length</th><th>Max Length</th><th>Avg Length</th><th>Nulls</th></tr></thead><tbody>';

                analysis.text_analysis.forEach(col => {
                    html += `
                        <tr>
                            <td><strong>${col.column}</strong></td>
                            <td>${col.min_length}</td>
                            <td>${col.max_length}</td>
                            <td>${col.avg_length}</td>
                            <td>${col.null_count}</td>
                        </tr>
                    `;
                });

                html += '</tbody></table></div>';
            }

            // Data distribution
            if (analysis.data_distribution && analysis.data_distribution.length > 0) {
                html += '<h6>Data Distribution</h6>';

                analysis.data_distribution.forEach(dist => {
                    html += `<h6 class="h7">${dist.column}</h6>`;
                    html += '<div class="mb-3">';

                    dist.value_counts.forEach(value => {
                        html += `
                            <div class="d-flex justify-content-between">
                                <span>${value.value}</span>
                                <span>${value.count} (${value.percentage}%)</span>
                            </div>
                        `;
                    });

                    html += '</div>';
                });
            }

        } else {
            html = `<p class="text-danger">Error analyzing table: ${result.error}</p>`;
        }

        document.getElementById('analysis').innerHTML = html;

    } catch (error) {
        document.getElementById('analysis').innerHTML =
            `<p class="text-danger">Error analyzing table: ${error.message}</p>`;
    } finally {
        hideLoading();
    }
}

async function generateTableSQL(tableName) {
    try {
        const result = await DatabaseExplorerAPI.generateTableSQL(tableName);

        if (result.success) {
            // Show SQL templates in a modal
            showSQLTemplatesModal(tableName, result.sql_templates);
        } else {
            showAlert('danger', 'Failed to generate SQL: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error generating SQL: ' + error.message);
    }
}

function showSQLTemplatesModal(tableName, templates) {
    const modalHtml = `
        <div class="modal fade" id="sqlTemplatesModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-code"></i> SQL Templates for ${tableName}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        ${generateSQLTemplatesHTML(templates)}
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal
    const existingModal = document.getElementById('sqlTemplatesModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('sqlTemplatesModal'));
    modal.show();
}

function generateSQLTemplatesHTML(templates) {
    let html = '';

    Object.entries(templates).forEach(([key, sql]) => {
        const title = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        html += `
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-center">
                    <h6>${title}</h6>
                    <button class="btn btn-sm btn-outline-primary" onclick="copyToClipboard('${sql.replace(/'/g, "\\'")}')">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                </div>
                <pre><code class="language-sql">${sql}</code></pre>
            </div>
        `;
    });

    return html;
}

// AI Chat functions
async function testAIConnection() {
    const lmStudioUrl = document.getElementById('lmStudioUrl').value;

    try {
        showLoading('Testing AI connection...');
        const result = await DatabaseExplorerAPI.validateAIConnection(lmStudioUrl);

        if (result.valid && result.connected) {
            showAlert('success', result.message + ` (${result.total_models} models available)`);
        } else {
            showAlert('warning', result.error + '. ' + result.suggestion);
        }
    } catch (error) {
        showAlert('danger', 'Error testing AI connection: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) {
        return;
    }

    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    const modelId = document.getElementById('aiModel').value;
    const lmStudioUrl = document.getElementById('lmStudioUrl').value;

    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';

    try {
        // Show typing indicator
        addChatMessage('AI is thinking...', 'ai', true);

        const result = await DatabaseExplorerAPI.sendAIQuery(message, modelId, false, lmStudioUrl);

        // Remove typing indicator
        removeChatMessage('typing');

        if (result.success) {
            addChatMessage(result.ai_response, 'ai');

            // If SQL was detected, offer to execute it
            if (result.sql_query) {
                addChatMessage(
                    `I found this SQL query in my response:\n\`\`\`sql\n${result.sql_query}\n\`\`\`\n\nWould you like me to execute it?`,
                    'ai'
                );

                // Add execute button
                addChatExecuteButton(result.sql_query);
            }

            // Update chat history
            chatHistory.push({
                user: message,
                assistant: result.ai_response
            });

        } else {
            addChatMessage(`Error: ${result.error}`, 'ai');
        }

    } catch (error) {
        removeChatMessage('typing');
        addChatMessage(`Error: ${error.message}`, 'ai');
    }
}

function addChatMessage(message, sender, isTyping = false) {
    const container = document.getElementById('chatContainer');
    const messageClass = sender === 'user' ? 'chat-user' : 'chat-ai';
    const icon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';

    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${messageClass}`;
    if (isTyping) messageEl.id = 'typing-message';

    messageEl.innerHTML = `
        <div class="d-flex align-items-start">
            <i class="${icon} me-2 mt-1"></i>
            <div>${formatChatMessage(message)}</div>
        </div>
    `;

    container.appendChild(messageEl);
    container.scrollTop = container.scrollHeight;
}

function removeChatMessage(type) {
    if (type === 'typing') {
        const typingEl = document.getElementById('typing-message');
        if (typingEl) {
            typingEl.remove();
        }
    }
}

function addChatExecuteButton(sql) {
    const container = document.getElementById('chatContainer');

    const buttonEl = document.createElement('div');
    buttonEl.className = 'chat-message chat-ai';
    buttonEl.innerHTML = `
        <button class="btn btn-primary btn-sm" onclick="executeAIQuery('${sql.replace(/'/g, "\\'")}')">
            <i class="fas fa-play"></i> Execute Query
        </button>
    `;

    container.appendChild(buttonEl);
    container.scrollTop = container.scrollHeight;
}

async function executeAIQuery(sql) {
    // Insert SQL into editor and execute
    document.getElementById('sqlQuery').value = sql;
    showTab('query');
    await executeQuery();
}

function formatChatMessage(message) {
    // Basic markdown-like formatting
    message = message.replace(/```sql\n([\s\S]*?)\n```/g, '<pre><code class="language-sql">$1</code></pre>');
    message = message.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    message = message.replace(/`([^`]+)`/g, '<code>$1</code>');
    message = message.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    message = message.replace(/\*(.*?)\*/g, '<em>$1</em>');

    return message.replace(/\n/g, '<br>');
}

async function getSchemaExplanation() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    const modelId = document.getElementById('aiModel').value;
    const lmStudioUrl = document.getElementById('lmStudioUrl').value;

    try {
        showLoading('Getting schema explanation...');
        const result = await DatabaseExplorerAPI.getSchemaExplanation('overview', modelId, lmStudioUrl);

        if (result.success) {
            addChatMessage(result.explanation, 'ai');
        } else {
            showAlert('danger', 'Failed to get schema explanation: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error getting schema explanation: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function getSuggestedQueries() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    const modelId = document.getElementById('aiModel').value;
    const lmStudioUrl = document.getElementById('lmStudioUrl').value;

    try {
        showLoading('Getting query suggestions...');
        const result = await DatabaseExplorerAPI.getSuggestedQueries('analysis', modelId, lmStudioUrl);

        if (result.success) {
            addChatMessage(result.suggestions, 'ai');
        } else {
            showAlert('danger', 'Failed to get query suggestions: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error getting query suggestions: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Search and sorting functions
async function searchTables() {
    const keyword = document.getElementById('tableSearch').value.trim();

    if (!keyword) {
        // Reset to show all tables
        updateTablesGrid(currentTables);
        return;
    }

    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    try {
        const result = await DatabaseExplorerAPI.searchTables(keyword);

        if (result.success) {
            // Filter current tables to show only matching ones
            const matchingTables = currentTables.filter(table =>
                result.tables.includes(table.name)
            );
            updateTablesGrid(matchingTables);
        } else {
            showAlert('danger', 'Search failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error searching tables: ' + error.message);
    }
}

function sortTables(sortBy) {
    const sortedTables = [...currentTables];

    switch (sortBy) {
        case 'name':
            sortedTables.sort((a, b) => a.name.localeCompare(b.name));
            break;
        case 'rows':
            sortedTables.sort((a, b) => b.rows - a.rows);
            break;
        case 'size':
            sortedTables.sort((a, b) => (b.rows * b.columns) - (a.rows * a.columns));
            break;
    }

    updateTablesGrid(sortedTables);
}

// Utility functions
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });

    // Show selected tab
    const targetTab = document.getElementById(`${tabName}-tab`);
    if (targetTab) {
        targetTab.classList.add('active');
    }

    // Update navigation
    document.querySelectorAll('.sidebar .nav-link').forEach(link => {
        link.classList.remove('active');
    });

    // Find and activate the corresponding nav link
    const navLinks = document.querySelectorAll('.sidebar .nav-link');
    navLinks.forEach(link => {
        if (link.getAttribute('onclick') && link.getAttribute('onclick').includes(`'${tabName}'`)) {
            link.classList.add('active');
        }
    });
}

function queryTable(tableName) {
    const query = `SELECT * FROM \`${tableName}\` LIMIT 100;`;
    document.getElementById('sqlQuery').value = query;
    showTab('query');
}

function insertQuery(query) {
    document.getElementById('sqlQuery').value = query;
}

function clearQuery() {
    document.getElementById('sqlQuery').value = '';

    // Clear results
    document.getElementById('queryResults').innerHTML = `
        <div class="text-muted text-center py-4">
            <i class="fas fa-play-circle fa-3x mb-3"></i>
            <p>Execute a query to see results here</p>
        </div>
    `;

    // Hide export buttons
    document.getElementById('exportBtn').style.display = 'none';
    document.getElementById('visualizeBtn').style.display = 'none';
    document.getElementById('resultCount').style.display = 'none';

    // Clear visualization columns
    document.getElementById('xColumn').innerHTML = '<option value="">Select column...</option>';
    document.getElementById('yColumn').innerHTML = '<option value="">Select column...</option>';

    currentQueryResults = [];
}

// Continuation of formatQuery() and remaining utility functions

function formatQuery() {
    // Basic SQL formatting
    const query = document.getElementById('sqlQuery').value;
    const formatted = query
        .replace(/\bSELECT\b/gi, 'SELECT')
        .replace(/\bFROM\b/gi, '\nFROM')
        .replace(/\bWHERE\b/gi, '\nWHERE')
        .replace(/\bGROUP BY\b/gi, '\nGROUP BY')
        .replace(/\bHAVING\b/gi, '\nHAVING')
        .replace(/\bORDER BY\b/gi, '\nORDER BY')
        .replace(/\bLIMIT\b/gi, '\nLIMIT')
        .replace(/\bJOIN\b/gi, '\nJOIN')
        .replace(/\bINNER JOIN\b/gi, '\nINNER JOIN')
        .replace(/\bLEFT JOIN\b/gi, '\nLEFT JOIN')
        .replace(/\bRIGHT JOIN\b/gi, '\nRIGHT JOIN')
        .replace(/\bFULL JOIN\b/gi, '\nFULL JOIN')
        .replace(/\bUNION\b/gi, '\nUNION')
        .replace(/\bINSERT INTO\b/gi, 'INSERT INTO')
        .replace(/\bUPDATE\b/gi, 'UPDATE')
        .replace(/\bSET\b/gi, '\nSET')
        .replace(/\bDELETE FROM\b/gi, 'DELETE FROM')
        .replace(/\bCREATE TABLE\b/gi, 'CREATE TABLE')
        .replace(/\bDROP TABLE\b/gi, 'DROP TABLE')
        .replace(/,/g, ',\n    ')
        .replace(/\n\s*\n/g, '\n') // Remove empty lines
        .trim();

    document.getElementById('sqlQuery').value = formatted;
}

function visualizeResults() {
    showTab('visualization');
}

function refreshTables() {
    if (isConnected) {
        loadTables();
        loadDatabaseInfo();
    }
}

async function analyzeTableData(tableName) {
    try {
        showLoading('Analyzing table data...');

        // Get duplicates analysis
        const duplicatesResult = await DatabaseExplorerAPI.findDuplicates(tableName);

        // Get null analysis
        const nullResult = await DatabaseExplorerAPI.getNullAnalysis(tableName);

        // Show analysis modal
        showAnalysisModal(tableName, duplicatesResult, nullResult);

    } catch (error) {
        showAlert('danger', 'Error analyzing table: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showAnalysisModal(tableName, duplicatesResult, nullResult) {
    const modalHtml = `
        <div class="modal fade" id="analysisModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-chart-line"></i> Data Analysis: ${tableName}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        ${generateAnalysisHTML(duplicatesResult, nullResult)}
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal
    const existingModal = document.getElementById('analysisModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('analysisModal'));
    modal.show();
}

function generateAnalysisHTML(duplicatesResult, nullResult) {
    let html = '';

    // Duplicates analysis
    html += '<h6>Duplicate Records</h6>';
    if (duplicatesResult.success && duplicatesResult.duplicates.length > 0) {
        html += `<p class="text-warning">Found ${duplicatesResult.total_duplicate_groups} groups of duplicate records</p>`;
        html += '<div class="table-responsive"><table class="table table-sm">';
        html += '<thead><tr>';

        // Headers from first duplicate
        if (duplicatesResult.duplicates[0]) {
            Object.keys(duplicatesResult.duplicates[0]).forEach(key => {
                if (key !== 'duplicate_count') {
                    html += `<th>${key}</th>`;
                }
            });
            html += '<th>Count</th>';
        }

        html += '</tr></thead><tbody>';

        duplicatesResult.duplicates.slice(0, 10).forEach(dup => {
            html += '<tr>';
            Object.entries(dup).forEach(([key, value]) => {
                if (key === 'duplicate_count') {
                    html += `<td><span class="badge bg-warning">${value}</span></td>`;
                } else {
                    html += `<td>${value}</td>`;
                }
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';

        if (duplicatesResult.duplicates.length > 10) {
            html += `<p class="text-muted">Showing first 10 of ${duplicatesResult.duplicates.length} duplicate groups</p>`;
        }
    } else {
        html += '<p class="text-success">No duplicate records found</p>';
    }

    // NULL analysis
    html += '<h6 class="mt-4">NULL Value Analysis</h6>';
    if (nullResult.success && nullResult.null_analysis.length > 0) {
        html += '<div class="table-responsive"><table class="table table-sm">';
        html += '<thead><tr><th>Column</th><th>Data Type</th><th>NULL Count</th><th>NULL %</th><th>Non-NULL Count</th></tr></thead><tbody>';

        nullResult.null_analysis.forEach(col => {
            const alertClass = col.null_percentage > 50 ? 'table-danger' :
                col.null_percentage > 20 ? 'table-warning' : '';

            html += `
                <tr class="${alertClass}">
                    <td><strong>${col.column}</strong></td>
                    <td>${col.data_type}</td>
                    <td>${col.null_count}</td>
                    <td>${col.null_percentage}%</td>
                    <td>${col.non_null_count}</td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';

        // Summary
        const highNullCols = nullResult.null_analysis.filter(col => col.null_percentage > 50);
        if (highNullCols.length > 0) {
            html += `<div class="alert alert-warning">
                <strong>Warning:</strong> ${highNullCols.length} columns have more than 50% NULL values
            </div>`;
        }
    } else {
        html += '<p class="text-muted">Could not analyze NULL values</p>';
    }

    return html;
}

async function showQueryTemplates() {
    try {
        const result = await DatabaseExplorerAPI.getSampleQueries();
        showQueryTemplatesModal(result);
    } catch (error) {
        showAlert('danger', 'Error loading query templates: ' + error.message);
    }
}

function showQueryTemplatesModal(queries) {
    const modalHtml = `
        <div class="modal fade" id="queryTemplatesModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-code"></i> SQL Query Templates
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        ${generateQueryTemplatesHTML(queries)}
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal
    const existingModal = document.getElementById('queryTemplatesModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('queryTemplatesModal'));
    modal.show();
}

function generateQueryTemplatesHTML(queries) {
    let html = '';

    Object.entries(queries).forEach(([category, categoryQueries]) => {
        const categoryTitle = category.charAt(0).toUpperCase() + category.slice(1);

        html += `<h6>${categoryTitle} Queries</h6>`;
        html += '<div class="row mb-4">';

        categoryQueries.forEach(query => {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title">${query.name}</h6>
                            <pre class="small"><code class="language-sql">${query.query}</code></pre>
                            <button class="btn btn-sm btn-primary" onclick="useQueryTemplate('${query.query.replace(/'/g, "\\'")}')">
                                Use Template
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
    });

    return html;
}

function useQueryTemplate(query) {
    document.getElementById('sqlQuery').value = query;

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('queryTemplatesModal'));
    if (modal) {
        modal.hide();
    }

    // Switch to query tab
    showTab('query');
}

// Schema management functions
function showCreateTableModal() {
    const modalHtml = `
        <div class="modal fade" id="createTableModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-table"></i> Create Table
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="createTableForm">
                            <div class="mb-3">
                                <label class="form-label">Table Name</label>
                                <input type="text" class="form-control" id="newTableName" required>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Columns</label>
                                <div id="columnsContainer">
                                    <div class="row mb-2">
                                        <div class="col-md-3">
                                            <input type="text" class="form-control" placeholder="Column name" required>
                                        </div>
                                        <div class="col-md-3">
                                            <select class="form-select">
                                                <option value="TEXT">TEXT</option>
                                                <option value="INTEGER">INTEGER</option>
                                                <option value="REAL">REAL</option>
                                                <option value="BLOB">BLOB</option>
                                                <option value="NUMERIC">NUMERIC</option>
                                            </select>
                                        </div>
                                        <div class="col-md-3">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox">
                                                <label class="form-check-label">NOT NULL</label>
                                            </div>
                                        </div>
                                        <div class="col-md-3">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox">
                                                <label class="form-check-label">PRIMARY KEY</label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <button type="button" class="btn btn-outline-secondary btn-sm" onclick="addColumnRow()">
                                    <i class="fas fa-plus"></i> Add Column
                                </button>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="createNewTable()">Create Table</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('createTableModal'));
    modal.show();
}

function addColumnRow() {
    const container = document.getElementById('columnsContainer');
    const newRow = document.createElement('div');
    newRow.className = 'row mb-2';
    newRow.innerHTML = `
        <div class="col-md-3">
            <input type="text" class="form-control" placeholder="Column name" required>
        </div>
        <div class="col-md-3">
            <select class="form-select">
                <option value="TEXT">TEXT</option>
                <option value="INTEGER">INTEGER</option>
                <option value="REAL">REAL</option>
                <option value="BLOB">BLOB</option>
                <option value="NUMERIC">NUMERIC</option>
            </select>
        </div>
        <div class="col-md-2">
            <div class="form-check">
                <input class="form-check-input" type="checkbox">
                <label class="form-check-label">NOT NULL</label>
            </div>
        </div>
        <div class="col-md-2">
            <div class="form-check">
                <input class="form-check-input" type="checkbox">
                <label class="form-check-label">PRIMARY KEY</label>
            </div>
        </div>
        <div class="col-md-2">
            <button type="button" class="btn btn-outline-danger btn-sm" onclick="removeColumnRow(this)">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;
    container.appendChild(newRow);
}

function removeColumnRow(button) {
    button.closest('.row').remove();
}

async function createNewTable() {
    const tableName = document.getElementById('newTableName').value.trim();

    if (!tableName) {
        showAlert('warning', 'Please enter a table name');
        return;
    }

    // Collect column definitions
    const columnRows = document.querySelectorAll('#columnsContainer .row');
    const columns = [];

    columnRows.forEach(row => {
        const nameInput = row.querySelector('input[type="text"]');
        const typeSelect = row.querySelector('select');
        const notNullCheck = row.querySelector('input[type="checkbox"]:nth-of-type(1)');
        const pkCheck = row.querySelector('input[type="checkbox"]:nth-of-type(2)');

        if (nameInput && nameInput.value.trim()) {
            const column = {
                name: nameInput.value.trim(),
                type: typeSelect.value,
                constraints: []
            };

            if (notNullCheck && notNullCheck.checked) {
                column.constraints.push('NOT NULL');
            }

            if (pkCheck && pkCheck.checked) {
                column.constraints.push('PRIMARY KEY');
            }

            columns.push(column);
        }
    });

    if (columns.length === 0) {
        showAlert('warning', 'Please define at least one column');
        return;
    }

    try {
        showLoading('Creating table...');
        const result = await DatabaseExplorerAPI.createTable(tableName, columns);

        if (result.success) {
            showAlert('success', result.message);

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createTableModal'));
            if (modal) {
                modal.hide();
            }

            // Refresh tables
            await loadTables();
        } else {
            showAlert('danger', 'Failed to create table: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error creating table: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showCreateIndexModal() {
    if (!isConnected || currentTables.length === 0) {
        showAlert('warning', 'No tables available');
        return;
    }

    const tableOptions = currentTables.map(table =>
        `<option value="${table.name}">${table.name}</option>`
    ).join('');

    const modalHtml = `
        <div class="modal fade" id="createIndexModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-index"></i> Create Index
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="createIndexForm">
                            <div class="mb-3">
                                <label class="form-label">Index Name</label>
                                <input type="text" class="form-control" id="newIndexName" required>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Table</label>
                                <select class="form-select" id="indexTableName" onchange="loadTableColumns(this.value)" required>
                                    <option value="">Select table...</option>
                                    ${tableOptions}
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Columns</label>
                                <div id="indexColumnsContainer">
                                    <p class="text-muted">Select a table first</p>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="uniqueIndex">
                                    <label class="form-check-label">Unique Index</label>
                                </div>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="createNewIndex()">Create Index</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('createIndexModal'));
    modal.show();
}

async function loadTableColumns(tableName) {
    if (!tableName) return;

    try {
        const result = await DatabaseExplorerAPI.getTableStructure(tableName);

        if (result.success) {
            const container = document.getElementById('indexColumnsContainer');
            let html = '';

            result.structure.columns.forEach(col => {
                html += `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="${col.name}" id="col_${col.name}">
                        <label class="form-check-label" for="col_${col.name}">
                            ${col.name} (${col.type})
                        </label>
                    </div>
                `;
            });

            container.innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading table columns:', error);
    }
}

async function createNewIndex() {
    const indexName = document.getElementById('newIndexName').value.trim();
    const tableName = document.getElementById('indexTableName').value;
    const isUnique = document.getElementById('uniqueIndex').checked;

    if (!indexName) {
        showAlert('warning', 'Please enter an index name');
        return;
    }

    if (!tableName) {
        showAlert('warning', 'Please select a table');
        return;
    }

    // Get selected columns
    const selectedColumns = [];
    document.querySelectorAll('#indexColumnsContainer input[type="checkbox"]:checked').forEach(checkbox => {
        selectedColumns.push(checkbox.value);
    });

    if (selectedColumns.length === 0) {
        showAlert('warning', 'Please select at least one column');
        return;
    }

    try {
        showLoading('Creating index...');
        const result = await DatabaseExplorerAPI.createIndex(indexName, tableName, selectedColumns, isUnique);

        if (result.success) {
            showAlert('success', result.message);

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createIndexModal'));
            if (modal) {
                modal.hide();
            }
        } else {
            showAlert('danger', 'Failed to create index: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error creating index: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showCreateViewModal() {
    const modalHtml = `
        <div class="modal fade" id="createViewModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-eye"></i> Create View
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="createViewForm">
                            <div class="mb-3">
                                <label class="form-label">View Name</label>
                                <input type="text" class="form-control" id="newViewName" required>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">SELECT Query</label>
                                <textarea class="form-control query-editor" id="viewQuery" rows="8" 
                                          placeholder="SELECT column1, column2 FROM table_name WHERE condition" required></textarea>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="createNewView()">Create View</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('createViewModal'));
    modal.show();
}

async function createNewView() {
    const viewName = document.getElementById('newViewName').value.trim();
    const selectQuery = document.getElementById('viewQuery').value.trim();

    if (!viewName) {
        showAlert('warning', 'Please enter a view name');
        return;
    }

    if (!selectQuery) {
        showAlert('warning', 'Please enter a SELECT query');
        return;
    }

    try {
        showLoading('Creating view...');
        const result = await DatabaseExplorerAPI.createView(viewName, selectQuery);

        if (result.success) {
            showAlert('success', result.message);

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createViewModal'));
            if (modal) {
                modal.hide();
            }
        } else {
            showAlert('danger', 'Failed to create view: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error creating view: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showCreateTriggerModal() {
    if (!isConnected || currentTables.length === 0) {
        showAlert('warning', 'No tables available');
        return;
    }

    const tableOptions = currentTables.map(table =>
        `<option value="${table.name}">${table.name}</option>`
    ).join('');

    const modalHtml = `
        <div class="modal fade" id="createTriggerModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-bolt"></i> Create Trigger
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="createTriggerForm">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">Trigger Name</label>
                                        <input type="text" class="form-control" id="newTriggerName" required>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">Table</label>
                                        <select class="form-select" id="triggerTableName" required>
                                            <option value="">Select table...</option>
                                            ${tableOptions}
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Event</label>
                                <select class="form-select" id="triggerEvent" required>
                                    <option value="">Select event...</option>
                                    <option value="BEFORE INSERT">BEFORE INSERT</option>
                                    <option value="AFTER INSERT">AFTER INSERT</option>
                                    <option value="BEFORE UPDATE">BEFORE UPDATE</option>
                                    <option value="AFTER UPDATE">AFTER UPDATE</option>
                                    <option value="BEFORE DELETE">BEFORE DELETE</option>
                                    <option value="AFTER DELETE">AFTER DELETE</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">WHEN Condition (Optional)</label>
                                <input type="text" class="form-control" id="triggerWhen" 
                                       placeholder="e.g., NEW.column_name > OLD.column_name">
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Trigger Body</label>
                                <textarea class="form-control query-editor" id="triggerBody" rows="6" 
                                          placeholder="INSERT INTO audit_table (table_name, action, timestamp) VALUES ('table_name', 'INSERT', datetime('now'));" required></textarea>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="createNewTrigger()">Create Trigger</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('createTriggerModal'));
    modal.show();
}

async function createNewTrigger() {
    const triggerName = document.getElementById('newTriggerName').value.trim();
    const tableName = document.getElementById('triggerTableName').value;
    const event = document.getElementById('triggerEvent').value;
    const whenCondition = document.getElementById('triggerWhen').value.trim();
    const triggerBody = document.getElementById('triggerBody').value.trim();

    if (!triggerName) {
        showAlert('warning', 'Please enter a trigger name');
        return;
    }

    if (!tableName) {
        showAlert('warning', 'Please select a table');
        return;
    }

    if (!event) {
        showAlert('warning', 'Please select an event');
        return;
    }

    if (!triggerBody) {
        showAlert('warning', 'Please enter trigger body');
        return;
    }

    try {
        showLoading('Creating trigger...');
        const result = await DatabaseExplorerAPI.createTrigger(triggerName, tableName, event, triggerBody, whenCondition);

        if (result.success) {
            showAlert('success', result.message);

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createTriggerModal'));
            if (modal) {
                modal.hide();
            }
        } else {
            showAlert('danger', 'Failed to create trigger: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error creating trigger: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('success', 'Copied to clipboard');
    }).catch(() => {
        showAlert('warning', 'Could not copy to clipboard');
    });
}

// Final completion of Database Explorer Frontend with remaining utility functions

function handleChatKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    const text = document.getElementById('loadingText');

    if (overlay && text) {
        text.textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showAlert(type, message) {
    // Create alert element
    const alertEl = document.createElement('div');
    alertEl.className = `alert alert-${type} alert-dismissible fade show alert-custom`;
    alertEl.setAttribute('role', 'alert');

    // Determine icon based on type
    const icons = {
        success: 'fas fa-check-circle',
        danger: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };

    const icon = icons[type] || icons.info;

    alertEl.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="${icon} me-2"></i>
            <div>${message}</div>
        </div>
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    // Add to page
    document.body.appendChild(alertEl);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertEl.parentNode) {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alertEl);
            bsAlert.close();
        }
    }, 5000);
}

// Schema visualization function
function generateSchemaVisualization() {
    if (!isConnected || !currentTables || currentTables.length === 0) {
        document.getElementById('schemaVisualization').innerHTML = `
            <div class="text-muted text-center py-4">
                <i class="fas fa-sitemap fa-3x mb-3"></i>
                <p>Connect to a database to view schema</p>
            </div>
        `;
        return;
    }

    // Simple text-based schema visualization
    let html = '<div class="row">';

    currentTables.forEach((table, index) => {
        if (index % 3 === 0 && index > 0) {
            html += '</div><div class="row">';
        }

        html += `
            <div class="col-md-4 mb-3">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <i class="fas fa-table"></i> ${table.name}
                    </div>
                    <div class="card-body">
                        <small class="text-muted">
                            ${table.rows.toLocaleString()} rows<br>
                            ${table.columns} columns
                        </small>
                        <div class="mt-2">
                            <button class="btn btn-sm btn-outline-primary" onclick="showTableDetails('${table.name}')">
                                View Details
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';

    document.getElementById('schemaVisualization').innerHTML = html;
}

// Performance monitoring functions
async function analyzePerformance() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    const modelId = document.getElementById('aiModel').value;
    const lmStudioUrl = document.getElementById('lmStudioUrl').value;

    try {
        showLoading('Getting AI performance analysis...');

        // Get AI suggestions for performance
        const result = await DatabaseExplorerAPI.getSchemaExplanation('performance', modelId, lmStudioUrl);

        if (result.success) {
            addChatMessage('Here\'s my analysis of your database performance:', 'ai');
            addChatMessage(result.explanation, 'ai');
        } else {
            showAlert('danger', 'Failed to get AI performance analysis: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error getting performance analysis: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Export functions
async function exportAllData() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    try {
        showLoading('Exporting all data...');

        // Export each table as CSV
        for (const table of currentTables) {
            if (table.rows > 0) {
                const data = await DatabaseExplorerAPI.getTableData(table.name, table.rows);
                if (data.success && data.data.length > 0) {
                    await DatabaseExplorerAPI.exportCSV(data.data, `${table.name}.csv`);
                }
            }
        }

        showAlert('success', 'All data exported successfully');
    } catch (error) {
        showAlert('danger', 'Error exporting data: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function generateReport() {
    if (!isConnected) {
        showAlert('danger', 'No database connected');
        return;
    }

    try {
        showLoading('Generating database report...');

        // Get comprehensive stats
        const statsResult = await DatabaseExplorerAPI.getDatabaseStats();
        const perfResult = await DatabaseExplorerAPI.performanceAnalysis();
        const validationResult = await DatabaseExplorerAPI.validateData();

        // Create report content
        const reportData = {
            timestamp: new Date().toISOString(),
            stats: statsResult.success ? statsResult.stats : null,
            performance: perfResult.success ? perfResult.analysis : null,
            validation: validationResult.success ? validationResult.results : null
        };

        // Generate and download report
        const reportJson = JSON.stringify(reportData, null, 2);
        const blob = new Blob([reportJson], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `database_report_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showAlert('success', 'Database report generated and downloaded');
    } catch (error) {
        showAlert('danger', 'Error generating report: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Join table functionality
function showJoinTablesModal() {
    if (!isConnected || currentTables.length < 2) {
        showAlert('warning', 'Need at least 2 tables to perform JOIN');
        return;
    }

    const tableOptions = currentTables.map(table =>
        `<option value="${table.name}">${table.name}</option>`
    ).join('');

    const modalHtml = `
        <div class="modal fade" id="joinTablesModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-link"></i> Join Tables
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="joinTablesForm">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">Left Table</label>
                                        <select class="form-select" id="leftTable" required>
                                            <option value="">Select table...</option>
                                            ${tableOptions}
                                        </select>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">Right Table</label>
                                        <select class="form-select" id="rightTable" required>
                                            <option value="">Select table...</option>
                                            ${tableOptions}
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Join Type</label>
                                <select class="form-select" id="joinType">
                                    <option value="INNER">INNER JOIN</option>
                                    <option value="LEFT">LEFT JOIN</option>
                                    <option value="RIGHT">RIGHT JOIN</option>
                                    <option value="FULL OUTER">FULL OUTER JOIN</option>
                                    <option value="CROSS">CROSS JOIN</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Join Condition</label>
                                <input type="text" class="form-control" id="joinCondition" 
                                       placeholder="e.g., table1.id = table2.foreign_id" required>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Columns to Select</label>
                                <input type="text" class="form-control" id="joinColumns" 
                                       value="*" placeholder="e.g., t1.name, t2.description">
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">WHERE Clause (Optional)</label>
                                <input type="text" class="form-control" id="joinWhere" 
                                       placeholder="e.g., t1.status = 'active'">
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Limit</label>
                                <input type="number" class="form-control" id="joinLimit" value="100" min="1" max="10000">
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="executeJoin()">Execute JOIN</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('joinTablesModal'));
    modal.show();
}

async function executeJoin() {
    const leftTable = document.getElementById('leftTable').value;
    const rightTable = document.getElementById('rightTable').value;
    const joinType = document.getElementById('joinType').value;
    const joinCondition = document.getElementById('joinCondition').value.trim();
    const columns = document.getElementById('joinColumns').value.trim() || '*';
    const whereClause = document.getElementById('joinWhere').value.trim();
    const limit = parseInt(document.getElementById('joinLimit').value) || 100;

    if (!leftTable || !rightTable) {
        showAlert('warning', 'Please select both tables');
        return;
    }

    if (!joinCondition && joinType !== 'CROSS') {
        showAlert('warning', 'Please enter join condition');
        return;
    }

    try {
        showLoading('Executing JOIN...');

        const result = await DatabaseExplorerAPI.joinTables(
            leftTable, rightTable, joinCondition, columns, whereClause, limit, joinType
        );

        if (result.success) {
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('joinTablesModal'));
            if (modal) {
                modal.hide();
            }

            // Show results in query tab
            currentQueryResults = result.data;
            displayQueryResults(result.data, result.columns);
            updateVisualizationColumns(result.columns);

            document.getElementById('exportBtn').style.display = 'inline-block';
            document.getElementById('visualizeBtn').style.display = 'inline-block';
            document.getElementById('resultCount').textContent = `${result.row_count} rows`;
            document.getElementById('resultCount').style.display = 'inline-block';

            // Show the JOIN query that was executed
            document.getElementById('sqlQuery').value = result.query;

            showTab('query');
            showAlert('success', `JOIN executed successfully. ${result.row_count} rows returned.`);
        } else {
            showAlert('danger', 'JOIN failed: ' + result.error);
        }
    } catch (error) {
        showAlert('danger', 'Error executing JOIN: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Cleanup function to remove modals
function cleanupModals() {
    const modals = [
        'tableModal', 'sqlTemplatesModal', 'analysisModal', 'queryTemplatesModal',
        'createTableModal', 'createIndexModal', 'createViewModal', 'createTriggerModal',
        'joinTablesModal'
    ];

    modals.forEach(modalId => {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.remove();
        }
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+Enter to execute query
    if (event.ctrlKey && event.key === 'Enter') {
        const activeElement = document.activeElement;
        if (activeElement && activeElement.id === 'sqlQuery') {
            event.preventDefault();
            executeQuery();
        }
        if (activeElement && activeElement.id === 'chatInput') {
            event.preventDefault();
            sendChatMessage();
        }
    }

    // Ctrl+Shift+F to format query
    if (event.ctrlKey && event.shiftKey && event.key === 'F') {
        if (document.getElementById('sqlQuery') === document.activeElement) {
            event.preventDefault();
            formatQuery();
        }
    }

    // F5 to refresh tables
    if (event.key === 'F5' && isConnected) {
        event.preventDefault();
        refreshTables();
    }

    // Escape to clear query
    if (event.key === 'Escape') {
        const activeTab = document.querySelector('.tab-pane.active');
        if (activeTab && activeTab.id === 'query-tab') {
            clearQuery();
        }
    }
});

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add search functionality to table search input
    const tableSearchInput = document.getElementById('tableSearch');
    if (tableSearchInput) {
        tableSearchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchTables();
            }
        });

        // Add debounced search
        let searchTimeout;
        tableSearchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                if (this.value.trim() === '') {
                    updateTablesGrid(currentTables);
                }
            }, 300);
        });
    }

    // Initialize chat input event listener
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', handleChatKeyPress);
    }

    // Add table search enter key functionality
    const searchInput = document.getElementById('tableSearch');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchTables();
            }
        });
    }

    // Check if we're on localhost and try to connect to API
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        // Try to get API status
        DatabaseExplorerAPI.getApiStatus()
            .then(result => {
                if (result.status === 'healthy') {
                    console.log('✅ Database Explorer API is healthy');

                    // Load AI models
                    return DatabaseExplorerAPI.getAIModels();
                }
            })
            .then(aiResult => {
                if (aiResult && aiResult.success) {
                    console.log('✅ AI models loaded:', aiResult.models.length);

                    // Update AI model dropdown
                    const modelSelect = document.getElementById('aiModel');
                    if (modelSelect && aiResult.models) {
                        modelSelect.innerHTML = '';
                        aiResult.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.model_id;
                            option.textContent = model.name;
                            if (model.model_id === aiResult.default_model) {
                                option.selected = true;
                            }
                            modelSelect.appendChild(option);
                        });
                    }
                }
            })
            .catch(error => {
                console.warn('⚠️ Could not connect to Database Explorer API:', error.message);
            });
    }

    // Initialize schema visualization
    setTimeout(() => {
        generateSchemaVisualization();
    }, 1000);

    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        cleanupModals();
    });

    // Show welcome message
    console.log(`
🚀 Database Explorer Frontend Loaded
📊 Ready to explore your databases!
🤖 AI assistance available with LM Studio
⌨️  Keyboard shortcuts:
   • Ctrl+Enter: Execute query
   • Ctrl+Shift+F: Format SQL
   • F5: Refresh tables
   • Escape: Clear query
    `);
});

// Export main functions for global access
window.DatabaseExplorer = {
    API: DatabaseExplorerAPI,
    connectDatabase,
    executeQuery,
    showTableDetails,
    generateChart,
    sendChatMessage,
    testAIConnection,
    showTab,
    runValidation,
    runPerformanceAnalysis,
    exportResults,
    exportSchema,
    getDatabaseStats,
    showJoinTablesModal,
    refreshTables
};

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showAlert('danger', 'An unexpected error occurred. Check the console for details.');
});

// Global unhandled promise rejection handler
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showAlert('warning', 'A network request failed. Please check your connection.');
    event.preventDefault();
});

// Expose utility functions globally
window.showAlert = showAlert;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.copyToClipboard = copyToClipboard;