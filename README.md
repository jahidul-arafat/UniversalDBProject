# Universal Database Explorer - Web Interface

A modern, browser-based database exploration tool built with Flask that provides an intuitive interface for exploring SQLite databases.

## Features

### 🖥️ **Web-Based Interface**
- Modern, responsive design using Bootstrap 5
- No desktop application installation required
- Works on any device with a web browser
- Real-time data visualization

### 📊 **Data Visualization**
- Interactive charts (Bar, Line, Pie, Histogram, Scatter)
- Automatic chart generation from query results
- Export charts and data
- Powered by Matplotlib and Seaborn

### 🔧 **Advanced Features**
- SQL query editor with syntax highlighting
- Table structure exploration
- Data validation and integrity checks
- Performance analysis and recommendations
- CSV export functionality
- Sample query templates

### 🚀 **Easy to Use**
- Drag-and-drop database file upload
- One-click table exploration
- Interactive table browsing
- Search functionality across tables

## Installation

1. **Clone or download the files:**
   ```bash
   # You need these files in the same directory:
   # - db_explorer.py (your existing CLI tool)
   # - web_db_explorer.py (the web interface)
   # - run.py (startup script)
   ```

2. **Install dependencies:**
   ```bash
   pip install flask matplotlib seaborn pandas
   ```

3. **Run the application:**
   ```bash
   python run.py
   # OR
   python web_db_explorer.py
   ```

4. **Open your browser:**
   ```
   http://localhost:5000
   ```

## Usage

### Getting Started
1. **Upload Database**: Click "Choose File" and select your SQLite database
2. **Connect**: Click the "Connect" button to establish connection
3. **Explore**: Browse tables in the sidebar or use the main interface tabs

### Main Interface Tabs

#### 📋 **Overview Tab**
- Database information and statistics
- Tables overview with row counts
- Quick statistics dashboard

#### 📝 **SQL Query Tab**
- Write and execute custom SQL queries
- View results in formatted tables
- Export results to CSV
- Quick query templates

#### 📊 **Visualization Tab**
- Create charts from query results
- Multiple chart types available
- Interactive column selection
- Export charts as images

#### 🔧 **Tools Tab**
- Search tables by keyword
- Data validation checks
- Performance analysis
- Export database schema

### Table Exploration
- Click any table in the sidebar to view details
- View table structure, data, and indexes
- Generate SQL queries for tables
- Export individual table data

## API Endpoints

The web interface provides a REST API:

- `POST /api/connect` - Connect to database
- `GET /api/database-info` - Get database information
- `GET /api/tables` - List all tables
- `GET /api/table/<name>/structure` - Get table structure
- `GET /api/table/<name>/data` - Get table data
- `POST /api/execute-query` - Execute SQL query
- `POST /api/visualize` - Create data visualization
- `GET /api/validate-data` - Run data validation
- `GET /api/performance-analysis` - Analyze performance

## File Structure

```
your-project/
├── db_explorer.py          # Core database exploration logic
├── web_db_explorer.py      # Flask web application
├── run.py                  # Simple startup script
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates (auto-generated)
│   ├── base.html
│   └── index.html
└── static/                 # Static files (auto-generated)
```

## Browser Compatibility

- Chrome/Chromium 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Performance Notes

- Large result sets are automatically limited for display performance
- Visualizations are limited to 100 data points for optimal rendering
- File upload limit: 100MB
- Recommended for databases up to 1GB for optimal performance

## Security Considerations

- This is designed for local development/analysis use
- Do not expose to public networks without additional security measures
- Database files are temporarily stored on the server
- No authentication system included by default

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `db_explorer.py` is in the same directory
2. **Missing Dependencies**: Install required packages with pip
3. **Large Files**: Ensure database is under 100MB file size limit
4. **Port Conflicts**: Change port in `web_db_explorer.py` if 5000 is in use

### Debug Mode
The application runs in debug mode by default for development. Set `app.config['DEBUG'] = False` for production use.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use and modify as needed.