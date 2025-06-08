#!/usr/bin/env python3
"""
GUI Database Explorer - Graphical interface for Universal Database Explorer
Usage: python gui_db_explorer.py
"""

import sys
import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the core functionality from db_explorer
try:
    from db_explorer import UniversalDatabaseExplorer, DatabaseSchema, TableInfo
except ImportError:
    print("❌ Error: db_explorer.py not found. Please ensure db_explorer.py is in the same directory.")
    sys.exit(1)

# GUI imports
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    import tkinter.font as tkFont
except ImportError:
    print("❌ Error: tkinter not available. Please install tkinter.")
    sys.exit(1)

# Data visualization imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: matplotlib not available. Data visualization will be limited.")
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: pandas not available. Some features will be limited.")
    PANDAS_AVAILABLE = False

class DataVisualizer:
    """Handle data visualization using matplotlib"""

    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.figure = None
        self.canvas = None

    def create_chart_frame(self):
        """Create the chart display frame"""
        if self.canvas:
            self.canvas.destroy()

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.parent_frame)
        self.canvas.draw()
        return self.canvas.get_tk_widget()

    def plot_data(self, data: List[Dict[str, Any]], chart_type: str = "bar",
                  x_column: str = None, y_column: str = None, title: str = "Data Visualization"):
        """Plot data using matplotlib"""
        if not MATPLOTLIB_AVAILABLE or not data:
            return None

        try:
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Convert data to appropriate format
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(data)
            else:
                # Manual data processing without pandas
                if not x_column or not y_column:
                    # Auto-detect columns
                    columns = list(data[0].keys())
                    x_column = columns[0] if len(columns) > 0 else None
                    y_column = columns[1] if len(columns) > 1 else columns[0]

                x_values = [row.get(x_column, '') for row in data]
                y_values = [row.get(y_column, 0) for row in data]

                # Try to convert y_values to numeric
                numeric_y = []
                for val in y_values:
                    try:
                        numeric_y.append(float(val) if val != '' else 0)
                    except (ValueError, TypeError):
                        numeric_y.append(0)
                y_values = numeric_y

            # Create different chart types
            if chart_type == "bar":
                if PANDAS_AVAILABLE:
                    df.plot(x=x_column, y=y_column, kind='bar', ax=ax)
                else:
                    ax.bar(range(len(x_values)), y_values)
                    ax.set_xticks(range(len(x_values)))
                    ax.set_xticklabels(x_values, rotation=45)
                    ax.set_ylabel(y_column)

            elif chart_type == "line":
                if PANDAS_AVAILABLE:
                    df.plot(x=x_column, y=y_column, kind='line', ax=ax)
                else:
                    ax.plot(y_values)
                    ax.set_ylabel(y_column)

            elif chart_type == "pie":
                if PANDAS_AVAILABLE:
                    df.set_index(x_column)[y_column].plot(kind='pie', ax=ax)
                else:
                    ax.pie(y_values, labels=x_values, autopct='%1.1f%%')

            elif chart_type == "histogram":
                if PANDAS_AVAILABLE:
                    df[y_column].plot(kind='hist', ax=ax, bins=20)
                else:
                    ax.hist(y_values, bins=20)
                    ax.set_xlabel(y_column)
                    ax.set_ylabel('Frequency')

            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            self.figure.tight_layout()

            if self.canvas:
                self.canvas.draw()

            return self.canvas.get_tk_widget()

        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None

class QueryResultsViewer:
    """Display query results in a tree view with export options"""

    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.tree = None
        self.current_data = []

    def create_results_frame(self):
        """Create the results display frame"""
        # Create frame for results
        results_frame = ttk.Frame(self.parent_frame)

        # Create treeview for results
        self.tree = ttk.Treeview(results_frame, show='headings')

        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(results_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack scrollbars and treeview
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)

        return results_frame

    def display_results(self, data: List[Dict[str, Any]], title: str = "Query Results"):
        """Display results in the tree view"""
        if not data or not self.tree:
            return

        self.current_data = data

        # Clear existing data
        self.tree.delete(*self.tree.get_children())

        # Configure columns
        columns = list(data[0].keys())
        self.tree["columns"] = columns

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, minwidth=100)

        # Insert data
        for i, row in enumerate(data):
            values = [str(row.get(col, '')) for col in columns]
            self.tree.insert("", "end", values=values)

    def export_to_csv(self, filename: str = None):
        """Export current results to CSV"""
        if not self.current_data:
            return False

        if not filename:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    if self.current_data:
                        # Write header
                        headers = list(self.current_data[0].keys())
                        f.write(','.join(headers) + '\n')

                        # Write data
                        for row in self.current_data:
                            values = []
                            for value in row.values():
                                if value is None:
                                    values.append('')
                                elif isinstance(value, str):
                                    if ',' in value or '"' in value:
                                        escaped_value = value.replace('"', '""')
                                        values.append(f'"{escaped_value}"')
                                    else:
                                        values.append(value)
                                else:
                                    values.append(str(value))
                            f.write(','.join(values) + '\n')
                return True
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {e}")
                return False
        return False

class DatabaseGUI:
    """Main GUI application for database exploration"""

    def __init__(self):
        self.root = tk.Tk()
        self.explorer = UniversalDatabaseExplorer()
        self.connected_db = None
        self.current_schema = None

        # GUI Components
        self.results_viewer = None
        self.visualizer = None

        # Setup GUI
        self.setup_gui()
        self.create_menu()
        self.create_main_interface()

    def setup_gui(self):
        """Initialize the main GUI window"""
        self.root.title("🗃️ Universal Database Explorer - GUI")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 600)

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        self.root.configure(bg='#f0f0f0')

        # Icon (if available)
        try:
            self.root.iconbitmap('database_icon.ico')
        except:
            pass

    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Database...", command=self.open_database, accelerator="Ctrl+O")
        file_menu.add_command(label="Close Database", command=self.close_database)
        file_menu.add_separator()
        file_menu.add_command(label="Export Schema...", command=self.export_schema)
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Validation", command=self.run_data_validation)
        tools_menu.add_command(label="Performance Analysis", command=self.run_performance_analysis)
        tools_menu.add_command(label="Schema Information", command=self.show_schema_info)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Schema", command=self.refresh_schema)
        view_menu.add_command(label="Clear Results", command=self.clear_results)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="SQL Reference", command=self.show_sql_reference)

        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.open_database())
        self.root.bind('<F5>', lambda e: self.refresh_schema())

    def create_main_interface(self):
        """Create the main application interface"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Database structure
        self.create_left_panel(main_paned)

        # Right panel - Query and results
        self.create_right_panel(main_paned)

        # Status bar
        self.create_status_bar()

    def create_left_panel(self, parent):
        """Create the left panel with database structure"""
        left_frame = ttk.Frame(parent)
        parent.add(left_frame, weight=1)

        # Database info frame
        info_frame = ttk.LabelFrame(left_frame, text="📁 Database Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.db_info_label = ttk.Label(info_frame, text="No database connected",
                                       font=('Arial', 10))
        self.db_info_label.pack(anchor=tk.W)

        # Tables tree
        tables_frame = ttk.LabelFrame(left_frame, text="📊 Database Structure", padding=5)
        tables_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create treeview for tables
        self.tables_tree = ttk.Treeview(tables_frame, show='tree headings')
        self.tables_tree["columns"] = ("Type", "Rows", "Info")
        self.tables_tree.heading("#0", text="Name")
        self.tables_tree.heading("Type", text="Type")
        self.tables_tree.heading("Rows", text="Rows")
        self.tables_tree.heading("Info", text="Info")

        # Column widths
        self.tables_tree.column("#0", width=150)
        self.tables_tree.column("Type", width=80)
        self.tables_tree.column("Rows", width=80)
        self.tables_tree.column("Info", width=100)

        # Scrollbar for tables tree
        tables_scrollbar = ttk.Scrollbar(tables_frame, orient="vertical",
                                         command=self.tables_tree.yview)
        self.tables_tree.configure(yscrollcommand=tables_scrollbar.set)

        # Pack tree and scrollbar
        self.tables_tree.pack(side="left", fill="both", expand=True)
        tables_scrollbar.pack(side="right", fill="y")

        # Bind events
        self.tables_tree.bind("<Double-1>", self.on_table_double_click)
        self.tables_tree.bind("<Button-3>", self.on_table_right_click)

        # Quick actions frame
        actions_frame = ttk.LabelFrame(left_frame, text="🔧 Quick Actions", padding=5)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(actions_frame, text="📋 View Table Data",
                   command=self.view_selected_table).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="🏗️ Table Structure",
                   command=self.show_table_structure).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="🔍 Search Tables",
                   command=self.search_tables_dialog).pack(fill=tk.X, pady=2)

    def create_right_panel(self, parent):
        """Create the right panel with query interface and results"""
        right_frame = ttk.Frame(parent)
        parent.add(right_frame, weight=3)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Query tab
        self.create_query_tab()

        # Results tab
        self.create_results_tab()

        # Visualization tab
        if MATPLOTLIB_AVAILABLE:
            self.create_visualization_tab()

        # Schema tab
        self.create_schema_tab()

    def create_query_tab(self):
        """Create the SQL query tab"""
        query_frame = ttk.Frame(self.notebook)
        self.notebook.add(query_frame, text="📝 SQL Query")

        # Query input frame
        input_frame = ttk.LabelFrame(query_frame, text="SQL Query", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # SQL text area
        self.sql_text = scrolledtext.ScrolledText(input_frame, height=10,
                                                  font=('Consolas', 11))
        self.sql_text.pack(fill=tk.BOTH, expand=True)

        # Add syntax highlighting (basic)
        self.setup_sql_syntax_highlighting()

        # Buttons frame
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(fill=tk.X, pady=5)

        ttk.Button(buttons_frame, text="🚀 Execute Query",
                   command=self.execute_query, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="🧹 Clear",
                   command=lambda: self.sql_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="📋 Sample Queries",
                   command=self.show_sample_queries).pack(side=tk.LEFT, padx=5)

        # Quick query buttons
        quick_frame = ttk.LabelFrame(query_frame, text="Quick Queries", padding=5)
        quick_frame.pack(fill=tk.X, padx=5, pady=5)

        quick_buttons = [
            ("Show All Tables", "SELECT name FROM sqlite_master WHERE type='table';"),
            ("Database Info", "PRAGMA database_list;"),
            ("Table Count", "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"),
        ]

        for i, (text, query) in enumerate(quick_buttons):
            ttk.Button(quick_frame, text=text,
                       command=lambda q=query: self.insert_query(q)).grid(row=0, column=i, padx=5, sticky="ew")

        # Configure grid weights
        for i in range(len(quick_buttons)):
            quick_frame.columnconfigure(i, weight=1)

    def create_results_tab(self):
        """Create the results display tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Results")

        # Results viewer
        self.results_viewer = QueryResultsViewer(results_frame)
        results_widget = self.results_viewer.create_results_frame()
        results_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results info frame
        info_frame = ttk.Frame(results_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.results_info_label = ttk.Label(info_frame, text="No results")
        self.results_info_label.pack(side=tk.LEFT)

        ttk.Button(info_frame, text="📤 Export to CSV",
                   command=self.export_results).pack(side=tk.RIGHT, padx=5)
        ttk.Button(info_frame, text="📈 Visualize",
                   command=self.visualize_results).pack(side=tk.RIGHT, padx=5)

    def create_visualization_tab(self):
        """Create the data visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📈 Visualization")

        # Controls frame
        controls_frame = ttk.LabelFrame(viz_frame, text="Chart Options", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Chart type selection
        ttk.Label(controls_frame, text="Chart Type:").grid(row=0, column=0, sticky="w", padx=5)
        self.chart_type_var = tk.StringVar(value="bar")
        chart_combo = ttk.Combobox(controls_frame, textvariable=self.chart_type_var,
                                   values=["bar", "line", "pie", "histogram"], state="readonly")
        chart_combo.grid(row=0, column=1, sticky="ew", padx=5)

        # Column selection
        ttk.Label(controls_frame, text="X Column:").grid(row=0, column=2, sticky="w", padx=5)
        self.x_column_var = tk.StringVar()
        self.x_column_combo = ttk.Combobox(controls_frame, textvariable=self.x_column_var, state="readonly")
        self.x_column_combo.grid(row=0, column=3, sticky="ew", padx=5)

        ttk.Label(controls_frame, text="Y Column:").grid(row=1, column=0, sticky="w", padx=5)
        self.y_column_var = tk.StringVar()
        self.y_column_combo = ttk.Combobox(controls_frame, textvariable=self.y_column_var, state="readonly")
        self.y_column_combo.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Button(controls_frame, text="📊 Generate Chart",
                   command=self.generate_chart).grid(row=1, column=2, columnspan=2, sticky="ew", padx=5, pady=5)

        # Configure grid weights
        for i in range(4):
            controls_frame.columnconfigure(i, weight=1)

        # Chart display frame
        chart_frame = ttk.LabelFrame(viz_frame, text="Chart Display", padding=5)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.visualizer = DataVisualizer(chart_frame)

    def create_schema_tab(self):
        """Create the schema information tab"""
        schema_frame = ttk.Frame(self.notebook)
        self.notebook.add(schema_frame, text="🏗️ Schema")

        # Schema tree
        self.schema_tree = ttk.Treeview(schema_frame, show='tree headings')
        self.schema_tree["columns"] = ("Type", "Details")
        self.schema_tree.heading("#0", text="Name")
        self.schema_tree.heading("Type", text="Type")
        self.schema_tree.heading("Details", text="Details")

        # Scrollbars
        schema_v_scroll = ttk.Scrollbar(schema_frame, orient="vertical",
                                        command=self.schema_tree.yview)
        schema_h_scroll = ttk.Scrollbar(schema_frame, orient="horizontal",
                                        command=self.schema_tree.xview)
        self.schema_tree.configure(yscrollcommand=schema_v_scroll.set,
                                   xscrollcommand=schema_h_scroll.set)

        # Pack schema tree
        self.schema_tree.pack(side="left", fill="both", expand=True)
        schema_v_scroll.pack(side="right", fill="y")
        schema_h_scroll.pack(side="bottom", fill="x")

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(self.status_bar, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.connection_label = ttk.Label(self.status_bar, text="No database", relief=tk.SUNKEN)
        self.connection_label.pack(side=tk.RIGHT)

    def setup_sql_syntax_highlighting(self):
        """Setup basic SQL syntax highlighting"""
        # Configure text tags for syntax highlighting
        self.sql_text.tag_configure("keyword", foreground="blue", font=('Consolas', 11, 'bold'))
        self.sql_text.tag_configure("string", foreground="green")
        self.sql_text.tag_configure("comment", foreground="gray")

        # SQL keywords
        self.sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP',
            'ALTER', 'TABLE', 'INDEX', 'VIEW', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'GROUP', 'ORDER', 'BY', 'HAVING', 'AS', 'AND', 'OR', 'NOT', 'NULL', 'IS',
            'LIKE', 'IN', 'BETWEEN', 'EXISTS', 'UNION', 'ALL', 'DISTINCT', 'COUNT',
            'SUM', 'AVG', 'MAX', 'MIN', 'LIMIT', 'OFFSET'
        ]

    def highlight_sql_syntax(self, event=None):
        """Apply basic SQL syntax highlighting"""
        content = self.sql_text.get("1.0", tk.END)

        # Clear existing tags
        self.sql_text.tag_remove("keyword", "1.0", tk.END)

        # Highlight keywords
        for keyword in self.sql_keywords:
            start = "1.0"
            while True:
                pos = self.sql_text.search(keyword, start, tk.END, nocase=True)
                if not pos:
                    break
                end = f"{pos}+{len(keyword)}c"
                self.sql_text.tag_add("keyword", pos, end)
                start = end

    def open_database(self):
        """Open a database file"""
        file_path = filedialog.askopenfilename(
            title="Open Database",
            filetypes=[
                ("SQLite databases", "*.db *.sqlite *.sqlite3"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.connect_to_database(file_path)

    def connect_to_database(self, db_path: str):
        """Connect to the specified database"""
        try:
            self.update_status("Connecting to database...")

            # Determine database type
            db_type = 'sqlite'  # Currently only supporting SQLite

            # Connect using the explorer
            if self.explorer.connect(db_type, {'database': db_path}):
                self.connected_db = db_path
                self.current_schema = self.explorer.schema

                # Update GUI
                self.update_database_info()
                self.populate_tables_tree()
                self.populate_schema_tree()

                self.update_status(f"Connected to: {os.path.basename(db_path)}")
                self.connection_label.config(text=f"📁 {os.path.basename(db_path)}")

                messagebox.showinfo("Success", f"Successfully connected to {os.path.basename(db_path)}")
            else:
                messagebox.showerror("Connection Error", "Failed to connect to database")
                self.update_status("Connection failed")

        except Exception as e:
            messagebox.showerror("Error", f"Error connecting to database: {e}")
            self.update_status("Connection error")

    def close_database(self):
        """Close the current database connection"""
        if self.explorer:
            self.explorer.close()

        self.connected_db = None
        self.current_schema = None

        # Clear GUI
        self.clear_tables_tree()
        self.clear_schema_tree()
        self.clear_results()

        self.db_info_label.config(text="No database connected")
        self.connection_label.config(text="No database")
        self.update_status("Database closed")

    def update_database_info(self):
        """Update the database information display"""
        if not self.current_schema:
            return

        info_text = f"Database: {os.path.basename(self.connected_db)}\n"
        info_text += f"Type: {self.current_schema.database_type}\n"
        info_text += f"Tables: {len(self.current_schema.tables)}\n"
        info_text += f"Views: {len(self.current_schema.views)}\n"
        info_text += f"Triggers: {len(self.current_schema.triggers)}\n"
        info_text += f"Indexes: {len(self.current_schema.indexes)}"

        self.db_info_label.config(text=info_text)

    def populate_tables_tree(self):
        """Populate the tables tree view"""
        if not self.current_schema:
            return

        # Clear existing items
        self.clear_tables_tree()

        # Add tables
        for table in self.current_schema.tables:
            item_id = self.tables_tree.insert("", "end", text=table.name,
                                              values=("Table", f"{table.row_count:,}",
                                                      f"{len(table.columns)} cols"))

            # Add columns as children
            for column in table.columns:
                col_info = f"{column['type']}"
                if not column['nullable']:
                    col_info += " NOT NULL"
                if column['name'] in table.primary_keys:
                    col_info += " PK"

                self.tables_tree.insert(item_id, "end", text=column['name'],
                                        values=("Column", "", col_info))

        # Add views
        for view in self.current_schema.views:
            self.tables_tree.insert("", "end", text=view,
                                    values=("View", "-", ""))

    def populate_schema_tree(self):
        """Populate the schema tree view"""
        if not self.current_schema:
            return

        # Clear existing items
        self.clear_schema_tree()

        # Add tables section
        tables_node = self.schema_tree.insert("", "end", text="Tables",
                                              values=("Section", f"{len(self.current_schema.tables)} items"))

        for table in self.current_schema.tables:
            table_node = self.schema_tree.insert(tables_node, "end", text=table.name,
                                                 values=("Table", f"{table.row_count} rows, {len(table.columns)} columns"))

            # Columns
            cols_node = self.schema_tree.insert(table_node, "end", text="Columns",
                                                values=("Section", f"{len(table.columns)} items"))
            for column in table.columns:
                col_detail = f"{column['type']}"
                if not column['nullable']:
                    col_detail += ", NOT NULL"
                if column['default']:
                    col_detail += f", DEFAULT {column['default']}"

                self.schema_tree.insert(cols_node, "end", text=column['name'],
                                        values=("Column", col_detail))

            # Indexes
            if table.indexes:
                idx_node = self.schema_tree.insert(table_node, "end", text="Indexes",
                                                   values=("Section", f"{len(table.indexes)} items"))
                for index in table.indexes:
                    idx_detail = f"Columns: {', '.join(index.get('columns', []))}"
                    if index.get('unique'):
                        idx_detail += ", UNIQUE"
                    self.schema_tree.insert(idx_node, "end", text=index['name'],
                                            values=("Index", idx_detail))

            # Foreign Keys
            if table.foreign_keys:
                fk_node = self.schema_tree.insert(table_node, "end", text="Foreign Keys",
                                                  values=("Section", f"{len(table.foreign_keys)} items"))
                for fk in table.foreign_keys:
                    fk_detail = f"{fk['column']} → {fk['referenced_table']}.{fk['referenced_column']}"
                    self.schema_tree.insert(fk_node, "end", text=f"FK_{fk['column']}",
                                            values=("Foreign Key", fk_detail))

        # Add views section
        if self.current_schema.views:
            views_node = self.schema_tree.insert("", "end", text="Views",
                                                 values=("Section", f"{len(self.current_schema.views)} items"))
            for view in self.current_schema.views:
                self.schema_tree.insert(views_node, "end", text=view,
                                        values=("View", ""))

        # Add triggers section
        if self.current_schema.triggers:
            triggers_node = self.schema_tree.insert("", "end", text="Triggers",
                                                    values=("Section", f"{len(self.current_schema.triggers)} items"))
            for trigger in self.current_schema.triggers:
                self.schema_tree.insert(triggers_node, "end", text=trigger['name'],
                                        values=("Trigger", ""))

    def clear_tables_tree(self):
        """Clear the tables tree view"""
        if self.tables_tree:
            self.tables_tree.delete(*self.tables_tree.get_children())

    def clear_schema_tree(self):
        """Clear the schema tree view"""
        if self.schema_tree:
            self.schema_tree.delete(*self.schema_tree.get_children())

    def clear_results(self):
        """Clear the results display"""
        if self.results_viewer and self.results_viewer.tree:
            self.results_viewer.tree.delete(*self.results_viewer.tree.get_children())
            self.results_viewer.current_data = []
        self.results_info_label.config(text="No results")

    def on_table_double_click(self, event):
        """Handle double-click on table in tree"""
        selection = self.tables_tree.selection()
        if selection:
            item = self.tables_tree.item(selection[0])
            if item['values'] and item['values'][0] == "Table":
                table_name = item['text']
                self.view_table_data(table_name)

    def on_table_right_click(self, event):
        """Handle right-click on table in tree"""
        selection = self.tables_tree.selection()
        if selection:
            item = self.tables_tree.item(selection[0])
            if item['values'] and item['values'][0] == "Table":
                # Create context menu
                context_menu = tk.Menu(self.root, tearoff=0)
                table_name = item['text']

                context_menu.add_command(label="View Data",
                                         command=lambda: self.view_table_data(table_name))
                context_menu.add_command(label="Show Structure",
                                         command=lambda: self.show_table_structure_for(table_name))
                context_menu.add_command(label="Count Rows",
                                         command=lambda: self.count_table_rows(table_name))
                context_menu.add_separator()
                context_menu.add_command(label="Generate SELECT",
                                         command=lambda: self.generate_select_query(table_name))

                try:
                    context_menu.tk_popup(event.x_root, event.y_root)
                finally:
                    context_menu.grab_release()

    def view_selected_table(self):
        """View data for the selected table"""
        selection = self.tables_tree.selection()
        if selection:
            item = self.tables_tree.item(selection[0])
            if item['values'] and item['values'][0] == "Table":
                table_name = item['text']
                self.view_table_data(table_name)
            else:
                messagebox.showwarning("Selection Error", "Please select a table")
        else:
            messagebox.showwarning("Selection Error", "Please select a table")

    def view_table_data(self, table_name: str, limit: int = 100):
        """View data for a specific table"""
        if not self.explorer or not self.connected_db:
            messagebox.showerror("Error", "No database connected")
            return

        try:
            self.update_status(f"Loading data from {table_name}...")

            # Get table data
            data = self.explorer.get_sample_data(table_name, limit)

            if data:
                # Display in results viewer
                self.results_viewer.display_results(data, f"Data from {table_name}")
                self.results_info_label.config(text=f"Showing {len(data)} rows from {table_name}")

                # Update column comboboxes for visualization
                self.update_column_comboboxes(data)

                # Switch to results tab
                self.notebook.select(1)  # Results tab

                self.update_status(f"Loaded {len(data)} rows from {table_name}")
            else:
                messagebox.showinfo("No Data", f"Table '{table_name}' contains no data")
                self.update_status("No data found")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading table data: {e}")
            self.update_status("Error loading data")

    def show_table_structure(self):
        """Show structure for the selected table"""
        selection = self.tables_tree.selection()
        if selection:
            item = self.tables_tree.item(selection[0])
            if item['values'] and item['values'][0] == "Table":
                table_name = item['text']
                self.show_table_structure_for(table_name)
            else:
                messagebox.showwarning("Selection Error", "Please select a table")
        else:
            messagebox.showwarning("Selection Error", "Please select a table")

    def show_table_structure_for(self, table_name: str):
        """Show detailed structure for a specific table"""
        if not self.explorer or not self.connected_db:
            messagebox.showerror("Error", "No database connected")
            return

        try:
            structure = self.explorer.get_table_structure(table_name)
            if structure:
                # Create a new window for table structure
                struct_window = tk.Toplevel(self.root)
                struct_window.title(f"📊 Table Structure: {table_name}")
                struct_window.geometry("800x600")

                # Create notebook for different aspects
                struct_notebook = ttk.Notebook(struct_window)
                struct_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Columns tab
                self.create_columns_tab(struct_notebook, structure)

                # Constraints tab
                self.create_constraints_tab(struct_notebook, structure)

                # Indexes tab
                self.create_indexes_tab(struct_notebook, structure)

                # Statistics tab
                self.create_statistics_tab(struct_notebook, structure)

            else:
                messagebox.showerror("Error", f"Could not retrieve structure for table '{table_name}'")

        except Exception as e:
            messagebox.showerror("Error", f"Error showing table structure: {e}")

    def create_columns_tab(self, notebook, structure):
        """Create the columns tab in table structure window"""
        columns_frame = ttk.Frame(notebook)
        notebook.add(columns_frame, text="📋 Columns")

        # Create treeview for columns
        columns_tree = ttk.Treeview(columns_frame, show='headings')
        columns_tree["columns"] = ("Name", "Type", "Nullable", "Default", "Primary Key")

        for col in columns_tree["columns"]:
            columns_tree.heading(col, text=col)
            columns_tree.column(col, width=120)

        # Populate columns
        for column in structure['columns']:
            values = (
                column['name'],
                column['type'],
                "Yes" if column['nullable'] else "No",
                column['default'] or "",
                "Yes" if column['name'] in structure['primary_keys'] else "No"
            )
            columns_tree.insert("", "end", values=values)

        # Add scrollbar
        columns_scrollbar = ttk.Scrollbar(columns_frame, orient="vertical", command=columns_tree.yview)
        columns_tree.configure(yscrollcommand=columns_scrollbar.set)

        columns_tree.pack(side="left", fill="both", expand=True)
        columns_scrollbar.pack(side="right", fill="y")

    def create_constraints_tab(self, notebook, structure):
        """Create the constraints tab in table structure window"""
        constraints_frame = ttk.Frame(notebook)
        notebook.add(constraints_frame, text="🔗 Constraints")

        # Primary Keys
        pk_frame = ttk.LabelFrame(constraints_frame, text="Primary Keys", padding=10)
        pk_frame.pack(fill=tk.X, padx=10, pady=5)

        if structure['primary_keys']:
            pk_text = ", ".join(structure['primary_keys'])
        else:
            pk_text = "No primary keys"
        ttk.Label(pk_frame, text=pk_text, font=('Arial', 10, 'bold')).pack(anchor=tk.W)

        # Foreign Keys
        fk_frame = ttk.LabelFrame(constraints_frame, text="Foreign Keys", padding=10)
        fk_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        if structure['foreign_keys']:
            fk_tree = ttk.Treeview(fk_frame, show='headings')
            fk_tree["columns"] = ("Column", "References Table", "References Column")

            for col in fk_tree["columns"]:
                fk_tree.heading(col, text=col)
                fk_tree.column(col, width=150)

            for fk in structure['foreign_keys']:
                fk_tree.insert("", "end", values=(
                    fk['column'],
                    fk['referenced_table'],
                    fk['referenced_column']
                ))

            fk_tree.pack(fill="both", expand=True)
        else:
            ttk.Label(fk_frame, text="No foreign keys").pack()

    def create_indexes_tab(self, notebook, structure):
        """Create the indexes tab in table structure window"""
        indexes_frame = ttk.Frame(notebook)
        notebook.add(indexes_frame, text="🔍 Indexes")

        if structure['indexes']:
            indexes_tree = ttk.Treeview(indexes_frame, show='headings')
            indexes_tree["columns"] = ("Name", "Columns", "Unique", "Type")

            for col in indexes_tree["columns"]:
                indexes_tree.heading(col, text=col)
                indexes_tree.column(col, width=150)

            for index in structure['indexes']:
                columns_str = ", ".join(index.get('columns', []))
                unique_str = "Yes" if index.get('unique') else "No"

                indexes_tree.insert("", "end", values=(
                    index['name'],
                    columns_str,
                    unique_str,
                    "Index"
                ))

            indexes_tree.pack(fill="both", expand=True, padx=10, pady=10)
        else:
            ttk.Label(indexes_frame, text="No indexes found",
                      font=('Arial', 12)).pack(expand=True)

    def create_statistics_tab(self, notebook, structure):
        """Create the statistics tab in table structure window"""
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📊 Statistics")

        # Basic statistics
        basic_frame = ttk.LabelFrame(stats_frame, text="Basic Statistics", padding=10)
        basic_frame.pack(fill=tk.X, padx=10, pady=5)

        stats_text = f"""
        Table Name: {structure['name']}
        Total Rows: {structure['row_count']:,}
        Total Columns: {len(structure['columns'])}
        Primary Keys: {len(structure['primary_keys'])}
        Foreign Keys: {len(structure['foreign_keys'])}
        Indexes: {len(structure['indexes'])}
        Triggers: {len(structure['triggers'])}
        """

        ttk.Label(basic_frame, text=stats_text, font=('Consolas', 10)).pack(anchor=tk.W)

        # Column statistics (if possible)
        if structure['row_count'] > 0:
            col_stats_frame = ttk.LabelFrame(stats_frame, text="Column Analysis", padding=10)
            col_stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            # Try to get column statistics
            try:
                numeric_columns = [col['name'] for col in structure['columns']
                                   if any(t in col['type'].upper() for t in ['INT', 'REAL', 'NUMERIC', 'FLOAT'])]

                if numeric_columns:
                    stats_tree = ttk.Treeview(col_stats_frame, show='headings')
                    stats_tree["columns"] = ("Column", "Min", "Max", "Avg")

                    for col in stats_tree["columns"]:
                        stats_tree.heading(col, text=col)
                        stats_tree.column(col, width=100)

                    # Get statistics for numeric columns (limit to first 3)
                    for col_name in numeric_columns[:3]:
                        try:
                            stats_query = f"""
                                SELECT 
                                    MIN({col_name}) as min_val,
                                    MAX({col_name}) as max_val,
                                    AVG({col_name}) as avg_val
                                FROM {structure['name']}
                                WHERE {col_name} IS NOT NULL
                            """
                            stats_result = self.explorer.execute_query(stats_query)

                            if stats_result and stats_result[0]:
                                min_val = stats_result[0].get('min_val', 'N/A')
                                max_val = stats_result[0].get('max_val', 'N/A')
                                avg_val = stats_result[0].get('avg_val', 'N/A')

                                if avg_val != 'N/A':
                                    avg_val = f"{float(avg_val):.2f}"

                                stats_tree.insert("", "end", values=(col_name, min_val, max_val, avg_val))
                        except:
                            pass  # Skip columns that cause errors

                    stats_tree.pack(fill="both", expand=True)
                else:
                    ttk.Label(col_stats_frame, text="No numeric columns found for analysis").pack()

            except Exception as e:
                ttk.Label(col_stats_frame, text=f"Error analyzing columns: {e}").pack()

    def search_tables_dialog(self):
        """Show dialog to search tables"""
        if not self.connected_db:
            messagebox.showerror("Error", "No database connected")
            return

        # Create search dialog
        search_dialog = tk.Toplevel(self.root)
        search_dialog.title("🔍 Search Tables")
        search_dialog.geometry("400x300")
        search_dialog.transient(self.root)
        search_dialog.grab_set()

        # Search frame
        search_frame = ttk.Frame(search_dialog)
        search_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(search_frame, text="Search keyword:").pack(anchor=tk.W)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(fill=tk.X, pady=5)

        # Results frame
        results_frame = ttk.LabelFrame(search_dialog, text="Search Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        results_listbox = tk.Listbox(results_frame)
        results_listbox.pack(fill=tk.BOTH, expand=True)

        def perform_search():
            keyword = search_var.get().strip()
            if keyword:
                try:
                    matching_tables = self.explorer.search_tables(keyword)
                    results_listbox.delete(0, tk.END)

                    if matching_tables:
                        for table in matching_tables:
                            results_listbox.insert(tk.END, table)
                    else:
                        results_listbox.insert(tk.END, "No matching tables found")
                except Exception as e:
                    messagebox.showerror("Search Error", f"Error searching tables: {e}")

        def on_table_select(event):
            selection = results_listbox.curselection()
            if selection:
                table_name = results_listbox.get(selection[0])
                if table_name != "No matching tables found":
                    search_dialog.destroy()
                    self.view_table_data(table_name)

        # Bind events
        search_entry.bind('<Return>', lambda e: perform_search())
        results_listbox.bind('<Double-1>', on_table_select)

        # Buttons
        button_frame = ttk.Frame(search_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="Search", command=perform_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=search_dialog.destroy).pack(side=tk.RIGHT, padx=5)

        search_entry.focus()

    def count_table_rows(self, table_name: str):
        """Count rows in a specific table"""
        try:
            result = self.explorer.execute_query(f"SELECT COUNT(*) as row_count FROM {table_name}")
            if result and result[0]:
                count = result[0]['row_count']
                messagebox.showinfo("Row Count", f"Table '{table_name}' has {count:,} rows")
        except Exception as e:
            messagebox.showerror("Error", f"Error counting rows: {e}")

    def generate_select_query(self, table_name: str):
        """Generate a SELECT query for the table"""
        if not self.explorer or not self.connected_db:
            return

        try:
            structure = self.explorer.get_table_structure(table_name)
            if structure:
                columns = [col['name'] for col in structure['columns']]
                query = f"SELECT {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''} FROM {table_name} LIMIT 10;"

                # Insert into SQL text area
                self.sql_text.delete(1.0, tk.END)
                self.sql_text.insert(1.0, query)

                # Switch to query tab
                self.notebook.select(0)  # Query tab
        except Exception as e:
            messagebox.showerror("Error", f"Error generating query: {e}")

    def execute_query(self):
        """Execute the SQL query"""
        if not self.explorer or not self.connected_db:
            messagebox.showerror("Error", "No database connected")
            return

        query = self.sql_text.get(1.0, tk.END).strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a SQL query")
            return

        try:
            self.update_status("Executing query...")

            # Execute query in a separate thread to prevent GUI freezing
            def execute_in_thread():
                try:
                    results = self.explorer.execute_query(query)

                    # Update GUI in main thread
                    self.root.after(0, lambda: self.handle_query_results(results, query))

                except Exception as e:
                    self.root.after(0, lambda: self.handle_query_error(e))

            thread = threading.Thread(target=execute_in_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.handle_query_error(e)

    def handle_query_results(self, results, query):
        """Handle query results in the main thread"""
        try:
            if results:
                if 'error' in results[0]:
                    messagebox.showerror("Query Error", results[0]['error'])
                    self.update_status("Query failed")
                elif query.upper().strip().startswith('SELECT'):
                    # Display SELECT results
                    self.results_viewer.display_results(results, "Query Results")
                    self.results_info_label.config(text=f"Query returned {len(results)} rows")

                    # Update column comboboxes for visualization
                    self.update_column_comboboxes(results)

                    # Switch to results tab
                    self.notebook.select(1)  # Results tab

                    self.update_status(f"Query executed successfully. {len(results)} rows returned.")
                else:
                    # Handle non-SELECT queries
                    affected_rows = results[0].get('affected_rows', 0)
                    messagebox.showinfo("Query Success", f"Query executed successfully. {affected_rows} rows affected.")
                    self.update_status(f"Query executed. {affected_rows} rows affected.")

                    # Refresh schema if it might have changed
                    if any(keyword in query.upper() for keyword in ['CREATE', 'DROP', 'ALTER']):
                        self.refresh_schema()
            else:
                messagebox.showinfo("Query Result", "Query executed successfully. No results returned.")
                self.update_status("Query executed successfully")

        except Exception as e:
            self.handle_query_error(e)

    def handle_query_error(self, error):
        """Handle query execution errors"""
        messagebox.showerror("Query Error", f"Error executing query: {error}")
        self.update_status("Query execution failed")

    def insert_query(self, query: str):
        """Insert a query into the SQL text area"""
        self.sql_text.delete(1.0, tk.END)
        self.sql_text.insert(1.0, query)

    def show_sample_queries(self):
        """Show sample SQL queries dialog"""
        sample_dialog = tk.Toplevel(self.root)
        sample_dialog.title("📋 Sample SQL Queries")
        sample_dialog.geometry("600x500")
        sample_dialog.transient(self.root)

        # Create notebook for different categories
        sample_notebook = ttk.Notebook(sample_dialog)
        sample_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Basic queries
        basic_queries = [
            ("Show all tables", "SELECT name FROM sqlite_master WHERE type='table';"),
            ("Database info", "PRAGMA database_list;"),
            ("Table info", "PRAGMA table_info(table_name);"),
            ("Count all rows", "SELECT name, (SELECT COUNT(*) FROM table_name) as count FROM sqlite_master WHERE type='table';"),
        ]

        # Advanced queries
        advanced_queries = [
            ("Find duplicate rows", "SELECT column1, column2, COUNT(*) FROM table_name GROUP BY column1, column2 HAVING COUNT(*) > 1;"),
            ("Table sizes", "SELECT name, COUNT(*) as rows FROM sqlite_master m LEFT JOIN table_name t GROUP BY name;"),
            ("Foreign key check", "PRAGMA foreign_key_check;"),
            ("Index usage", "EXPLAIN QUERY PLAN SELECT * FROM table_name WHERE condition;"),
        ]

        # Analysis queries
        analysis_queries = [
            ("Column statistics", "SELECT MIN(column), MAX(column), AVG(column) FROM table_name;"),
            ("Null value count", "SELECT COUNT(*) - COUNT(column) as null_count FROM table_name;"),
            ("Distinct values", "SELECT COUNT(DISTINCT column) FROM table_name;"),
            ("Data distribution", "SELECT column, COUNT(*) FROM table_name GROUP BY column ORDER BY COUNT(*) DESC;"),
        ]

        def create_query_tab(parent, title, queries):
            frame = ttk.Frame(parent)
            parent.add(frame, text=title)

            # Create listbox
            listbox = tk.Listbox(frame)
            listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Populate listbox
            for description, query in queries:
                listbox.insert(tk.END, description)

            # Store queries as attribute
            listbox.queries = {i: query for i, (desc, query) in enumerate(queries)}

            def on_select(event):
                selection = listbox.curselection()
                if selection:
                    query = listbox.queries[selection[0]]
                    self.insert_query(query)
                    sample_dialog.destroy()

            listbox.bind('<Double-1>', on_select)

            # Add button
            ttk.Button(frame, text="Use Selected Query",
                       command=lambda: on_select(None) if listbox.curselection() else None).pack(pady=5)

        create_query_tab(sample_notebook, "Basic", basic_queries)
        create_query_tab(sample_notebook, "Advanced", advanced_queries)
        create_query_tab(sample_notebook, "Analysis", analysis_queries)

    def update_column_comboboxes(self, data):
        """Update column comboboxes for visualization"""
        if data and MATPLOTLIB_AVAILABLE and hasattr(self, 'x_column_combo'):
            columns = list(data[0].keys())
            self.x_column_combo['values'] = columns
            self.y_column_combo['values'] = columns

            # Set default values
            if len(columns) > 0:
                self.x_column_var.set(columns[0])
            if len(columns) > 1:
                self.y_column_var.set(columns[1])
            elif len(columns) > 0:
                self.y_column_var.set(columns[0])

    def visualize_results(self):
        """Switch to visualization tab and prepare for charting"""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("Feature Unavailable",
                                   "Data visualization requires matplotlib. Please install matplotlib to use this feature.")
            return

        if not self.results_viewer.current_data:
            messagebox.showwarning("No Data", "No data available for visualization. Execute a query first.")
            return

        # Switch to visualization tab
        self.notebook.select(2)  # Visualization tab

        # Update column choices
        self.update_column_comboboxes(self.results_viewer.current_data)

        # Automatically generate a chart if possible
        if self.results_viewer.current_data:
            self.generate_chart()

    def generate_chart(self):
        """Generate a chart based on selected options"""
        if not MATPLOTLIB_AVAILABLE or not self.visualizer:
            return

        if not self.results_viewer.current_data:
            messagebox.showwarning("No Data", "No data available for visualization")
            return

        try:
            chart_type = self.chart_type_var.get()
            x_column = self.x_column_var.get()
            y_column = self.y_column_var.get()

            if not x_column or not y_column:
                messagebox.showwarning("Column Selection", "Please select columns for X and Y axes")
                return

            # Limit data for performance
            data_to_plot = self.results_viewer.current_data[:100]  # Limit to 100 rows

            title = f"{chart_type.title()} Chart: {y_column} by {x_column}"

            # Create chart
            chart_widget = self.visualizer.plot_data(
                data_to_plot,
                chart_type,
                x_column,
                y_column,
                title
            )

            if chart_widget:
                # Clear previous chart
                for widget in self.visualizer.parent_frame.winfo_children():
                    widget.destroy()

                chart_widget.pack(fill=tk.BOTH, expand=True)
                self.update_status(f"Generated {chart_type} chart")
            else:
                messagebox.showerror("Chart Error", "Failed to generate chart")

        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error generating chart: {e}")

    def export_results(self):
        """Export current results to CSV"""
        if not self.results_viewer or not self.results_viewer.current_data:
            messagebox.showwarning("No Data", "No results to export")
            return

        try:
            if self.results_viewer.export_to_csv():
                messagebox.showinfo("Export Success", "Results exported successfully")
            else:
                messagebox.showwarning("Export Cancelled", "Export operation was cancelled")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results: {e}")

    def export_schema(self):
        """Export database schema"""
        if not self.connected_db or not self.explorer:
            messagebox.showerror("Error", "No database connected")
            return

        try:
            filename = filedialog.asksaveasfilename(
                title="Export Schema",
                defaultextension=".sql",
                filetypes=[("SQL files", "*.sql"), ("Text files", "*.txt"), ("All files", "*.*")]
            )

            if filename:
                # Use the explorer's export functionality
                self.update_status("Exporting schema...")

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"-- Database Schema Export\n")
                    f.write(f"-- Source: {self.connected_db}\n")
                    f.write(f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    if self.current_schema:
                        # Export tables
                        for table in self.current_schema.tables:
                            f.write(f"-- Table: {table.name}\n")
                            f.write(f"-- Rows: {table.row_count}\n")
                            f.write(f"-- Columns: {len(table.columns)}\n\n")

                            # Basic CREATE TABLE structure
                            f.write(f"CREATE TABLE {table.name} (\n")

                            column_defs = []
                            for col in table.columns:
                                col_def = f"    {col['name']} {col['type']}"
                                if not col['nullable']:
                                    col_def += " NOT NULL"
                                if col['default']:
                                    col_def += f" DEFAULT {col['default']}"
                                column_defs.append(col_def)

                            if table.primary_keys:
                                pk_def = f"    PRIMARY KEY ({', '.join(table.primary_keys)})"
                                column_defs.append(pk_def)

                            f.write(',\n'.join(column_defs))
                            f.write("\n);\n\n")

                messagebox.showinfo("Export Success", f"Schema exported to {filename}")
                self.update_status("Schema exported successfully")

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting schema: {e}")
            self.update_status("Schema export failed")

    def run_data_validation(self):
        """Run data validation checks"""
        if not self.connected_db or not self.explorer:
            messagebox.showerror("Error", "No database connected")
            return

        # Create validation results window
        validation_window = tk.Toplevel(self.root)
        validation_window.title("🧪 Data Validation Results")
        validation_window.geometry("800x600")

        # Create text widget for results
        results_text = scrolledtext.ScrolledText(validation_window, font=('Consolas', 10))
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def run_validation():
            results_text.insert(tk.END, "🧪 RUNNING DATA VALIDATION CHECKS\n")
            results_text.insert(tk.END, "=" * 50 + "\n\n")

            try:
                # Check for empty tables
                results_text.insert(tk.END, "1. Checking for empty tables...\n")
                empty_tables = []

                if self.current_schema:
                    for table in self.current_schema.tables:
                        if table.row_count == 0:
                            empty_tables.append(table.name)

                if empty_tables:
                    results_text.insert(tk.END, f"   ⚠️  Found {len(empty_tables)} empty tables:\n")
                    for table in empty_tables:
                        results_text.insert(tk.END, f"      • {table}\n")
                else:
                    results_text.insert(tk.END, "   ✅ No empty tables found\n")

                results_text.insert(tk.END, "\n")

                # Check referential integrity
                results_text.insert(tk.END, "2. Checking referential integrity...\n")
                integrity_issues = 0

                for table in self.current_schema.tables:
                    if table.foreign_keys:
                        for fk in table.foreign_keys:
                            try:
                                query = f"""
                                    SELECT COUNT(*) as orphaned_count
                                    FROM {table.name} t1
                                    LEFT JOIN {fk['referenced_table']} t2 
                                    ON t1.{fk['column']} = t2.{fk['referenced_column']}
                                    WHERE t1.{fk['column']} IS NOT NULL 
                                    AND t2.{fk['referenced_column']} IS NULL
                                """

                                result = self.explorer.execute_query(query)
                                if result and result[0].get('orphaned_count', 0) > 0:
                                    integrity_issues += 1
                                    results_text.insert(tk.END, f"   ❌ {table.name}.{fk['column']} → {fk['referenced_table']}.{fk['referenced_column']}\n")
                                    results_text.insert(tk.END, f"      Orphaned records: {result[0]['orphaned_count']}\n")
                            except:
                                pass

                if integrity_issues == 0:
                    results_text.insert(tk.END, "   ✅ No referential integrity issues found\n")
                else:
                    results_text.insert(tk.END, f"   ⚠️  Found {integrity_issues} referential integrity issues\n")

                results_text.insert(tk.END, "\n")

                # Check for large tables without indexes
                results_text.insert(tk.END, "3. Checking indexing recommendations...\n")
                recommendations = 0

                for table in self.current_schema.tables:
                    if table.row_count > 1000 and len(table.indexes) == 0:
                        recommendations += 1
                        results_text.insert(tk.END, f"   💡 Consider adding indexes to large table: {table.name} ({table.row_count:,} rows)\n")

                    # Check foreign keys without indexes
                    unindexed_fks = []
                    for fk in table.foreign_keys:
                        fk_col = fk['column']
                        has_index = any(fk_col in idx.get('columns', []) for idx in table.indexes)
                        if not has_index:
                            unindexed_fks.append(fk_col)

                    if unindexed_fks:
                        recommendations += 1
                        results_text.insert(tk.END, f"   💡 Consider indexing foreign keys in {table.name}: {', '.join(unindexed_fks)}\n")

                if recommendations == 0:
                    results_text.insert(tk.END, "   ✅ No indexing issues found\n")

                results_text.insert(tk.END, "\n" + "=" * 50 + "\n")
                results_text.insert(tk.END, "✅ Data validation completed!\n")

            except Exception as e:
                results_text.insert(tk.END, f"❌ Error during validation: {e}\n")

            # Auto-scroll to bottom
            results_text.see(tk.END)

        # Run validation in thread
        thread = threading.Thread(target=run_validation)
        thread.daemon = True
        thread.start()

    def run_performance_analysis(self):
        """Run performance analysis"""
        if not self.connected_db or not self.explorer:
            messagebox.showerror("Error", "No database connected")
            return

        # Create performance analysis window
        perf_window = tk.Toplevel(self.root)
        perf_window.title("📈 Performance Analysis")
        perf_window.geometry("800x600")

        # Create notebook for different analysis types
        perf_notebook = ttk.Notebook(perf_window)
        perf_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Table sizes analysis
        sizes_frame = ttk.Frame(perf_notebook)
        perf_notebook.add(sizes_frame, text="📊 Table Sizes")

        sizes_tree = ttk.Treeview(sizes_frame, show='headings')
        sizes_tree["columns"] = ("Table", "Rows", "Columns", "Indexes", "Est. Size")

        for col in sizes_tree["columns"]:
            sizes_tree.heading(col, text=col)
            sizes_tree.column(col, width=120)

        # Populate table sizes
        if self.current_schema:
            tables_by_size = sorted(self.current_schema.tables, key=lambda x: x.row_count, reverse=True)

            for table in tables_by_size:
                est_size = table.row_count * len(table.columns) * 50  # Rough estimate
                est_size_mb = est_size / (1024 * 1024)

                sizes_tree.insert("", "end", values=(
                    table.name,
                    f"{table.row_count:,}",
                    len(table.columns),
                    len(table.indexes),
                    f"{est_size_mb:.2f} MB"
                ))

        sizes_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Index analysis
        index_frame = ttk.Frame(perf_notebook)
        perf_notebook.add(index_frame, text="🔍 Index Analysis")

        index_text = scrolledtext.ScrolledText(index_frame, font=('Consolas', 10))
        index_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Populate index analysis
        index_text.insert(tk.END, "INDEX ANALYSIS REPORT\n")
        index_text.insert(tk.END, "=" * 40 + "\n\n")

        if self.current_schema:
            total_indexes = sum(len(table.indexes) for table in self.current_schema.tables)
            index_text.insert(tk.END, f"Total Indexes: {total_indexes}\n\n")

            for table in self.current_schema.tables:
                index_text.insert(tk.END, f"Table: {table.name}\n")
                index_text.insert(tk.END, f"Rows: {table.row_count:,}\n")
                index_text.insert(tk.END, f"Indexes: {len(table.indexes)}\n")

                if table.indexes:
                    for idx in table.indexes:
                        columns_str = ', '.join(idx.get('columns', []))
                        unique_str = " (UNIQUE)" if idx.get('unique') else ""
                        index_text.insert(tk.END, f"  • {idx['name']}: {columns_str}{unique_str}\n")
                else:
                    index_text.insert(tk.END, "  No indexes\n")

                # Recommendations
                if table.row_count > 1000 and len(table.indexes) == 0:
                    index_text.insert(tk.END, "  💡 Recommendation: Add indexes for better performance\n")

                index_text.insert(tk.END, "\n")

    def show_schema_info(self):
        """Show detailed schema information"""
        if not self.current_schema:
            messagebox.showerror("Error", "No database connected")
            return

        # Create schema info window
        info_window = tk.Toplevel(self.root)
        info_window.title("🏗️ Schema Information")
        info_window.geometry("600x500")

        # Create text widget for schema info
        info_text = scrolledtext.ScrolledText(info_window, font=('Consolas', 10))
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Populate schema information
        info_text.insert(tk.END, "DATABASE SCHEMA INFORMATION\n")
        info_text.insert(tk.END, "=" * 50 + "\n\n")

        info_text.insert(tk.END, f"Database: {os.path.basename(self.connected_db)}\n")
        info_text.insert(tk.END, f"Type: {self.current_schema.database_type}\n")
        info_text.insert(tk.END, f"Path: {self.connected_db}\n\n")

        # File size
        try:
            file_size = os.path.getsize(self.connected_db)
            size_mb = file_size / (1024 * 1024)
            info_text.insert(tk.END, f"File Size: {size_mb:.2f} MB ({file_size:,} bytes)\n\n")
        except:
            pass

        # Summary statistics
        total_rows = sum(table.row_count for table in self.current_schema.tables)
        total_columns = sum(len(table.columns) for table in self.current_schema.tables)

        info_text.insert(tk.END, "SUMMARY STATISTICS:\n")
        info_text.insert(tk.END, f"Tables: {len(self.current_schema.tables)}\n")
        info_text.insert(tk.END, f"Views: {len(self.current_schema.views)}\n")
        info_text.insert(tk.END, f"Triggers: {len(self.current_schema.triggers)}\n")
        info_text.insert(tk.END, f"Indexes: {len(self.current_schema.indexes)}\n")
        info_text.insert(tk.END, f"Total Rows: {total_rows:,}\n")
        info_text.insert(tk.END, f"Total Columns: {total_columns}\n\n")

        # Table details
        info_text.insert(tk.END, "TABLE DETAILS:\n")
        info_text.insert(tk.END, "-" * 30 + "\n")

        for table in self.current_schema.tables:
            info_text.insert(tk.END, f"\n{table.name}:\n")
            info_text.insert(tk.END, f"  Rows: {table.row_count:,}\n")
            info_text.insert(tk.END, f"  Columns: {len(table.columns)}\n")
            info_text.insert(tk.END, f"  Primary Keys: {', '.join(table.primary_keys) if table.primary_keys else 'None'}\n")
            info_text.insert(tk.END, f"  Foreign Keys: {len(table.foreign_keys)}\n")
            info_text.insert(tk.END, f"  Indexes: {len(table.indexes)}\n")
            info_text.insert(tk.END, f"  Triggers: {len(table.triggers)}\n")

    def refresh_schema(self):
        """Refresh the database schema"""
        if not self.explorer or not self.connected_db:
            return

        try:
            self.update_status("Refreshing schema...")
            self.current_schema = self.explorer.connector.get_schema()
            self.update_database_info()
            self.populate_tables_tree()
            self.populate_schema_tree()
            self.update_status("Schema refreshed")
        except Exception as e:
            messagebox.showerror("Error", f"Error refreshing schema: {e}")
            self.update_status("Schema refresh failed")

    def show_about(self):
        """Show about dialog"""
        about_text = """
🗃️ Universal Database Explorer - GUI

A comprehensive database exploration tool with:
• Interactive GUI interface
• SQL query execution
• Data visualization
• Schema analysis
• Performance monitoring
• Data validation

Built with Python and Tkinter
Visualization powered by Matplotlib

Version: 1.0
"""
        messagebox.showinfo("About", about_text)

    def show_sql_reference(self):
        """Show SQL reference dialog"""
        ref_window = tk.Toplevel(self.root)
        ref_window.title("📚 SQL Reference")
        ref_window.geometry("700x600")

        ref_notebook = ttk.Notebook(ref_window)
        ref_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Basic SQL
        basic_frame = ttk.Frame(ref_notebook)
        ref_notebook.add(basic_frame, text="Basic SQL")

        basic_text = scrolledtext.ScrolledText(basic_frame, font=('Consolas', 10))
        basic_text.pack(fill=tk.BOTH, expand=True)

        basic_sql = """
BASIC SQL COMMANDS:

SELECT - Retrieve data
  SELECT column1, column2 FROM table_name;
  SELECT * FROM table_name WHERE condition;
  SELECT COUNT(*) FROM table_name;

INSERT - Add data
  INSERT INTO table_name (col1, col2) VALUES (val1, val2);

UPDATE - Modify data
  UPDATE table_name SET column1 = value WHERE condition;

DELETE - Remove data
  DELETE FROM table_name WHERE condition;

COMMON FUNCTIONS:
  COUNT(*) - Count rows
  SUM(column) - Sum values
  AVG(column) - Average value
  MIN(column) - Minimum value
  MAX(column) - Maximum value

FILTERING:
  WHERE column = value
  WHERE column LIKE 'pattern%'
  WHERE column IN (val1, val2, val3)
  WHERE column BETWEEN val1 AND val2
  WHERE column IS NULL / IS NOT NULL

SORTING:
  ORDER BY column ASC/DESC
  ORDER BY column1, column2

GROUPING:
  GROUP BY column
  HAVING condition (used with GROUP BY)

LIMITING:
  LIMIT number
  LIMIT number OFFSET number
"""

        basic_text.insert(tk.END, basic_sql)
        basic_text.config(state=tk.DISABLED)

        # Advanced SQL
        advanced_frame = ttk.Frame(ref_notebook)
        ref_notebook.add(advanced_frame, text="Advanced SQL")

        advanced_text = scrolledtext.ScrolledText(advanced_frame, font=('Consolas', 10))
        advanced_text.pack(fill=tk.BOTH, expand=True)

        advanced_sql = """
ADVANCED SQL COMMANDS:

JOINS:
  INNER JOIN - Records in both tables
    SELECT * FROM table1 
    INNER JOIN table2 ON table1.id = table2.id;
  
  LEFT JOIN - All records from left table
    SELECT * FROM table1 
    LEFT JOIN table2 ON table1.id = table2.id;

SUBQUERIES:
  SELECT * FROM table1 
  WHERE column IN (SELECT column FROM table2);

WINDOW FUNCTIONS:
  SELECT column, ROW_NUMBER() OVER (ORDER BY column) 
  FROM table_name;

COMMON TABLE EXPRESSIONS (CTE):
  WITH cte_name AS (
    SELECT column FROM table_name WHERE condition
  )
  SELECT * FROM cte_name;

CASE STATEMENTS:
  SELECT column,
    CASE 
      WHEN condition1 THEN value1
      WHEN condition2 THEN value2
      ELSE default_value
    END as new_column
  FROM table_name;

DATA DEFINITION:
  CREATE TABLE table_name (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
  );
  
  CREATE INDEX idx_name ON table_name (column);
  DROP TABLE table_name;
  ALTER TABLE table_name ADD COLUMN new_col TEXT;

SQLITE SPECIFIC:
  PRAGMA table_info(table_name);
  PRAGMA foreign_keys = ON;
  PRAGMA database_list;
"""

        advanced_text.insert(tk.END, advanced_sql)
        advanced_text.config(state=tk.DISABLED)

    def update_status(self, message: str):
        """Update the status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def run(self):
        """Start the GUI application"""
        try:
            # Center the window
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')

            # Show startup message
            self.update_status("Ready - Open a database to get started")

            # Start the main loop
            self.root.mainloop()

        except Exception as e:
            messagebox.showerror("Application Error", f"Error starting application: {e}")
        finally:
            # Cleanup
            if self.explorer:
                self.explorer.close()

def main():
    """Main function to start the GUI application"""
    try:
        # Create and run the GUI
        app = DatabaseGUI()
        app.run()

    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("\nTo install required packages:")
        print("pip install tkinter matplotlib pandas")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error starting GUI application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()