# run.py - Simple startup script
#!/usr/bin/env python3
"""
Simple startup script for Universal Database Explorer Web Interface
"""

import sys
import os

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from web_db_explorer import main
    main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required files are in the same directory:")
    print("  - db_explorer.py")
    print("  - web_db_explorer.py")
    print("  - run.py")
    print()
    print("Install required packages with:")
    print("  pip install flask matplotlib seaborn pandas")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n👋 Goodbye!")
    sys.exit(0)