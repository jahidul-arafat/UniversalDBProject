# setup.py (optional - for package installation)
from setuptools import setup, find_packages

setup(
    name="universal-db-explorer",
    version="1.0.0",
    description="Universal Database Explorer - Web Interface",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pandas>=1.5.0",
        "werkzeug>=2.3.0"
    ],
    python_requires=">=3.8",
    author="Universal DB Explorer Team",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)