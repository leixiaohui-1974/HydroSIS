from setuptools import setup, find_packages

setup(
    name="hydrosis",
    version="0.1.0",
    packages=find_packages(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A hydrological modeling framework.",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/HydroSIS",  # Replace with your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "GDAL>=3.2.0",
        "pyproj>=3.2.0",
        "Shapely>=1.7.0",
        "Fiona>=1.8.0",
        "rasterio>=1.2.0",
        "richdem>=0.3.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "jinja2>=3.0.0",
        "python-multipart>=0.0.5",
        "sqlalchemy>=1.4.0",
        "alembic>=1.7.0",
        "matplotlib>=3.5.0",
        "plotly>=5.5.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=5.4.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
    ]
)