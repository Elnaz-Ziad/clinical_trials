from setuptools import setup

setup(
    name="clinical_trials",
    version="0.1.0",
    author="Elnaz Ziad",
    author_email="elnaz.ziad@mail.utoronto.ca",
    description="Data cleaning, analysis, and visualization tools for clinical trials research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/elnazziad/clinical_trials",
    packages=["src"],
    package_dir={"src": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "pandasql>=0.7.3",
        "mlxtend>=0.21.0",
        "networkx>=2.8.0",
        "jupyter>=1.0.0",
    ],
)
