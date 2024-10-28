from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ChromaPy",
    version="1.0.0", 
    author="Caleb Coatney", 
    author_email="Caleb.Coatney@nrel.gov", 
    description="A customtkinter app for processing gas chromatography data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/calebcoatney/ChromaPy",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.9.2",
        "customtkinter==5.2.2",
        "pandas==2.2.3",
        "numpy==2.1.2",
        "tabulate==0.9.0",
        "scipy==1.14.1",
        "openpyxl==3.1.5"
    ],
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "chromapy=chromapy.main:main", 
        ],
    },
    license="MIT",
    keywords="gas chromatography, data processing, customtkinter",
)
