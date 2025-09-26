#!/usr/bin/env python3
"""
Setup script for RAG Assistant
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A RAG-powered question answering assistant using Groq and LangChain",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "rag-cli=cli_rag:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="rag retrieval-augmented-generation ai langchain groq nlp",
)