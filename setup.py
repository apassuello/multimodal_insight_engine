from setuptools import setup, find_packages

setup(
    name="multimodal_insight_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0",
        "pytest-cov>=4.0",
    ],
    python_requires=">=3.8",
)
