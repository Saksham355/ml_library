from setuptools import setup, find_packages

setup(
    name="ml_library",
    version="0.1.0",
    description="An attempt to implement a simple ML library with PCA, LDA, QDA, Boosting, and Neural Net utilities",
    author="Saksham Bhardwaj",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.7",
)
