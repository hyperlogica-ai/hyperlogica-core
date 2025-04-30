from setuptools import setup, find_packages

setup(
    name="hyperlogica",
    version="0.2.0",
    description="ACEP-based vector reasoning system using hyperdimensional computing",
    author="Hyperlogica Team",
    author_email="info@hyperlogica.ai",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "faiss-cpu>=1.7.0",  # or faiss-gpu for GPU support
        "jsonschema>=4.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.6b0",
            "isort>=5.9.2",
            "mypy>=0.910",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
