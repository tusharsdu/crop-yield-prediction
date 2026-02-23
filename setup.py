from setuptools import setup, find_packages

setup(
    name="crop-yield-prediction",
    version="1.0.0",
    description="Climate-Resilient Crop Yield Prediction using IoT & Machine Learning",
    author="AAI-530 Group 4",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dl": ["tensorflow>=2.11.0"],
        "boost": ["xgboost>=1.7.0"],
        "notebook": ["jupyter>=1.0.0", "ipykernel>=6.0.0"],
        "all": ["tensorflow>=2.11.0", "xgboost>=1.7.0",
                "jupyter>=1.0.0", "plotly>=5.13.0"],
    },
    entry_points={
        "console_scripts": [
            "crop-yield=main:main",
        ]
    },
)
