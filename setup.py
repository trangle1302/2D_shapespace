from setuptools import setup, find_packages

setup(
    name="2D_shapespace",  # Package name
    version="0.1.0",
    description="Analysis of cell line shapespace and organelle correlations",
    author="trangle1302",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn==1.4.2",
        "scikit-image==0.22.0",
        "seaborn==0.13.0",
        "opencv-python-headless==4.6.0.66",
        "more_itertools==10.7.0",
        "tqdm==4.67.1",
        "numpy==1.26.4",
        # Optional / experimental (not used in final version):
        # "pyefd",
        # "PyWavelets",
    ],
)
