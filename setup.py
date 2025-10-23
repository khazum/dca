from setuptools import setup

setup(
    name="DCA",
    version="0.3.4",
    description="Count autoencoder for scRNA-seq denoising - modernized implementation (Tensorflow==2.20.0) of the package DCA(https://github.com/theislab/dca)",
    author="Marcin Malec",
    author_email="khazum@gmail.com",
    packages=["dca"],
    install_requires=[
        "tensorflow[and-cuda]>=2.20.0",
        "numpy",
        "h5py",
        "six",
        "scikit-learn",
        "scanpy",
        "keras-tuner",
        "pandas",
    ],
    url="https://github.com/KHAZUM/dca",
    entry_points={"console_scripts": ["dca = dca.__main__:main"]},
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)
