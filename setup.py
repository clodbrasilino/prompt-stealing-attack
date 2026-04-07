"""
Setup script for prompt-stealing-attack.
Enables: pip install git+https://github.com/clodbrasilino/prompt-stealing-attack.git
"""
from setuptools import setup, find_packages

setup(
    name="prompt-stealing-attack",
    version="0.1.0",
    description="Prompt Stealing Attacks Against Text-to-Image Generation Models",
    author="Clodoaldo Brasilino",
    author_email="clodbrasilino@zju.edu.cn",
    url="https://github.com/clodbrasilino/prompt-stealing-attack",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "BLIP_finetune": ["configs/*.yaml"],
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "timm>=0.4.12",
        "transformers>=4.15.0",
        "fairscale>=0.4.4",
        "ruamel.yaml>=0.17.32",
        "pycocoevalcap",
        "ftfy",
        "regex",
        "tqdm",
        "opencv-python-headless",
        "scikit-learn",
        "imagehash",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
    ],
)
