from setuptools import setup, find_packages

setup(
    name="qwen25vl",  
    version="0.1.0",  
    description="A Python library for qwen25vl functionalities, incorporating modifications to qwen25vl with methods for KV cache compression. ", 
    packages=find_packages(),  
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',  
)
