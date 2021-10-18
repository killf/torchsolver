from setuptools import setup, find_packages

setup(
    name="torchsolver",
    version="1.5",
    author="killf",
    author_email="killf@foxmail.com",
    description="A pytorch based deep learning solver framework.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/killf/torchsolver",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
