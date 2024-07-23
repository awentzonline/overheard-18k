import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="overheard-18k",
    version="0.0.1",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    description="18k fake conversations for babies",
    long_description=read("README.md"),
    license="MIT",
    url="https://github.com/awentzonline/overheard-18k",
    packages=find_packages(),
    install_requires=[
        'anthropic',
        'click',
        'ray',
    ]
)
