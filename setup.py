from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='gym_simpletetris',
    version='0.2.1',
    author="Tristan Russell",
    license='MIT',
    description="A simple Tetris engine for OpenAI Gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tristanrussell/gym-simpletetris",
    project_urls={
        "Bug Tracker": "https://github.com/tristanrussell/gym-simpletetris/issues",
    },
    install_requires=['gym>=0.21.0',
                      'numpy>=0.21.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
)
