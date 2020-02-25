import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SeisSrcMoment",
    version="0.0.1",
    author="Tom Hudson",
    author_email="thomas.hudson@earth.ox.ac.uk",
    description="A package for calculating the seismic moment of an earthquake using a time domain approximation of the spectral method.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomSHudson/SeisSrcMoment/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
