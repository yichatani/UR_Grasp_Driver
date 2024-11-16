from setuptools import setup, find_packages

setup(
    name="my_libs",
    version="0.0.2",
    description="My own custom libraries for my_ur_driver",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        # List any dependencies here, e.g.,
        numpy,
    ],
    zip_safe=False,
)

