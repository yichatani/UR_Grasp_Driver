from setuptools import setup, find_packages

setup(
    name="my_libs",
    version="0.0.2",
    description="My own custom libraries for my_ur_driver",
    author="Abdo",
    author_email="abdo@example.com",
    packages=find_packages(),
    install_requires=[
        'numpy','setuptools'
    ],
    zip_safe=False,
)

