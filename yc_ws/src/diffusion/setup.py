# diffusion/setup.py
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'diffusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ani',
    maintainer_email='yichatma@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'record_numpy = diffusion.record_numpy:main',
            'load_numpy = diffusion.load_numpy:main',
            'pre_pointcloud = diffusion.pre_pointcloud:main',
            'rrt_path = diffusion.rrt_path:main',
            'rrt_star_path = diffusion.rrt_star_path:main',
            '6d_rrt_star = diffusion.6d_rrt_star:main',
        ],
    },
)
