# setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'my_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    # Include all .py files in the package
    py_modules=[
        'coordinator.camera_interface',
        'coordinator.camera_interface_controller',
        'coordinator.grasp_detector',
        'coordinator.grasp_detector_controller',
        'coordinator.coordinator'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Camera Interface and Coordinator Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_interface = coordinator.camera_interface:main',
            'coordinator = coordinator.coordinator:main',
            # Add other nodes if they are executable scripts
        ],
    },
)
