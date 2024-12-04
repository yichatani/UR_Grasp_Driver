# app/setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'frames'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        
        # Add other data files if necessary
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy', 'scipy', 'open3d', 'torch'],
    zip_safe=True,
    maintainer='artc',
    maintainer_email='abdelrohman.atia@gmail.com',
    description='Coordinator and related nodes package',
    license='Apache License 2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coordinator2 = scripts.coordinator2:main',
            'grasp_detector_controller = scripts.grasp_detector_controller:main',
            'robot_controller_moveit = scripts.robot_controller_moveit:main',
            'robot_controller_rtde = scripts.robot_controller_rtde:main',
            'gripper_controller = scripts.gripper_controller:main',
            # Add other executables as needed
        ],
    },
)


