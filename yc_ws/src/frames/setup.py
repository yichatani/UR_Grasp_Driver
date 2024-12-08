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
            'robot_grasp_coordinator = frames.robot_grasp_coordinator:main',
            'grasp_detector = frames.grasp_detector:main',
            'camera_controller = frames.camera_controller:main',
            'grasp_pose_transformer = frames.grasp_pose_transformer:main',
            'gripper_controller = frames.gripper_controller:main',
            'myframe = frames.myframe:main',
            # Add other executables as needed
        ],
    },
)


