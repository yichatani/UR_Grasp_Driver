from setuptools import find_packages, setup

package_name = 'diffusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ani',
    maintainer_email='yichatma@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'record_numpy = diffusion.record_numpy:main',
            # 'load_numpy = diffusion.load_numpy:main',
            # 'pre_pointcloud = diffusion.pre_pointcloud:main',
            # 'RRT_path = diffusion.RRT_path:main',
            # 'RRT*_path = diffusion.RRT*_path:main',
            # '6D_RRT* = diffusion.6D_RRT*:main',
        ],
    },
)
