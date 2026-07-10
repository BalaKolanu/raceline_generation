from glob import glob
from os.path import isfile, join

from setuptools import setup

package_name = 'raceline_generation'


def collect_files(*patterns):
    files = []
    for pattern in patterns:
        files.extend(path for path in glob(pattern) if isfile(path))
    return sorted(set(files))


data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml', 'README.md']),
]

resource_files = collect_files('resource/*.py')
if resource_files:
    data_files.append((join('share', package_name, 'resource'), resource_files))

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@example.com',
    description='ROS 2 package for raceline generation tools and nodes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'process_raceline = raceline_generation.process_raceline_node:main',
            'process_raceline_node = raceline_generation.process_raceline_node:main',
            'edit_raceline = raceline_generation.raceline_editor:main',
            'generate_lanes = raceline_generation.lane_generation_node:main',
            'lane_generation_node = raceline_generation.lane_generation_node:main',
        ],
    },
)
