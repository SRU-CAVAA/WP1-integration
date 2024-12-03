from setuptools import find_packages, setup

package_name = 'ca_webots_pkg'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))

data_files.append(('share/' + package_name + '/launch', [
    'launch/epuck_launch.py']))

data_files.append(('share/' + package_name + '/worlds', [
    'worlds/Open_arena.wbt',
    'worlds/Tmaze.wbt',
    'worlds/Double_Tmaze.wbt',
    'worlds/Linear_Track.wbt']))

data_files.append(('share/' + package_name + '/resource', [
    'resource/webots_epuck.urdf']))

data_files.append(('share/' + package_name + '/protos', [
    'protos/E-puck.proto',
    'protos/E-puckDistanceSensor.proto',
    'protos/HexagonalColorFloor.proto',
    'protos/LinearTrackFloor.proto',
    'protos/WallL.proto',
    'protos/WallR.proto',
    'protos/RewardBall.proto']))

data_files.append(('share/' + package_name + '/textures', [
    'textures/LinearTrack_floor.jpg',
    'textures/WallL.jpg',
    'textures/WallR.jpg',
    'textures/Webots_floor.jpg',
    'textures/Webots_floor2.jpg']))

data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oguerrero',
    maintainer_email='oscar3tri@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_gathering = ca_webots_pkg.data_gathering:main',
            'supervisor = ca_webots_pkg.supervisor:main',
            'experiment = ca_webots_pkg.experiment:main'
        ],
    },
)
