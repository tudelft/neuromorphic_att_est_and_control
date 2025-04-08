from setuptools import setup, find_packages

setup(
    name='crazyflie-snn',
    version='0.1.0',
    packages=find_packages(include=['crazyflie_snn', 
                                    'crazyflie_snn.*',
                                    'spiking', 
                                    'spiking.*',])
)
