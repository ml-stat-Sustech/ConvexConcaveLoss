

from setuptools import setup, find_packages

setup(
    name="source",
    version="0.1",
    install_requires=[
        'imutils',
        'GPUtil',
        "runx",
        "torchkit",
        "opacus",
        "art",
        "torchvision",
    ],
    packages=["source.data_preprocessing",
              "source.attacks",
              "source.defenses",
              ]
)
