from setuptools import setup, find_packages

setup(
    name='DriveVLMs', 
    author="bjxx",
    version='v0.2',
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=[
        "transformers==4.48.2",
        "accelerate>=1.3.0",
        "soundfile==0.13.1",
        "pillow>=11.1.0",
        "scipy>=1.13.1",
        "torchvision==0.21.0",
        "backoff==2.2.1",
        "peft==0.15.1"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
