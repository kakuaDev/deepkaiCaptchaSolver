import setuptools

setuptools.setup(
    name="deepkaiCaptchaSolver",
    version='0.0.0.1',  # [Production Version].[Tested Version].[Fixes Version].[Features Version]
    author='Kakua Dev',
    author_email='kakua.develop@gmail.com',
    maintainer='Kakua Dev',
    maintainer_email='kakua.develop@gmail.com',
    description="deepkaiCaptchaSolver",
    url="https://github.com/kakuaDev/deepkaiCaptchaSolver",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        "tensorflow==2.12.0"
    ]
)