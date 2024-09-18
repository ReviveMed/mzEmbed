from setuptools import setup, find_packages

setup(
    name='mz-embed engine',       # Name of your package
    version='0.1',                  # Version number
    packages=find_packages(),  # Tells setuptools to look for packages in 'src' folder
    install_requires=[
        # List your dependencies here (e.g., 'numpy', 'torch', etc.)
        # Dependencies will be installed automatically when the package is installed
    ],
    # Optionally, add entry points or scripts if needed
    # entry_points={
    #     'console_scripts': [
    #         'script_name=module:function',
    #     ],
    # },
)
