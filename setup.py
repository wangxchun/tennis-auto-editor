import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tennis-auto-editor", # Replace with your own package name
    version="1.0",
    author="wangxchun",
    author_email="wang.x.chun@sjtu.edu.cn",
    description="A auto editor which can help you get all playing moment in a tennis contest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangxchun/tennis-auto-editor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
