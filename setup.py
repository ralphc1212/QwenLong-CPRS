from setuptools import setup, find_packages

setup(
    name='qwen_long_cprs',  # 包的名字
    version='0.1',
    packages=find_packages(where='src'),  # 告诉setuptools在src目录下查找包
    package_dir={'': 'src'},  # 设置包的根目录是src
    install_requires=[
        "transformers==4.44.2"
        "torch==2.4.1"
        "uvicorn==0.30.4"
        "openai"
    ],  # 安装依赖项，可以从requirements.txt中读取或直接写在这
    )
