from setuptools import setup, find_packages

setup(
    name='deepke',  # 打包后的包文件名
    version='2.0.0',    #版本号
    keywords=["pip", "RE","NER","AE"],    # 关键字
    description='DeepKE 是基于 Pytorch 的深度学习中文关系抽取处理套件。',  # 说明
    long_description="client",  #详细说明
    license="MIT",  # 许可
    url='https://github.com/zjunlp/deepke',
    author='ZJUNLP',
    author_email='zhangningyu@zju.edu.cn',
    include_package_data=True,
    platforms="any",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch>=1.5,<=1.11',
        'hydra-core==1.1.1',
        'tensorboard==2.8.0',
        'matplotlib==3.5.1',
        'transformers==4.16.2',
        'jieba==0.42.1',
        'scikit-learn==1.0.2',
        'seqeval==1.2.2',
        'tqdm==4.63.1',
        'opt-einsum==3.3.0',
        'wandb==0.12.11',
        "ujson"
    ], 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
