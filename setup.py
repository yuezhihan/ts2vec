from setuptools import setup

def load_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ts2vec",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="TS2Vec: Towards Universal Representation of Time Series",
    url="https://github.com/zhihanyue/ts2vec",
    packages=[".", "models", "tasks"],
    install_requires=load_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8,<3.9",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
