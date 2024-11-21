from setuptools import find_packages, setup


def read_requirements(filename):
    """Read a requirements file and return a list of dependencies."""
    with open(filename, "r") as f:
        return f.read().splitlines()


setup(
    name="bomberman_rl",
    version="0.1.0",
    description="Bomberman environment for reinforcement learning",
    author="N.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements("requirements.txt"),
    extras_require={
    },
    python_requires=">=3.8",
)
