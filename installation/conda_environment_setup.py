import yaml
import subprocess
import os

def create_conda_env(env_name):
    try:
        subprocess.run(["conda", "create", "-n", env_name, "-y"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create environment '{env_name}'.")
        raise e

def install_pytorch(env_name):
    packages = ["pytorch", "torchvision", "torchaudio", "pytorch-cuda"]
    channels = ["-c", "pytorch-nightly", "-c", "nvidia"]
    try:
        subprocess.run(["conda", "install", "-n", env_name, "-y"] + channels + packages, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to install PyTorch.")
        print(e)

def install_speakeasy(env_name, whl_path):
    if not os.path.isfile(whl_path):
        print(f"Error: The .whl file does not exist at the specified path: {whl_path}")
        return

    try:
        subprocess.run(["conda", "run", "-n", env_name, "pip", "install", whl_path], check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to install speakeasypy.")
        print(e)

def install_packages(env_name, env_file):
    try:
        subprocess.run(["conda", "env", "update", "-n", env_name, "-f", env_file], check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to install packages.")
        print(e)

def main():
    environment_path = "environment.yml"
    with open(environment_path) as f:
        env_data = yaml.safe_load(f)
    env_name = env_data["name"]

    # create_conda_env(env_name)
    # install_pytorch(env_name)
    # whl_path = "speakeasypy-1.0.0-py3-none-any.whl"
    # install_speakeasy(env_name, whl_path)
    install_packages(env_name, environment_path)

if __name__ == "__main__":
    main()
