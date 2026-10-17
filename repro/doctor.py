"""Check the interpreter and dependencies without downloading or training."""
import platform
import shutil
import sys
import importlib.metadata


def main():
    print('Python', platform.python_version(), sys.executable)
    for name in ('torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML', 'gdown', 'tensorboardX', 'matplotlib', 'psycopg2-binary', 'pytest'):
        print(name, importlib.metadata.version(name))
    import torch
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name())
    print('PostgreSQL build tools:', {name: shutil.which(name) for name in ('make','gcc','g++','perl')})
    print('Environment ready. CPU: verification, rendering and inference; CUDA GPU: full model training.')


if __name__ == '__main__':
    main()
