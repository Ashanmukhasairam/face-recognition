from setuptools import setup, find_packages

setup(
    name="face-recognition-attendance",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'opencv-python-headless==4.8.0.74',
        'face-recognition==1.3.0',
        'flask==2.3.3',
        'pandas==2.0.3',
        'pymongo==4.4.1',
        'python-dotenv==1.0.0',
        'gunicorn==21.2.0',
        'openpyxl==3.1.2'
    ]
)
