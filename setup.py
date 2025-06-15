from setuptools import setup, find_packages

setup(
    name="face-recognition-app",
    packages=["backend"],
    include_package_data=True,
    package_data={
        "backend": [
            "templates/*.html",
            "*.xlsx",
            "data/*",
            "known_faces/*"
        ]
    }
)
