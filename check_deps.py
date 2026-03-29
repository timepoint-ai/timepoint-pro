#!/usr/bin/env python3
"""
check_deps.py - Check if all required dependencies are installed
"""

import sys

required = {
    "sqlmodel": "sqlmodel",
    "bleach": "bleach",
    "hydra": "hydra-core",
    "pytest": "pytest",
    "pytest_asyncio": "pytest-asyncio",
    "pytest_cov": "pytest-cov",
    "dotenv": "python-dotenv",
}

print("Checking test dependencies...\n")

missing = []
installed = []

for module, package in required.items():
    try:
        __import__(module)
        print(f"  {package} - OK")
        installed.append(package)
    except ImportError:
        print(f"  {package} - MISSING")
        missing.append(package)

print(f"\nInstalled: {len(installed)}/{len(required)}")

if missing:
    print(f"\nMissing {len(missing)} package(s):")
    for pkg in missing:
        print(f"   - {pkg}")
    print(f"\nTo install: pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll test dependencies installed!")
    sys.exit(0)
