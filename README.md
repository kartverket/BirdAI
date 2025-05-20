# BirdAI

Objektdeteksjon av flyvende objekter

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

BirdAI is a project for object detection of flying objects.

## Features

- Detects various flying objects using AI
- Written primarily in Python
- Includes C++ components for performance

## Installation

```bash
# Clone the repository
git clone https://github.com/kartverket/BirdAI.git
cd BirdAI

# Install dependencies
pip install -r requirements.txt

## Usage

# Run inference
python tools/inference/web_inf.py -c configs/dfine/dfine_hgnetv2_n_custom.yml -r best.pth --device CPU

## Contributing
TBA

## License
TBA
