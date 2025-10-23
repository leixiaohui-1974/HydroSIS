# RichDEM - Local Bundled Version

This directory contains a local copy of RichDEM 0.3.4, including the pre-compiled binary extension.

## Why Local Copy?

RichDEM is a critical dependency for HydroSIS, but installing it from PyPI often fails due to:
- Complex C++ compilation requirements
- Missing system dependencies
- Build toolchain issues

By bundling a pre-compiled version, we ensure reliable, consistent behavior across environments.

## Contents

- `__init__.py` - Main RichDEM Python interface
- `cli.py` - Command-line interface
- `_richdem.cpython-312-x86_64-linux-gnu.so` - Pre-compiled C++ extension for Python 3.12 on Linux x86_64

## Usage

Import richdem from hydrosis:

```python
from hydrosis import richdem as rd

# Use richdem functions normally
dem_filled = rd.FillDepressions(dem_array)
flow_dir = rd.FlowAccumulation(dem_filled, method='D8')
```

## Original Source

Original RichDEM source: https://github.com/r-barnes/richdem
Version: 0.3.4
License: GPL-3.0

## Notes

- The binary is compiled for Python 3.12 on Linux x86_64
- For other platforms, you may need to recompile from the original source in `/richdem-0.3.4/`
- This local copy removes the `pkg_resources` dependency for version checking
