## Served docs

The docs are automatically served to [https://docs.modula.systems/](docs.modula.systems/).

## Building the docs locally

To build these docs locally install sphinx with the furo theme:
```bash
pip install sphinx sphinxext-opengraph sphinx-inline-tabs sphinx-autobuild sphinx-copybutton sphinxcontrib-youtube sphinx-design furo matplotlib
```
And then do the build:
```bash
cd docs
make livedirhtml
```
