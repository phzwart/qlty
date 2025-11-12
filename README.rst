====
qlty
====

.. image:: https://img.shields.io/pypi/v/qlty.svg
        :target: https://pypi.python.org/pypi/qlty

.. image:: https://img.shields.io/travis/phzwart/qlty.svg
        :target: https://travis-ci.com/phzwart/qlty

.. image:: https://readthedocs.org/projects/qlty/badge/?version=latest
        :target: https://qlty.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

qlty is a Python library designed to handle large 2D or 3D tensors efficiently by splitting them into smaller, manageable chunks. This library is particularly useful for processing large datasets that do not fit into memory, enabling chunked processing for machine learning workflows.

Features
--------

* Efficient tensor splitting and stitching
* Intelligent border handling to minimize artifacts
* Support for both in-memory and disk-cached processing
* 2D and 3D tensor support
* Sparse data handling utilities
* Patch pair extraction helpers for partially overlapping regions in 2D and 3D
* Numba acceleration for 2D stitching with race-free colored parallelism

Quick Start
-----------

Installation::

    pip install qlty torch zarr numpy einops dask numba

Basic Usage::

    import torch
    from qlty import NCYXQuilt

    # Create a quilt object
    quilt = NCYXQuilt(
        Y=128, X=128,
        window=(32, 32),
        step=(16, 16),
        border=(5, 5),
        border_weight=0.1
    )

    # Split data into patches
    data = torch.randn(10, 3, 128, 128)
    patches = quilt.unstitch(data)

    # Process patches (e.g., with a neural network)
    processed = your_model(patches)

    # Stitch back together
    reconstructed, weights = quilt.stitch(processed)

Documentation
-------------

Full documentation is available at https://qlty.readthedocs.io

* `Installation Guide <installation.html>`_
* `Usage Guide <usage.html>`_
* `Examples <examples.html>`_
* `API Reference <api.html>`_
* `Troubleshooting <troubleshooting.html>`_

Modules
-------

In-Memory Classes
~~~~~~~~~~~~~~~~~

* **NCYXQuilt**: 2D tensor splitting and stitching (shape: N, C, Y, X)
* **NCZYXQuilt**: 3D tensor splitting and stitching (shape: N, C, Z, Y, X)

Disk-Cached Classes
~~~~~~~~~~~~~~~~~~~

* **LargeNCYXQuilt**: 2D with on-disk caching using Zarr
* **LargeNCZYXQuilt**: 3D with on-disk caching using Zarr

License
-------

* Free software: BSD license

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
