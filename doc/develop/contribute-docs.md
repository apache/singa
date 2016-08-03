# How to Contribute Documentation

___


## Website
This document gives step-by-step instructions for deploying [Singa website](http://singa.incubator.apache.org).

Singa website is built by [Sphinx](http://www.sphinx-doc.org) 1.4.4 from a source tree stored in git: https://github.com/apache/incubator-singa/tree/master/doc.

To install Sphinx on Ubuntu:

    $ apt-get install python-sphinx

To install the markdown support for Sphinx:

    $ pip install recommonmark

You can build the website by executing the following command from the doc folder:

    $ make html

The procedure for contributing documentation is the same as [contributing code](contribute-code.html).


## CPP API

To generate docs, run "doxygen" from the doc folder (Doxygen >= 1.8 recommended)
