## How to Contribute Documentation

___

This document gives step-by-step instructions for deploying [Singa website](http://singa.incubator.apache.org).

Singa website is built by [Sphinx](http://www.sphinx-doc.org) from a source tree stored in git: https://github.com/apache/incubator-singa/tree/master/doc. 

To install sphinx on Ubuntu: 

    $ apt-get install python-sphinx

To install the markdown support for sphinx: 

    $ pip install recommonmark 

You can build the website by executing the following command from the doc folder:

    $ make html

The procedure for contributing documentation is the same as [contributing code](contribute-code.html).

