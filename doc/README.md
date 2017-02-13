# How to Contribute to Documentation


## Website
This document gives step-by-step instructions for deploying [Singa website](http://singa.incubator.apache.org).

Singa website is built by [Sphinx](http://www.sphinx-doc.org) >=1.4.4 from a source tree stored in git: https://github.com/apache/incubator-singa/tree/master/doc.

To install Sphinx:

    $ pip install -U Sphinx

To install the markdown support for Sphinx:

    $ pip install recommonmark

To install the rtd theme:

    $ pip install sphinx_rtd_theme

You can build the website by executing the following command from the doc folder:

    $ ./build.sh html

Committers can update the [SINGA website](http://singa.apache.org/en/index.html) by following these steps:

    $ cd _build
    $ svn co https://svn.apache.org/repos/asf/incubator/singa/site/trunk
    $ cp -r html/* trunk
    # svn add <newly added html files>
    $ svn commit -m "commit messages" --username  <committer ID> --password <password>


## CPP API

To generate docs, run "doxygen" from the doc folder (Doxygen >= 1.8 recommended)
