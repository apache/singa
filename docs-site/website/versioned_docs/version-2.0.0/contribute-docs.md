---
id: version-2.0.0-contribute-docs
title: How to Contribute to Documentation
original_id: contribute-docs
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Docusaurus Website

This website was created with [Docusaurus](https://docusaurus.io/).

You need at least `node` and `yarn` to get started with setting up a local development environment.

1. Start from the SINGA root directory, install any website specific dependencies by `yarn install`.

```sh
# Install dependencies
$ yarn install
```

2.  Run a development server with hot-reloading to check changes by running `yarn start` in the website directory.

```sh
# Start the site
$ yarn run start:website
```

## Docs for the Docusaurus Site

All the docs are located in the `SINGA_ROOT/docs-site/docs/` folder.

## News for the Docusaurus Site

All the news are located in the `SINGA_ROOT/docs-site/website/blog/` folder.

## Website (old version)

This document gives step-by-step instructions for deploying [SINGA website](http://singa.apache.org). SINGA website is built by [Sphinx](http://www.sphinx-doc.org) from a source tree stored in the [git repo](https://github.com/apache/singa/tree/master/doc).

To install Sphinx:

    pip install -U Sphinx==1.5.6

To install the markdown support for Sphinx:

    pip install recommonmark==0.5.0

To install the rtd theme:

    pip install sphinx_rtd_theme==0.4.3

You can build the website by executing the following command from the doc folder:

    ./build.sh html

Committers can update the [SINGA website](http://singa.apache.org/en/index.html) by copying the updated files to the [website repo](https://github.com/apache/singa-site) (suppose the site repo is ~/singa-sit)

    cd _build
    rsync --checksum -rvh html/ ~/singa-site/
    cd ~/singa-site
    git commit -m "update xxxx"
    git push

We fix the versions of the libs in order to generate the same (checksum) html file if the source file is not changed. Otherwise, everytime we build the documentation, the html file of the same source file could be different. As a result, many html files in the site repo need updating.

### Python API

### CPP API

To generate docs, run "doxygen" from the doc folder (Doxygen >= 1.8 recommended)

### Using Visual Studio Code (vscode)

#### Preview

The document files (rst and md files) can be previewed in vscode via the [reStructuredText Extension](https://docs.restructuredtext.net/).

1.  Install the extension in vscode.
2.  Install the dependent libs. All libs required to build the website should be installed (see the above instructions). In addition, there are two more libs to be installed.

        pip install sphinx-autobuild=0.7.1
        pip install doc8=0.8.0

3.  Configure the conf path for `restructuredtext.confPath` to the [conf.py](./conf.py)

#### Docstring Snippet

[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) generates the docstring of functions, classes, etc. Choose the DocString Format to `google`.

#### Spell Check

[Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) can be configured to check the comments of the code, or .md and .rst files.

To do spell check only for comments of Python code, add the following snippet via `File - Preferences - User Snippets - python.json`

    "cspell check" : {
    "prefix": "cspell",
    "body": [
        "# Directives for doing spell check only for python and c/cpp comments",
        "# cSpell:includeRegExp #.* ",
        "# cSpell:includeRegExp (\"\"\"|''')[^\1]*\1",
        "# cSpell: CStyleComment",
    ],
    "description": "# spell check only for python comments"
    }

To do spell check only for comments of Cpp code, add the following snippet via `File - Preferences - User Snippets - cpp.json`

    "cspell check" : {
    "prefix": "cspell",
    "body": [
        "// Directive for doing spell check only for cpp comments",
        "// cSpell:includeRegExp CStyleComment",
    ],
    "description": "# spell check only for cpp comments"
    }
