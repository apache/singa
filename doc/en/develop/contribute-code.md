<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->
## How to Contribute Code


### Coding Style

The SINGA codebase follows the [Google C++ Style Guide](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml).

To check if your code follows the style, you can use the provided cpplint tool:

    $ ./tool/cpplint.py YOUR_FILE


### JIRA format

Like other Apache projects, SINGA uses JIRA to track bugs, improvements and
other high-level discussions (e.g., system design and features).  Github pull requests are
used for implementation discussions, e.g., code review and code merge.

* Provide a descriptive Title.
* Write a detailed Description. For bug reports, this should ideally include a
  short reproduction of the problem. For new features, it may include a design
  document.
* Set [required fields](https://cwiki.apache.org/confluence/display/SPARK/Contributing+to+Spark#ContributingtoSpark-JIRA)

### Pull Request

The work flow is

* Fork the [SINGA Github repository](https://github.com/apache/incubator-singa) to
your own Github account.

* Clone your fork, create a new branch (e.g., feature-foo or fixbug-foo),
 work on it. After finishing your job,
 [rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) it to the
 current latest master and push commits to your own Github account (the new
 branch).

* Open a pull request against the master branch of apache/incubator-singa.
The PR title should be of the form SINGA-xxxx Title, where
SINGA-xxxx is the relevant JIRA number, and Title may be the JIRA's title or a
more specific title describing the PR itself, for example, "SINGA-6 Implement thread-safe singleton". Detailed description can be copied from the JIRA.
Consider identifying committers or other contributors who have worked on the
code being changed. Find the file(s) in Github and click "Blame" to see a
line-by-line annotation of who changed the code last.  You can add @username in
the PR description to ping them immediately.
Please state that the contribution is your original work and that you license
the work to the project under the project's open source license. Further commits (e.g., bug fix)
to your new branch will be added to this pull request automatically by Github.

* Wait for one committer to review the patch. If no conflicts, the committers will merge it with
the master branch. The merge should a) not use rebase b) disable fast forward merge c) check the
commit message format and test the code/feature.

* If there are too many small commit messages, you will be told to squash your commits into fewer meaningful
commits. If your commit message does not follow the format (i.e., SINGA-xxxx), you will be told to
reword your commit message. Both changes can be done using interactive git rebase. Once you
get the commits corrected, push them to you own github again. Your pull request
will be automatically updated. For details, please refer to
[Rebase Pull Requests](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request).
