---
id: version-2.0.0-contribute-code
title: How to Contribute Code
original_id: contribute-code
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Coding Style

The SINGA codebase follows the Google Style for both [CPP](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml) and [Python](http://google.github.io/styleguide/pyguide.html) code.

A simple way to enforce the Google coding styles is to use the linting and formating tools in the Visual Studio Code editor:

- [C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

Once the extensions are installed, edit the `settings.json` file.

```json
"editor.formatOnSave": true,
"python.formatting.provider": "yapf",
"python.formatting.yapfArgs": [
    "--style",
    "{based_on_style: google}"
],
"python.linting.enabled": true,
"python.linting.lintOnSave": true,
"C_Cpp.clang_format_style": "Google"
```

You need to fix the format errors before submitting the pull requests.

## JIRA format

Like other Apache projects, SINGA uses JIRA to track bugs, improvements and other high-level discussions (e.g., system design and features). Github pull requests are used for implementation discussions, e.g., code review and code merge.

- Provide a descriptive Title.
- Write a detailed Description. For bug reports, this should ideally include a short reproduction of the problem. For new features, it may include a design document.
- Set [required fields](https://cwiki.apache.org/confluence/display/SPARK/Contributing+to+Spark#ContributingtoSpark-JIRA)

## Git Workflow

1. Fork the [SINGA Github repository](https://github.com/apache/singa) to your own Github account.

2. Clone the **repo** (short for repository) from your Github

   ```shell
   git clone https://github.com/<Github account>/singa.git
   git remote add apache https://github.com/apache/singa.git
   ```

3. Create a new branch (e.g., `feature-foo` or `fixbug-foo`), work on it and commit your code.

   ```shell
   git checkout -b feature-foo
   # write your code
   git add <created/updated files>
   git commit
   ```

   The commit message should have a **title which consists of the JIRA ticket No (SINGA-xxx) and title**. A brief description of the commit should be added in the commit message.

   If your branch has many small commits, you need to clean those commits via

   ```shell
   git rebase -i <commit id>
   ```

   You can [squash and reword](https://help.github.com/en/articles/about-git-rebase) the commits.

4. When you are working on the code, the `master` of SINGA may have been updated by others; In this case, you need to pull the latest master

   ```shell
   git checkout master
   git pull apache master:master
   git checkout feature-foo
   ```

5. [Rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) `feature-foo` onto the `master` branch and push commits to your own Github account (the new branch).

   ```shell
   git rebase master
   git push origin feature-foo:feature-foo
   ```

6. Open a pull request (PR) against the master branch of apache/singa on Github website. The PR title should be the JIRA ticket title. If you want to inform other contributors who worked on the same files, you can find the file(s) on Github and click "Blame" to see a line-by-line annotation of who changed the code last. Then, you can add @username in the PR description to ping them immediately. Please state that the contribution is your original work and that you license the work to the project under the project's open source license. Further commits (e.g., bug fix) to your new branch will be added to this pull request automatically by Github.

7. Wait for committers to review the PR. If no conflicts and errors, the committers will merge it with the master branch. The merge should **a) not use rebase b) disable fast forward merge c) check the commit message format and test the code/feature**. During this time, the master of SINGA may have been updated by others, and then you need to [merge the latest master](https://docs.fast.ai/dev/git.html#how-to-keep-your-feature-branch-up-to-date) to resolve conflicts. Some people [rebase the PR onto the latest master](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request) instead of merging. However, if other developers fetch this PR to add new features and then send PR, the rebase operation would introduce **duplicate commits** (with different hash) in the future PR. See [The Golden Rule of Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) for the details of when to avoid using rebase. Another simple solution to update the PR (to fix conflicts or commit errors) is to checkout a new branch from the latest master branch of Apache SINGAS repo; copy and paste the updated/added code; commit and send a new PR.

## Developing Environment

Visual Studio Code is recommended as the editor. Extensions like Python, C/C++, Code Spell Checker, autoDocstring, vim, Remote Development could be installed. A reference configuration (i.e., `settings.json`) of these extensions is [here](https://gist.github.com/nudles/3d23cfb6ffb30ca7636c45fe60278c55).

If you update the CPP code, you need to recompile SINGA [from source](./build.md). It is recommended to use the native building tools in the `*-devel` Docker images or `conda build`.

If you only update the Python code, you can install SINGAS once, and then copy the updated Python files to replace those in the Python installation folder,

```shell
cp python/singa/xx.py  <path to conda>/lib/python3.7/site-packages/singa/
```
