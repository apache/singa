---
id: version-2.0.0-source-repository
title: Source Repository
original_id: source-repository
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

This project uses [Git](http://git-scm.com/) to manage its source code. Instructions on Git use can be found at http://git-scm.com/documentation .

## Web Access

The following is a link to the online source repository.

- https://gitbox.apache.org/repos/asf?p=singa.git

## Contributors

Contributors are encouraged to rebase their commits onto the latest master before sending the pull requests to make the git history clean. The following git instructors should be executed after committing the current work:

```shell
git checkout master
git pull <apache/singa upstream> master:master
git checkout <new feature branch>
git rebase master
```

## Committers

- To connect your Apache account with your Github account, Please follow the instructions on: https://gitbox.apache.org/setup/. After that you can directly merge PRs using GitHub’s UI.

To merge pull request https://github.com/apache/singa/pull/xxx, the following instructions should be executed,

```shell
git clone https://github.com/apache/singa.git
git remote add asf https://gitbox.apache.org/repos/asf/singa.git
# optional
git pull asf master:master
git fetch origin pull/xxx/head:prxxx
git merge prxxx
git push asf master:master
```

- To migrate from git-wip-us.apache.org to Gitbox: If you already cloned the SINGA repository from the old repo https://git-wip-us.apache.org/repos/asf/singa.git, you can update the master by:

```shell
git remote set-url origin git@github.com/apache/singa.git
```
