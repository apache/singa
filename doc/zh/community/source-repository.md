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
# 源代码库

___

该项目使用 [Git](http://git-scm.com/) 来管理其源代码. 有关 Git 的使用说明，请访问 [http://git-scm.com/documentation](http://git-scm.com/documentation).

## Web 访问

以下是指向在线源代码库的链接.

* [https://git-wip-us.apache.org/repos/asf?p=singa.git;a=summary](https://git-wip-us.apache.org/repos/asf?p=singa.git;a=summary)


## 提交者的上游 Upstream

提交者需要将上游 upstream 端点设置为 Apache git（而不是github）的 repo 地址, 例如,

    $ git remote add asf https://git-wip-us.apache.org/repos/asf/singa.git

然后你（提交者）可以用这种方式推送你的代码,

    $ git push asf <local-branch>:<remote-branch>
