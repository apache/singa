# 源代码库

___

该项目使用 [Git](http://git-scm.com/) 来管理其源代码. 有关 Git 的使用说明，请访问 [http://git-scm.com/documentation](http://git-scm.com/documentation).

## Web 访问

以下是指向在线源代码库的链接.

* [https://git-wip-us.apache.org/repos/asf?p=incubator-singa.git;a=summary](https://git-wip-us.apache.org/repos/asf?p=incubator-singa.git;a=summary)


## 提交者的上游 Upstream

提交者需要将上游 upstream 端点设置为 Apache git（而不是github）的 repo 地址, 例如,

    $ git remote add asf https://git-wip-us.apache.org/repos/asf/incubator-singa.git

然后你（提交者）可以用这种方式推送你的代码,

    $ git push asf <local-branch>:<remote-branch>
