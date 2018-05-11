## 如何贡献代码


### 代码风格

SINGA 代码库遵循 [Google C++ 风格指导](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml).

如果想检查你的代码是否符合风格, 你可以用如下 cpplint 工具:

    $ ./tool/cpplint.py YOUR_FILE


### JIRA 格式

像其他 Apache 项目一样，SINGA 使用 JIRA 来追踪错误，改进和其他高层讨论（例如，系统设计和功能). 
Github pull requests 用于实施讨论，例如代码审查和代码合并.

* 提供一个描述性标题.
* 写一个详细的描述. 对于错误报告，这应该最好包括一个问题的短暂再现. 对于新功能，它可能包含一个设计文档.
* 填写[必填字段](https://cwiki.apache.org/confluence/display/SPARK/Contributing+to+Spark#ContributingtoSpark-JIRA)

### Pull Request

工作流程是

* Fork [SINGA Github repository](https://github.com/apache/incubator-singa) 到你自己的 Github 帐户.

* Clone 你自己的 fork, 创建一个新的 branch (例如, feature-foo or fixbug-foo),
 进行这项工作. 完成你的工作后，
 [rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) 
它到当前最新的 master 并 push commits 到你自己的 Github 帐户 (新 branch).

* 针对 apache / incubator-singa 的 master branch 打开一个 pull request.
PR 标题应该是 SINGA-xxxx 的格式，其中 SINGA-xxxx 是相关的JIRA编号，
标题可以是 JIRA 的标题或描述 PR 本身的更具体的标题, 例如，"SINGA-6 Implement thread-safe singleton". 
详细描述可以从 JIRA 复制.
考虑确定提交者或者在被改变的代码工作的其他贡献者. 在 Github 中找到文件并点击 "Blame" 查看最后修改了代码的逐行注释. 您可以在中添加含有 @username 的 pull request 描述并立即 ping 他们.
请说明你的原创作品和贡献并且您根据项目的开源许可证将工作许可给项目. 
进一步向你的新分支进行的提交(例如错误修复)会被 Github 自动添加到这个 pull request.

* 等待一个提交者查看该补丁. 如果没有冲突，提交者会将其与 master branch 合并. 
合并应该 a) 不用 rebase b) 禁用 fast forward merge c) 检查提交消息格式并测试代码/功能.

* 如果有太多的小提交信息，你会被告知将你的提交压缩成更少的有意义的提交. 
如果您的提交信息不符合格式 (如 SINGA-xxxx), 你会被告知重新提交你的提交信息. 
这两个更改都可以使用交互式 git rebase. 一旦你得到了更正的提交，
再次将它们推送到你自己的 github. 你的 pull request 会自动更新. 
详情请参阅 [Rebase Pull Requests](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request).
