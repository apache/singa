
## Introduction

We perform a maturity assessment for SINGA podling as part of the graduation process for SINGA from the Apache Incubator. The assessment is based on the ASF project maturity model at https://community.apache.org/apache-way/apache-project-maturity-model.html

All project members and mentors are welcome to comment or update assessment.

## Maturity assessment status

Some open items are still under development.

## Maturity model assessment

Mentors and community members are encouraged to contribute to this page and comment on it, the following table summarizes project’s self-assessment against the Apache Maturity Model.


| ID | Description | Status |
|:---|:------------|:-------|
| ***Code*** | | |
| CD10     | The project produces Open Source software, for distribution to the public at no charge. | **YES** The project source code is licensed under the Apache License, version 2.0.                      |
| CD20     | The project's code is easily discoverable and publicly accessible. | **YES** The project's code can be found from the website.  The code is available via GitBox https://gitbox.apache.org/repos/asf?p=incubator-singa.git and https://github.com/apache/incubator-singa. There is a Fork Me on GitHub ribbon on the SINGA website.|
| CD30     | The code can be built in a reproducible way using widely available standard tools. | **YES** The build of the C++ code uses standard tools such as gcc and cmake. Continuous integration with Travis CI is used to automate the testing of new commits. |
| CD40     | The full history of the project's code is available via a source code control system, in a way that allows any released version to be recreated. | **YES**  All the history of the project is available through Git. All releases are properly tagged. |
| CD50     | The provenance of each line of code is established via the source code control system, in a reliable way based on strong authentication of the committer. When third-party contributions are committed, commit messages provide reliable information about the code provenance.  | **YES** The git repository is managed by Apache Infra. Only SINGA committers have write access. For 3rd party contribution, the commit message and logs will include all the details of author and committer. |
| ***Licenses and Copyright*** | | |
| LC10     | The code is released under the Apache License, version 2.0. | **YES** The LICENCE file is included with both the source and binary distributions. |
| LC20     | Libraries that are mandatory dependencies of the project's code do not create more restrictions than the Apache License does. | **YES** All the dependencies of SINGA are reviewed and no more restrictions than the Apache Licence was found.|
| LC30     | The libraries mentioned in LC20 are available as Open Source software. | **YES** All SINGA dependencies are available as open source software. |
| LC40     | Committers are bound by an Individual Contributor Agreement (the "Apache iCLA") that defines which code they are allowed to commit and how they need to identify code that is not their own. | **YES** The project uses a repository managed by Apache Infra. Write access requires an Apache account, which requires an ICLA on file. |
| LC50     | The copyright ownership of everything that the project produces is clearly defined and documented. | **YES** Automated process is in place to ensure every file has expected headers. |
| ***Releases*** | | |
| RE10     | Releases consist of source code, distributed using standard and open archive formats that are expected to stay readable in the long term. | **YES** Source releases are distributed via https://dist.apache.org/repos/dist/release/incubator/singa/ and linked from the website at https://singa.incubator.apache.org/download/.
| RE20     | Releases are approved by the project's PMC (see CS10), in order to make them an act of the Foundation.| **YES** All incubating releases have been approved by the SINGA community with at least 3 PPMC votes and from the Incubator with 3 IPMC votes. |
| RE30     | Releases are signed and/or distributed along with digests that can be reliably used to validate the downloaded archives. | **YES** All releases are signed, and the KEYS file is provided on dist.apache.org. |
| RE40     | Convenience binaries can be distributed alongside source code but they are not Apache Releases -- they are just a convenience provided with no guarantee. | **YES** Convenience binaries are distributed via via dist.apache.org. Conda and Debian binary packages are also distributed |
| RE50     | The release process is documented and repeatable to the extent that someone new to the project is able to independently generate the complete set of artifacts required for a release. | **YES** Step-by-step release guide is available describing the entire process. |
| ***Quality*** | | |
| QU10     | The project is open and honest about the quality of its code. Various levels of quality and maturity for various modules are natural and acceptable as long as they are clearly communicated. | **YES** The project records all bugs in the GitHub issue tracker at https://issues.apache.org/jira/browse/singa |
| QU20     | The project puts a very high priority on producing secure software. | **YES** Security issues are treated with the highest priority.|
| QU30     | The project provides a well-documented channel to report security issues, along with a documented way of responding to them. | **NO**  https://issues.apache.org/jira/browse/SINGA-417 |
| QU40     | The project puts a high priority on backwards compatibility and aims to document any incompatible changes and provide tools and documentation to help users transition to new features. | **YES** SINGA releases do not to break backward compatibility with new releases unless there is no other option. The documentation provides all the details for the new changes and how to use them.|
| QU50     | The project strives to respond to documented bug reports in a timely manner. | **YES** The project has resolved 292 issues during incubation. |
| ***Community*** | | |
| CO10     | The project has a well-known homepage that points to all the information required to operate according to this maturity model. | **YES** The project website has a description of the project with technical details, how to contribute code or documentation, how to release. |
| CO20     | The community welcomes contributions from anyone who acts in good faith and in a respectful manner and adds value to the project. | **YES** It’s part of the contribution guide (http://singa.apache.org/en/develop/how-contribute.html) and the current committers are really keen to welcome contributions.
| CO30     | Contributions include not only source code, but also documentation, constructive bug reports, constructive discussions, marketing and generally anything that adds value to the project. | **YES** The website refers to documentation contribution guide: http://singa.apache.org/en/develop/contribute-docs.html and mail lists: http://singa.apache.org/en/community/mail-lists.html . Links to profiles on social networks are also available for internet marketing contributions.
| CO40     | The community is meritocratic and over time aims to give more rights and responsibilities to contributors who add value to the project. | **YES** The community has elected new committers during incubation, based on meritocracy. |
| CO50     | The way in which contributors can be granted more rights such as commit access or decision power is clearly documented and is the same for all contributors. | **NO** The criteria will be documented in the contribution guide. |
| CO60     | The community operates based on consensus of its members (see CS10) who have decision power. Dictators, benevolent or not, are not welcome in Apache projects. | **YES** The project works to build consensus. Voting is used when there is an important decision.
| CO70     | The project strives to answer user questions in a timely manner. | **YES** The project typically provides detailed answers to user questions via dev@ mailing list or Jira website.
| ***Consensus Building*** | | |
| CS10     | The project maintains a public list of its contributors who have decision power -- the project's PMC (Project Management Committee) consists of those contributors. | **NO** The website will contains the list of committers and PPMC members at http://singa.apache.org/en/community/team-list.html. See https://github.com/apache/incubator-singa/pull/426
| CS20     | Decisions are made by consensus among PMC members and are documented on the project's main communications channel. Community opinions are taken into account but the PMC has the final word if needed. | **YES** The project has been making all decisions on the project mailing lists. |
| CS30     | Documented voting rules are used to build consensus when discussion is not sufficient. | **YES** The project uses the standard ASF voting rules. Voting rules are clearly stated before the voting starts for each individual vote. |
| CS40     | In Apache projects, vetoes are only valid for code commits and are justified by a technical explanation, as per the Apache voting rules defined in CS30. | **YES** The project hasn’t used a veto at any point. |
| CS50     | All "important" discussions happen asynchronously in written form on the project's main communications channel. Offline, face-to-face or private discussions that affect the project are also documented on that channel. | **YES** The project has been making all decisions on the project mailing lists.
| **Independence** | | |
| IN10     | The project is independent from any corporate or organizational influence. | **YES** The SINGA community has developers mainly from National University of Singapore, Zhejiang University, NetEase, Osaka University, yzBigData, and other volunteers. |
| IN20     | Contributors act as themselves as opposed to representatives of a corporation or organization. | **YES** All contributors are aware that contributions to the project are made on an individual base and not on behalf of a corporation or organization. |
