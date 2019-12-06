---
id: version-2.0.0-how-to-release
title: How to Prepare a Release
original_id: how-to-release
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

This is a guide for the release preparing process in SINGA.

## Select a release manager

The release manager (RM) is the coordinator for the release process. It is the RM's signature (.asc) that is uploaded together with the release. The RM generates KEY (RSA 4096-bit) and uploads it to a public key server. The RM needs to get his key endorsed (signed) by other Apache user, to be connected to the web of trust. http://www.apache.org/dev/release-signing.html

Check:

- The codebase does not include third-party code which is not compatible to APL
- The dependencies are compatible with APL. GNU-like licenses are NOT compatible
- All source files written by us MUST include the Apache license header: http://www.apache.org/legal/src-headers.html. There's a script in there which helps propagating the header to all files.
- The build process is error-free.
- Unit tests are included (as much as possible)
- The Jupyter notebooks are working with the new release
- The online documentation on the Apache website is up to date.

## Prepare LICENSE file

copy and paste this http://apache.org/licenses/LICENSE-2.0.txt

## Prepare NOTICE file

- Use this template: http://apache.org/legal/src-headers.html#notice
- If we include any third party code in the release package which is not APL, must state it at the end of the NOTICE file.
- Example: http://apache.org/licenses/example-NOTICE.txt

## Prepare RELEASE_NOTES file

- Introduction, Features, Bugs (link to JIRA), Changes (N/A for first erlease), Dependency list, Incompatibility issues.
- Follow this example: http://commons.apache.org/proper/commons-digester/commons-digester-3.0/RELEASE-NOTES.txt

## Prepare README file

- How to build, run test, run examples
- List of dependencies.
- Mail list, website, etc. Any information useful for user to start.

## Package the release

The release should be packaged into : apache-singa-xx.xx.xx.tar.gz

- src/
- README
- LICENSE
- NOTICE
- RELEASE_NOTES
- ...

## Upload the release

The release is uploaded to the RM’s Apache page: people.apache.org/~ID/...

- apache-singa-xx.xx.xx.tar.gz
- KEY
- XX.acs
- XX.md5

## Roll out artifacts to mirrors

svn add to “dist/release/singa”

Delete old artifacts (automatically archived)

## Update the Download page

The tar.gz file MUST be downloaded from mirror, using closer.cgi script other artifacts MUST be downloaded from main Apache site Good idea to update EC2 image and make it available for download as well

## Make the internal announcements

Template for singa-dev@ voting:

```text
To: dev@singa.apache.org
Subject: [VOTE] Release apache-singa-X.Y.Z (release candidate N)

Hi all,

I have created a build for Apache SINGA X.Y.Z, release candidate N.

The artifacts to be voted on are located here:
https://dist.apache.org/repos/dist/dev/singa/apache-singa-X.Y.Z-rcN/

The hashes of the artifacts are as follows:
apache-singa-X.Y.Z.tar.gz.md5 XXXX
apache-singa-X.Y.Z.tar.gz.sha256 XXXX

Release artifacts are signed with the following key:
https://people.apache.org/keys/committer/{Apache ID of the Release Manager}.asc

and the signature file is:
apache-singa-X.Y.Z.tar.gz.asc

Please vote on releasing this package. The vote is open for at least 72 hours and passes if a majority of at least three +1 votes are cast.

[ ] +1 Release this package as Apache SINGA X.Y.Z
[ ]  0 I don't feel strongly about it, but I'm okay with the release
[ ] -1 Do not release this package because...

Here is my vote:

+1

{SINGA Team Member Name}
```

Wait at least 48 hours for test responses

Any PMC, committer or contributor can test features for releasing, and feedback. Based on that, PMC will decide whether start a vote.

## Call a vote in dev

Call a vote in dev@singa.apache.org

## Vote Check

All PMC members and committers should check these before vote +1 :

## Vote result mail

Template for singa-dev@ voting (results):

```text
Subject: [RESULT] [VOTE] Release apache-singa-X.Y.Z (release candidate N)
To: dev@singa.apache.org

Thanks to everyone who has voted and given their comments. The tally is as follows.

N binding +1s:
<names>

N non-binding +1s:
<names>

No 0s or -1s.

I am delighted to announce that the proposal to release
Apache SINGA X.Y.Z has passed.

I'll now start a vote on the general list. Those of you in the IPMC, please recast your vote on the new thread.

{SINGA Team Member Name}
```

## Publish release

Template for ANNOUNCING the release

```text
To: announce@apache.org, dev@singa.apache.org
Subject: [ANNOUNCE] Apache SINGA X.Y.Z released

We are pleased to announce that SINGA X.Y.Z is released.

SINGA is a general distributed deep learning platform for training big deep learning models over large datasets. It is designed with an intuitive programming model based on the layer abstraction. SINGA supports a wide variety of popular deep learning models.

The release is available at:
http://singa.apache.org/downloads.html

The main features of this release include XXX

We look forward to hearing your feedbacks, suggestions, and contributions to the project.

On behalf of the SINGA team,
{SINGA Team Member Name}
```
