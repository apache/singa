#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import subprocess
import re
import argparse
import sys


def tag_string(tags):
    if len(tags) == 4:
        return ".".join(str(i) for i in tags[:3]) + "-rc" + str(tags[3])
    elif len(tags) == 3:
        return ".".join(str(i) for i in tags)
    else:
        raise ValueError("malfomated tags %s" % ".".join(str(i) for i in tags))


def main(args):
    # current version
    last_tag = subprocess.run(
        ['git', 'describe', '--abbrev=0', '--tags'],
        stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    # last_tag="4.2.3-rc1"
    # last_tag="3.2.1"
    tags = re.split("\.|-rc", last_tag)
    new_tags = [int(i) for i in tags]

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-y',
        default=False,
        dest='confirmed',
        action='store_true',
        help="In interactive mode, for user to confirm. Could be used in script"
    )
    parser.add_argument('type',
                        choices=['major', 'minor', 'patch', 'rc', 'stable'],
                        help="Release types")
    args = parser.parse_args(args)

    # check eligibility of arg
    if args.type in ['major', 'minor', 'patch']:
        if len(tags) == 4:
            exit("Current type \"%s\" is not allowed in pre release(version %s)"
                 % (args.type, last_tag))
    if args.type in ['stable', 'rc']:
        if len(tags) == 3:
            exit(
                "Current type \"%s\" is not allowed in stable release(version %s)"
                % (args.type, last_tag))

    # new version
    if args.type == 'major':
        new_tags[0] += 1
        new_tags[1] = 0
        new_tags[2] = 0
        new_tags.append(0)
    elif args.type == 'minor':
        new_tags[1] += 1
        new_tags[2] = 0
        new_tags.append(0)
    elif args.type == 'patch':
        new_tags[2] += 1
        new_tags.append(0)
    elif args.type == 'stable':
        new_tags.pop(-1)
    elif args.type == 'rc':
        new_tags[3] += 1

    # ask for confirmation
    print("Please confirm bumping version from %s to %s" %
          (last_tag, tag_string(new_tags)))
    ans = "y" if args.confirmed else ""
    while ans not in ['y', 'n']:
        ans = input("OK to continue [Y/N]? ").lower()
    if ans == "y":
        print("Confirmed bumping version from %s to %s" %
              (last_tag, tag_string(new_tags)))
    else:
        exit("Aborted")

    # do the rest of the work
    # git tag -a $NEW_VERSION -m "Version: $NEW_VERSION"
    print(
        subprocess.run(
            ['git', 'tag', '-a',
             tag_string(new_tags), '-m', 'new version'],
            stdout=subprocess.PIPE).stdout)
    # git push dcslin -f --tags
    # print( subprocess.run(['git', 'push', 'dcslin', '-f', '--tags'], stdout=subprocess.PIPE).stdout) # test
    print(
        subprocess.run(['git', 'push', '--tags'],
                       stdout=subprocess.PIPE).stdout)
    print("Done. Pushed to remote")


if __name__ == "__main__":
    main(sys.argv[1:])
    # main(["-y","major"])
    # main(["-y","patch"])
    # main(["-y","minor"])
    # main(["-y","stable"])
    # main(["stable"])
    # main(["-y","rc"])
