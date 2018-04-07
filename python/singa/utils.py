# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import sys


def update_progress(progress, info):
    """Display progress bar and user info.

    Args:
        progress (float): progress [0, 1], negative for halt, and >=1 for done.
        info (str): a string for user provided info to be displayed.
    """
    barLength = 20  # bar length
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float. "
    if progress < 0:
        progress = 0
        status = "Halt. "
    if progress >= 1:
        progress = 1
        status = "Done. "
    status = status + info
    block = int(round(barLength * progress))
    text = "[{0}] {1:3.1f}% {2}".format("." * block + " " * (barLength - block),
                                        progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.write('\b' * (9 + barLength + len(status)))
    sys.stdout.flush()
