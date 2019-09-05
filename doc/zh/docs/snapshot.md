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
# Snapshot

此模块包含io::snapshot类及其方法。

示例用法：

```python
from singa import snapshot

sn1 = snapshot.Snapshot('param', False)
params = sn1.read()  # read all params as a dictionary

sn2 = snapshot.Snapshot('param_new', False)
for k, v in params.iteritems():
    sn2.write(k, v)
```

---

### class singa.snapshot.Snapshot(f, mode, buffer_size=10)

`singa::Snapshot`类和成员函数。

---

#### read()

调用read方法加载所有信息（参数名，参数值）。

**返回值：** （参数名，参数值）的字典

---

#### write(param_name, param_val)

调用write方法写回参数。

**参数：**
- **param_name (string)** – 参数名
- **param_val (Tensor)** – 参数值的tensor

---
