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
