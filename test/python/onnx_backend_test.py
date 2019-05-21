import unittest
import onnx.backend.test

from singa import sonnx

backend_test = onnx.backend.test.BackendTest(sonnx.SingaBackend, __name__)

globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()