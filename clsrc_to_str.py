#!/usr/bin/python

import os.path

distribution = "./src/core/tensor/distribution.cl"
tensormath = "./src/core/tensor/tensor_math_opencl.cl"
im2col = "./src/model/layer/im2col.cl"
pooling = "./src/model/layer/pooling.cl"

files = {"distribution_str" : distribution,
		 "tensormath_str" : tensormath,
		 "im2col_str" : im2col,
		 "pooling_str" : pooling}

if __name__ == "__main__":

	for name, path in files.items():
		with open(path, 'r') as file:
			src = file.read()
		src = repr(src)
		src = src[1:-1]
		src = src.replace('\"', '\\"') # Escape double quotes
		src = src.replace('\\t', '') # Strip out tabs
		
		fullpath = os.path.dirname(path)
		fullpath = fullpath + "/" + name + ".cpp"
		
		with open(fullpath, 'w') as file:
			file.write("// This file is auto-generated, do not edit manually.\n")
			file.write("// If any error occurs during compilation, please refer to clsrc_to_str.py\n")
			file.write("#include <string>\n\n")
			file.write("namespace singa {\n\n")
			file.write("std::string " + name + " = \"")
			file.write(src)
			file.write("\";")
			file.write("\n\n} // namespace singa")
			file.close()
