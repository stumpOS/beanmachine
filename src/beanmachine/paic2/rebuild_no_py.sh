rm -r build
mkdir build
cd build
cmake ../ -G Ninja -DCMAKE_CXX_COMPILER=/usr/local/compiler/clang+llvm-14.0.6-x86_64-apple-darwin/bin/clang++ -DBM_ROOT=/Users/stumpos/code/bm_2/beanmachine -DMLIR_DIR=/Users/stumpos/code/bm_2/beanmachine/externals/llvm-project/build/lib/cmake/mlir -DLLVM_DIR=/Users/stumpos/code/bm_2/beanmachine/externals/llvm-project/build/lib/cmake/llvm -DPYTHON_EXECUTABLE=/usr/local/Caskroom/miniconda/base/envs/bean-machine-9/bin/python
cmake --build .
