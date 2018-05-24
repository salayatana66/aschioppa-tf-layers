# Compilation Advice

I was using a Debian on Amd64 with a gcc 4.9.

The first step uses Tensor Flow (here version 1.5 ;) ) to get the compilation flags.

```
export TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
export TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

```

The second step justs compiles to produce a shared library. On my compiler the following is enough:

```
g++ -std=c++11 -shared shardedXEntSfmax.cc -o shardedXEntSfmax.so \
-fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -ltensorflow_framework -O2 -w
```

On some newer compilers the following seems necessary:

```
g++ -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11 -shared shardedXEntSfmax.cc \
-o shardedXEntSfmax.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -ltensorflow_framework -O2
```
