# some_timings_kokkos
Some timing of kokkos experiments. To be launched and profiled with nsight systems

get kokkos and mynvtx:
```
git submodule update --init --recursive
```

Compile for CUDA on the GPU and OpenMP on the CPU:

```
mkdir build
cd build
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON -DKokkos_ENABLE_OPENMP=ON .. (for A5000)
make -j 12
```

launch and profile with Nsight systems:

With the UI
```
nsys-ui profile ./my_program 
```

Without the UI (load the output in Nsight Systems)
```
nsys profile ./my_program 
```