# some_timings_kokkos
Some timing of kokkos experiments. To be launched and profiled with nsight systems

get kokkos and mynvtx
git submodule update --init --recursive

```
mkdir build
cd build
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON .. (for A5000)
make -j 12
```