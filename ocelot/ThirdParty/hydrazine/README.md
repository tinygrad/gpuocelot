# Hydrazine

A utility library for GPUOcelot.

## Building

```
mkdir build
cd build
cmake ..
make -j8
```

## Testing

```
cd build
ctests
```

## Known issues

* `TestActiveTimer` deadlocks with `-v` verbose mode enabled
* `TestBTree` crashes

## Components

* **Threading**: A wrapper around pthreads providing a message passing interface rather than a locking based interface.
* **Argument Parser**: A parser for command line arguments.
* **B-Tree**: A replacement for std::map implementing the complete ISO/IEC 14882:2003 standard with a Btree relying on mmapped pages.
* **Debugging**: Conditional debugging messages as well as a more informative version of assert (assert.h).
* **Timer**: Interface to high precision linux timers as well as rdtsc timers on x86 processors.
* **Active Timer**: A wrapper around pthreads providing an asynchronous split-phase interface rather than a locking interface.
* **XML Parser**: Basic parser for XML.

## Original author

* Gregory Daimos
