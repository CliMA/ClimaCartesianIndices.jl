# ClimaCartesianIndices.jl

This package implements a type called `FastCartesianIndices`, which is a drop-in replacement for `CartesianIndices`. `FastCartesianIndices` uses Julia Base's
`SignedMultiplicativeInverse` to avoid integer division when converting
linear indexes to cartesian indexes. This is especially useful on the gpu,
which can have a nearly 2x performance impact.
