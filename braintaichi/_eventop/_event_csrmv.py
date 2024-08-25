# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

"""

Key points for the operator customization:

1. `index` has two kinds of types: int32, int64
2. `data` has two kinds of types: float32, float64
3. `events` has three kinds of types: bool (True or False), float32, float64

"""

from typing import Union, Tuple

import jax
import jax.numpy as jnp
import taichi as ti
from jax.interpreters import ad

from braintaichi._primitive._xla_custom_op import XLACustomOp
from braintaichi._sparseop._sparse_csrmv import raw_csrmv_taichi as normal_csrmv_taichi
from braintaichi._sparseop._sparse_utils import csr_to_coo


def raw_csrmv_taichi(
    data: Union[float, jax.Array],
    indices: jax.Array,
    indptr: jax.Array,
    events: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False
):
  if transpose:
    if events.dtype == jnp.bool_:
      if data.shape[0] == 1:
        prim = _event_csrmv_transpose_bool_homo_p
      else:
        prim = _event_csrmv_transpose_bool_heter_p
    else:
      return normal_csrmv_taichi(data, indices, indptr, events, shape=shape, transpose=transpose)
  else:
    if events.dtype == jnp.bool_:
      if data.shape[0] == 1:
        prim = _event_csrmv_bool_homo_p
      else:
        prim = _event_csrmv_bool_heter_p
    else:
      return normal_csrmv_taichi(data, indices, indptr, events, shape=shape, transpose=transpose)

  # computing
  return prim(data,
              indices,
              indptr,
              events,
              outs=[jax.ShapeDtypeStruct(shape=(shape[1] if transpose else shape[0],), dtype=data.dtype)],
              transpose=transpose,
              shape=shape)


# -------------
# CPU operators
# -------------

# 1. The benchmarking shows that the performance of the following transpose
#    kernels is maximized when using serialized mode
# 2. Since our Taichi-JAX kernel does not support the non-differentiable/non-jittable
#    arguments, we have to define each kernel separately when the
#    non-differentiable/non-jittable arguments are different.

@ti.kernel
def _event_csr_matvec_transpose_bool_homo_cpu(values: ti.types.ndarray(ndim=1),
                                              indices: ti.types.ndarray(ndim=1),
                                              indptr: ti.types.ndarray(ndim=1),
                                              events: ti.types.ndarray(ndim=1),
                                              out: ti.types.ndarray(ndim=1)):
  value = values[0]
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i]:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += value


@ti.kernel
def _event_csr_matvec_transpose_bool_heter_cpu(values: ti.types.ndarray(ndim=1),
                                               indices: ti.types.ndarray(ndim=1),
                                               indptr: ti.types.ndarray(ndim=1),
                                               events: ti.types.ndarray(ndim=1),
                                               out: ti.types.ndarray(ndim=1)):
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i]:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += values[j]


@ti.kernel
def _event_csr_matvec_transpose_homo_cpu(values: ti.types.ndarray(ndim=1),
                                         indices: ti.types.ndarray(ndim=1),
                                         indptr: ti.types.ndarray(ndim=1),
                                         events: ti.types.ndarray(ndim=1),
                                         out: ti.types.ndarray(ndim=1)):
  value = values[0]
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i] != 0.:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += value


@ti.kernel
def _event_csr_matvec_transpose_heter_cpu(values: ti.types.ndarray(ndim=1),
                                          indices: ti.types.ndarray(ndim=1),
                                          indptr: ti.types.ndarray(ndim=1),
                                          events: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i] != 0.:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += values[j]


@ti.kernel
def _event_csr_matvec_bool_homo_cpu(values: ti.types.ndarray(ndim=1),
                                    indices: ti.types.ndarray(ndim=1),
                                    indptr: ti.types.ndarray(ndim=1),
                                    events: ti.types.ndarray(ndim=1),
                                    out: ti.types.ndarray(ndim=1)):
  value = values[0]
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]]:
        r += value
    out[row_i] = r


@ti.kernel
def _event_csr_matvec_bool_heter_cpu(values: ti.types.ndarray(ndim=1),
                                     indices: ti.types.ndarray(ndim=1),
                                     indptr: ti.types.ndarray(ndim=1),
                                     events: ti.types.ndarray(ndim=1),
                                     out: ti.types.ndarray(ndim=1)):
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]]:
        r += values[j]
    out[row_i] = r


@ti.kernel
def _event_csr_matvec_homo_cpu(values: ti.types.ndarray(ndim=1),
                               indices: ti.types.ndarray(ndim=1),
                               indptr: ti.types.ndarray(ndim=1),
                               events: ti.types.ndarray(ndim=1),
                               out: ti.types.ndarray(ndim=1)):
  value = values[0]
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]] != 0.:
        r += value
    out[row_i] = r


@ti.kernel
def _event_csr_matvec_heter_cpu(values: ti.types.ndarray(ndim=1),
                                indices: ti.types.ndarray(ndim=1),
                                indptr: ti.types.ndarray(ndim=1),
                                events: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]] != 0.:
        r += values[j]
    out[row_i] = r


# -------------
# GPU operators
# -------------

# 1. GPU kernels are different from the CPU ones, since the GPU kernels need
#    to use warp-level parallelism to achieve the best performance.

@ti.kernel
def _event_csr_matvec_transpose_bool_homo_gpu(values: ti.types.ndarray(ndim=1),
                                              indices: ti.types.ndarray(ndim=1),
                                              indptr: ti.types.ndarray(ndim=1),
                                              events: ti.types.ndarray(ndim=1),
                                              out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i]:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += value
        j += 32


@ti.kernel
def _event_csr_matvec_transpose_homo_gpu(values: ti.types.ndarray(ndim=1),
                                         indices: ti.types.ndarray(ndim=1),
                                         indptr: ti.types.ndarray(ndim=1),
                                         events: ti.types.ndarray(ndim=1),
                                         out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i] != 0.:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += value
        j += 32


# TODO
# It is important to note that the following warp-based kernels
# should be improved, since the atomic_add for each thread is not
# very efficient. Instead, the warp-level reduction primitive
# should be used.
# see ``warp_reduce_sum()`` function in tifunc.py.
# However, currently Taichi does not support general warp-level primitives.

@ti.kernel
def _event_csr_matvec_bool_homo_gpu(values: ti.types.ndarray(ndim=1),
                                    indices: ti.types.ndarray(ndim=1),
                                    indptr: ti.types.ndarray(ndim=1),
                                    events: ti.types.ndarray(ndim=1),
                                    out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]]:
        r += value
      j += 32
    out[row_i] += r  # TODO: warp-level primitive


@ti.kernel
def _event_csr_matvec_homo_gpu(values: ti.types.ndarray(ndim=1),
                               indices: ti.types.ndarray(ndim=1),
                               indptr: ti.types.ndarray(ndim=1),
                               events: ti.types.ndarray(ndim=1),
                               out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]] != 0.:
        r += value
      j += 32
    out[row_i] += r  # TODO: warp-level primitive


@ti.kernel
def _event_csr_matvec_transpose_bool_heter_gpu(values: ti.types.ndarray(ndim=1),
                                               indices: ti.types.ndarray(ndim=1),
                                               indptr: ti.types.ndarray(ndim=1),
                                               events: ti.types.ndarray(ndim=1),
                                               out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i]:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += values[j]
        j += 32


@ti.kernel
def _event_csr_matvec_transpose_heter_gpu(values: ti.types.ndarray(ndim=1),
                                          indices: ti.types.ndarray(ndim=1),
                                          indptr: ti.types.ndarray(ndim=1),
                                          events: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i] != 0.:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += values[j]
        j += 32


@ti.kernel
def _event_csr_matvec_bool_heter_gpu(values: ti.types.ndarray(ndim=1),
                                     indices: ti.types.ndarray(ndim=1),
                                     indptr: ti.types.ndarray(ndim=1),
                                     events: ti.types.ndarray(ndim=1),
                                     out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]]:
        r += values[j]
      j += 32
    out[row_i] += r  # TODO: warp-level primitive


@ti.kernel
def _event_csr_matvec_heter_gpu(values: ti.types.ndarray(ndim=1),
                                indices: ti.types.ndarray(ndim=1),
                                indptr: ti.types.ndarray(ndim=1),
                                events: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]] != 0.:
        r += values[j]
      j += 32
    out[row_i] += r  # TODO: warp-level primitive


def _event_csr_matvec_jvp_values_taichi(val_dot, values, indices, indptr, events, *, outs, transpose, shape):
  return normal_csrmv_taichi(val_dot, indices, indptr, events, shape=shape, transpose=transpose)


def _event_csr_matvec_jvp_events_taichi(evt_dot, values, indices, indptr, events, *, outs, transpose, shape):
  return normal_csrmv_taichi(values, indices, indptr, evt_dot, shape=shape, transpose=transpose)


def _event_csr_matvec_transpose_taichi(
    ct, values, indices, indptr, events, *, outs, transpose, shape
):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")
  if ad.is_undefined_primal(events):
    ct_events = normal_csrmv_taichi(values, indices, indptr, ct[0], shape=shape, transpose=transpose)[0]
    return values, indices, indptr, (ad.Zero(events) if type(ct[0]) is ad.Zero else ct_events)
  else:
    if type(ct[0]) is ad.Zero:
      ct_values = ad.Zero(values)
    else:
      if values.aval.shape[0] == 1:  # scalar
        ct_values = raw_csrmv_taichi(jnp.ones(1), indices, indptr, events, shape=shape, transpose=transpose)[0]
        ct_values = jnp.inner(ct[0], ct_values)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_values = events[row] * ct[0][col] if transpose else events[col] * ct[0][row]
    return ct_values, indices, indptr, events


def _define_op(cpu_kernel, gpu_kernel):
  prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
  prim.defjvp(_event_csr_matvec_jvp_values_taichi, None, None, _event_csr_matvec_jvp_events_taichi)
  prim.def_transpose_rule(_event_csr_matvec_transpose_taichi)
  return prim


# transpose bool homo
_event_csrmv_transpose_bool_homo_p = _define_op(_event_csr_matvec_transpose_bool_homo_cpu,
                                                _event_csr_matvec_transpose_bool_homo_gpu)

# transpose homo
_event_csrmv_transpose_homo_p = _define_op(_event_csr_matvec_transpose_homo_cpu,
                                           _event_csr_matvec_transpose_homo_gpu)

# not transpose bool homo
_event_csrmv_bool_homo_p = _define_op(_event_csr_matvec_bool_homo_cpu,
                                      _event_csr_matvec_bool_homo_gpu)

# not transpose homo
_event_csrmv_homo_p = _define_op(_event_csr_matvec_homo_cpu,
                                 _event_csr_matvec_homo_gpu)

# transpose bool heter
_event_csrmv_transpose_bool_heter_p = _define_op(_event_csr_matvec_transpose_bool_heter_cpu,
                                                 _event_csr_matvec_transpose_bool_heter_gpu)

# transpose heter
_event_csrmv_transpose_heter_p = _define_op(_event_csr_matvec_transpose_heter_cpu,
                                            _event_csr_matvec_transpose_heter_gpu)

# not transpose bool heter
_event_csrmv_bool_heter_p = _define_op(_event_csr_matvec_bool_heter_cpu,
                                       _event_csr_matvec_bool_heter_gpu)

# not transpose heter
_event_csrmv_heter_p = _define_op(_event_csr_matvec_heter_cpu,
                                  _event_csr_matvec_heter_gpu)
