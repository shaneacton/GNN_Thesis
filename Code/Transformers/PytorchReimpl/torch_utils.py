from typing import List, Optional, Tuple

import torch
from torch import Tensor, _VF
from torch.nn.functional import linear
from torch.overrides import has_torch_function_unary, handle_torch_function


def _pad_circular(input: Tensor, padding: List[int]) -> Tensor:
    """Circularly pads tensor.

    Tensor values at the beginning are used to pad the end, and values at the
    end are used to pad the beginning. For example, consider a single dimension
    with values [0, 1, 2, 3]. With circular padding of (1, 1) it would be
    padded to [3, 0, 1, 2, 3, 0], and with padding (1, 2) it would be padded to
    [3, 0, 1, 2, 3, 0, 1]. If negative padding is applied then the ends of the
    tensor get removed. With circular padding of (-1, -1) the previous example
    would become [1, 2]. Circular padding of (-1, 1) would produce
    [1, 2, 3, 1].

    The first and second dimensions of the tensor are not padded.

    Args:
        input: Tensor with shape :math:`(N, C, D[, H, W])`.
        padding: Tuple containing the number of elements to pad each side of
            the tensor. The length of padding must be twice the number of
            paddable dimensions. For example, the length of padding should be 4
            for a tensor of shape :math:`(N, C, H, W)`, and the length should
            be 6 for a tensor of shape :math:`(N, C, D, H, W)`.

    Examples::

import Code.Transformers.PytorchReimpl.torch_utils        >>> x = torch.tensor([[[[0, 1, 2], [3, 4, 5]]]])  # Create tensor
        >>> # Example 1
        >>> padding = (1, 1, 1, 1)
        >>> y = Code.Transformers.PytorchReimpl.torch_utils.pad(x, padding, mode='circular')
        >>> print(y)
        tensor([[[[5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0]]]])
        >>> print(y.shape)
        torch.Size([1, 1, 4, 5])
        >>> # Example 2
import Code.Transformers.PytorchReimpl.torch_utils
        >>> padding = (1, 1, 2, 2)
        >>> z = Code.Transformers.PytorchReimpl.torch_utils.pad(x, padding, mode='circular')
        >>> print(z)
        tensor([[[[2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3]]]])
        >>> print(z.shape)
        torch.Size([1, 1, 6, 5])
    """
    in_shape = input.shape
    paddable_shape = in_shape[2:]
    ndim = len(paddable_shape)

    for idx, size in enumerate(paddable_shape):
        # Only supports wrapping around once
        assert padding[-(idx * 2 + 1)] <= size, "Padding value causes wrapping around more than once."
        assert padding[-(idx * 2 + 2)] <= size, "Padding value causes wrapping around more than once."
        # Negative padding should not result in negative sizes
        assert (
            padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)] + size >= 0
        ), "Negative padding value is resulting in an empty dimension."

    # Get shape of padded tensor
    out_shape = in_shape[:2]
    for idx, size in enumerate(paddable_shape):
        out_shape += (size + padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)],)

    out = torch.empty(out_shape, dtype=input.dtype, layout=input.layout, device=input.device)

    # Put original array in padded array
    if ndim == 1:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        out[..., out_d0:out_d1] = input[..., in_d0:in_d1]
    elif ndim == 2:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        out_h0 = max(padding[-4], 0)
        out_h1 = out_shape[3] - max(padding[-3], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        in_h0 = max(-padding[-4], 0)
        in_h1 = in_shape[3] - max(-padding[-3], 0)

        out[..., out_d0:out_d1, out_h0:out_h1] = input[..., in_d0:in_d1, in_h0:in_h1]
    elif ndim == 3:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        out_h0 = max(padding[-4], 0)
        out_h1 = out_shape[3] - max(padding[-3], 0)

        out_w0 = max(padding[-6], 0)
        out_w1 = out_shape[4] - max(padding[-5], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        in_h0 = max(-padding[-4], 0)
        in_h1 = in_shape[3] - max(-padding[-3], 0)

        in_w0 = max(-padding[-6], 0)
        in_w1 = in_shape[4] - max(-padding[-5], 0)

        out[..., out_d0:out_d1, out_h0:out_h1, out_w0:out_w1] = input[..., in_d0:in_d1, in_h0:in_h1, in_w0:in_w1]

    # The following steps first pad the beginning of the tensor (left side),
    # and then pad the end of the tensor (right side).
    # Note: Corners will be written more than once when ndim > 1.

    # Only in cases where padding values are > 0 are when additional copying
    # is required.

    # Pad first dimension (depth)
    if padding[-2] > 0:
        i0 = out_shape[2] - padding[-2] - max(padding[-1], 0)
        i1 = out_shape[2] - max(padding[-1], 0)
        o0 = 0
        o1 = padding[-2]
        out[:, :, o0:o1] = out[:, :, i0:i1]
    if padding[-1] > 0:
        i0 = max(padding[-2], 0)
        i1 = max(padding[-2], 0) + padding[-1]
        o0 = out_shape[2] - padding[-1]
        o1 = out_shape[2]
        out[:, :, o0:o1] = out[:, :, i0:i1]

    # Pad second dimension (height)
    if len(padding) > 2:
        if padding[-4] > 0:
            i0 = out_shape[3] - padding[-4] - max(padding[-3], 0)
            i1 = out_shape[3] - max(padding[-3], 0)
            o0 = 0
            o1 = padding[-4]
            out[:, :, :, o0:o1] = out[:, :, :, i0:i1]
        if padding[-3] > 0:
            i0 = max(padding[-4], 0)
            i1 = max(padding[-4], 0) + padding[-3]
            o0 = out_shape[3] - padding[-3]
            o1 = out_shape[3]
            out[:, :, :, o0:o1] = out[:, :, :, i0:i1]

    # Pad third dimension (width)
    if len(padding) > 4:
        if padding[-6] > 0:
            i0 = out_shape[4] - padding[-6] - max(padding[-5], 0)
            i1 = out_shape[4] - max(padding[-5], 0)
            o0 = 0
            o1 = padding[-6]
            out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]
        if padding[-5] > 0:
            i0 = max(padding[-6], 0)
            i1 = max(padding[-6], 0) + padding[-5]
            o0 = out_shape[4] - padding[-5]
            o1 = out_shape[4]
            out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]

    return out


def _pad(input: Tensor, pad: List[int], mode: str = "constant", value: float = 0) -> Tensor:
    r"""Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.

    Padding mode:
        See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
        :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate padding is implemented for padding the last 3 dimensions of 5D input
        tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of
        3D input tensor. Reflect padding is only implemented for padding the last 2
        dimensions of 4D input tensor, or the last dimension of 3D input tensor.

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])

    """
    if has_torch_function_unary(input):
        return handle_torch_function(_pad, (input,), input, pad, mode=mode, value=value)
    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    assert len(pad) // 2 <= input.dim(), "Padding length too large"
    if mode == "constant":
        return _VF.constant_pad_nd(input, pad, value)
    else:
        assert value == 0, 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
        if input.dim() == 3:
            assert len(pad) == 2, "3D tensors expect 2 values for padding"
            if mode == "reflect":
                return torch._C._nn.reflection_pad1d(input, pad)
            elif mode == "replicate":
                return torch._C._nn.replication_pad1d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif input.dim() == 4:
            assert len(pad) == 4, "4D tensors expect 4 values for padding"
            if mode == "reflect":
                return torch._C._nn.reflection_pad2d(input, pad)
            elif mode == "replicate":
                return torch._C._nn.replication_pad2d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif input.dim() == 5:
            assert len(pad) == 6, "5D tensors expect 6 values for padding"
            if mode == "reflect":
                raise NotImplementedError
            elif mode == "replicate":
                return torch._C._nn.replication_pad3d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Only 3D, 4D, 5D padding with non-constant padding are supported for now")


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

pad = _pad