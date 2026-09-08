from __future__ import annotations

import enum
from typing import Any, Callable

import PIL.Image
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.transforms.v2._utils import check_type, has_any, is_pure_tensor
from torchvision.utils import _log_api_usage_once

from .functional._utils import _get_kernel


class Transform(nn.Module):
    """Base class to implement your own v2 transforms.

    See  :ref:`sphx_glr_auto_examples_transforms_plot_custom_transforms.py` for
    more details.
    """

    # Class attribute defining transformed types. Other types are passed-through without any transformation
    # We support both Types and callables that are able to do further checks on the type of the input.
    _transformed_types: tuple[type | Callable[[Any], bool], ...] = (torch.Tensor, PIL.Image.Image)

    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    def check_inputs(self, flat_inputs: list[Any]) -> None:
        pass

    # When v2 was introduced, this method was private and called
    # `_get_params()`. Now it's publicly exposed as `make_params()`. It cannot
    # be exposed as `get_params()` because there is already a `get_params()`
    # methods for v2 transforms: it's the v1's `get_params()` that we have  to
    # keep in order to guarantee 100% BC with v1. (It's defined in
    # __init_subclass__ below).
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Method to override for custom transforms.

        See :ref:`sphx_glr_auto_examples_transforms_plot_custom_transforms.py`"""
        return dict()

    def _call_kernel(self, functional: Callable, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = _get_kernel(functional, type(inpt), allow_passthrough=True)
        return kernel(inpt, *args, **kwargs)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Method to override for custom transforms.

        See :ref:`sphx_glr_auto_examples_transforms_plot_custom_transforms.py`"""
        raise NotImplementedError

    def forward(self, *inputs: Any) -> Any:
        """Do not override this! Use ``transform()`` instead."""
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        self.check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self.make_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self.transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _needs_transform_list(self, flat_inputs: list[Any]) -> list[bool]:
        # Below is a heuristic on how to deal with pure tensor inputs:
        # 1. Pure tensors, i.e. tensors that are not a tv_tensor, are passed through if there is an explicit image
        #    (`tv_tensors.Image` or `PIL.Image.Image`) or video (`tv_tensors.Video`) in the sample.
        # 2. If there is no explicit image or video in the sample, only the first encountered pure tensor is
        #    transformed as image, while the rest is passed through. The order is defined by the returned `flat_inputs`
        #    of `tree_flatten`, which recurses depth-first through the input.
        #
        # This heuristic stems from two requirements:
        # 1. We need to keep BC for single input pure tensors and treat them as images.
        # 2. We don't want to treat all pure tensors as images, because some datasets like `CelebA` or `Widerface`
        #    return supplemental numerical data as tensors that cannot be transformed as images.
        #
        # The heuristic should work well for most people in practice. The only case where it doesn't is if someone
        # tries to transform multiple pure tensors at the same time, expecting them all to be treated as images.
        # However, this case wasn't supported by transforms v1 either, so there is no BC concern.

        needs_transform_list = []
        transform_pure_tensor = not has_any(flat_inputs, tv_tensors.Image, tv_tensors.Video, PIL.Image.Image)
        for inpt in flat_inputs:
            needs_transform = True

            if not check_type(inpt, self._transformed_types):
                needs_transform = False
            elif is_pure_tensor(inpt):
                if transform_pure_tensor:
                    transform_pure_tensor = False
                else:
                    needs_transform = False
            needs_transform_list.append(needs_transform)
        return needs_transform_list

    def extra_repr(self) -> str:
        extra = []
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)

    # If the v1 transform has a static `get_params` method, expose it under the same name on the v2 transform.
    _v1_transform_cls: type[nn.Module] | None = None

    def __init_subclass__(cls) -> None:
        # Since `get_params` is a `@staticmethod`, we have to bind it to the class itself rather than to an instance.
        # This method is called after subclassing has happened, i.e. `cls` is the subclass, e.g. `Resize`.
        if cls._v1_transform_cls is not None and hasattr(cls._v1_transform_cls, "get_params"):
            cls.get_params = staticmethod(cls._v1_transform_cls.get_params)  # type: ignore[attr-defined]


class _RandomApplyTransform(Transform):
    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")

        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self.check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self.check_inputs(flat_inputs)

        if torch.rand(1) >= self.p:
            return inputs

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self.make_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self.transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)
