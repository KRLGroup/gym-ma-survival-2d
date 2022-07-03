from typing import TypeVar, Any, Union, Tuple, Callable, List, Dict, Optional, no_type_check

import torch


# A simple fixed-size buffer that also supports appending batches of arbitrary size, batched along arbitrary axes.
class BatchBuffer:

    max_size: int
    batch_axis: int
    dtype: torch.dtype
    size: int
    buffer: torch.Tensor

    def __init__(self, shape: Union[Tuple[int], int, List[int]], batch_axis: int = 0, dtype: torch.dtype = torch.float32):
        self.max_size = shape if isinstance(shape, int) else shape[batch_axis]
        self.batch_axis = batch_axis
        self.dtype = dtype
        self.buffer = torch.zeros(shape, dtype=dtype)
        self.size = 0

    @property
    def full(self) -> bool:
        assert self.size <= self.max_size
        return self.size == self.max_size

    # Simply returns a copy of the buffer and resets the size to 0; does *NOT* delete previous data.
    def flush(self, return_copy: bool = False) -> Optional[torch.Tensor]:
        self.size = 0
        if return_copy:
            return self.buffer.clone().detach()
        return None

    #TODO annotate with np.ArrayLike or smth
    def append(self, batch: Any, batch_axis: Optional[int] = None):
        batch_axis = batch_axis if batch_axis is not None else self.batch_axis
        if isinstance(batch, BatchBuffer):
            batch = batch.buffer
        batch = torch.as_tensor(batch, dtype=self.dtype)
        batch_size = batch.shape[batch_axis]
        assert self.size + batch_size <= self.max_size, f'batch size {batch_size} too large for buffer of size {self.size} and max_size {self.max_size}'
        batch_indices = [slice(None) for _ in self.buffer.shape]
        batch_slice = slice(self.size, self.size + batch_size)
        batch_indices[batch_axis] = batch_slice
        self.buffer[tuple(batch_indices)] = batch
        self.size += batch_size
