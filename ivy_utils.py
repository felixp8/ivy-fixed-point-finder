import math
import numpy as np
import ivy
from ivy.stateful import Optimizer

from typing import Optional, Literal, Union


class LRScheduler:
    def __init__(
        self,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self._last_lr = None
        self._initialized = False
        self._dev = ivy.default(device, ivy.default_device())
        self._count = ivy.array([0], device=self._dev)
        self._verbose = verbose
    
    def _reset(self, **kwargs):
        pass
    
    def reset(self, **kwargs):
        self._count = ivy.array([0], device=self._dev)
        self._reset(**kwargs)

    def _initialize(self, optimizer: Union[Optimizer, list[Optimizer]], **kwargs):
        raise NotImplementedError
    
    def initialize(self, optimizer: Union[Optimizer, list[Optimizer]], **kwargs):
        self._initialized = True
        self._initialize(optimizer, **kwargs)
    
    def _step(self, optimizer: Union[Optimizer, list[Optimizer]], **kwargs):
        raise NotImplementedError

    def step(self, optimizer: Union[Optimizer, list[Optimizer]], **kwargs):
        assert self._initialized, "Please initialize the LRScheduler before calling `step()`"
        self._count += 1
        self._step(optimizer, **kwargs)

    def set_state(self, state: ivy.Container):
        """
        Set state of the optimizer.

        Parameters
        ----------
        state
            Nested state to update.
        """
        self._last_lr = state.last_lr

    @property
    def state(self):
        return ivy.Container({"last_lr": self._last_lr})


class ReduceLROnPlateau(LRScheduler):
    def __init__(
        self, 
        mode: Literal["min", "max"] = 'min', 
        factor: float = 0.1, 
        patience: int = 10,
        threshold: float = 1e-4, 
        threshold_mode: Literal["rel", "abs"] = 'rel', 
        cooldown: int = 0,
        min_lr: Union[float, list[float]] = 0, 
        eps: float = 1e-8,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        verbose: bool = False,
    ):
        super().__init__(device=device, verbose=verbose)
        assert factor < 1.0 and factor > 0.0, f"`factor` must be > 0 and < 1, got {factor}"
        assert patience >= 0, f"`patience` must be >= 0, got {patience}"
        assert cooldown >= 0, f"`cooldown` must be >= 0, got {cooldown}"
        assert eps > 0, f"`eps` must be > 0, got {eps}"
        assert min_lr >= 0, f"`min_lr` must be >= 0, got {min_lr}"

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.eps = eps
        self.min_lr = min_lr

        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better(
            mode=mode, 
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
        self.reset()
    
    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def _initialize(self, optimizer: Union[Optimizer, list[Optimizer]]):
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        if isinstance(self.min_lr, list):
            assert len(optimizer) == len(self.min_lr), (
                "if `min_lr` is a list, its length should match the length of `optimizer`. "
                f"Expected: {len(optimizer)}, got: {len(self.min_lr)}"
            )
        self._last_lr = [optim._lr for optim in optimizer]
    
    def _step(
        self, 
        optimizer: Union[Optimizer, list[Optimizer]], 
        metrics: float,
    ):
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(optimizer)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [optim._lr for optim in optimizer]

    def _reduce_lr(self, optimizer: Union[Optimizer, list[Optimizer]]):
        if not isinstance(optimizer, list):
            optimizer = [optimizer]

        for i, optim in enumerate(optimizer):
            min_lr = self.min_lr[i] if isinstance(self.min_lr, list) else self.min_lr
            old_lr = optim._lr
            new_lr = max(old_lr * self.factor, min_lr)
            if old_lr - new_lr > self.eps:
                optim._lr = new_lr
                if self._verbose:
                    step_str = str(int(self._count[0]))
                    print(f'Step {step_str}: reducing learning rate of optimizer {i} to {new_lr:.4e}.')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -math.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class AdaptiveLearningRate(LRScheduler):
    def __init__(
        self,
        mode: Literal["min", "max"] = 'min', 
        do_decrease_rate: bool = True,
        decrease_factor: float = 0.95, 
        decrease_patience: int = 5,
        do_increase_rate: bool = True,
        increase_factor: float = 1/0.95, 
        increase_patience: int = 5,
        min_lr: Union[float, list[float]] = 0, 
        eps: float = 1e-8,
		n_warmup_steps: int = 0,
		warmup_scale: float = 1e-3,
		warmup_shape: Literal["linear", "exp", "gaussian"] = "gaussian",
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        verbose: bool = False,
    ):
        super().__init__(device=device, verbose=verbose)
        assert decrease_factor < 1.0 and decrease_factor > 0.0, f"`decrease_factor` must be > 0 and < 1, got {decrease_factor}"
        assert decrease_patience >= 0, f"`decrease_patience` must be >= 0, got {decrease_patience}"
        assert increase_factor > 1.0, f"`increase_factor` must be > 1, got {increase_factor}"
        assert increase_patience >= 0, f"`decrease_patience` must be >= 0, got {increase_patience}"
        assert eps > 0, f"`eps` must be > 0, got {eps}"
        assert min_lr >= 0, f"`min_lr` must be >= 0, got {min_lr}"

        self.mode = mode
        self.do_decrease_rate = do_decrease_rate
        self.decrease_factor = decrease_factor
        self.decrease_patience = decrease_patience
        self.do_increase_rate = do_increase_rate
        self.increase_factor = increase_factor
        self.increase_patience = increase_patience
        self.eps = eps
        self.min_lr = min_lr

        self.n_warmup_steps = n_warmup_steps
        self.warmup_scale = warmup_scale
        self.warmup_shape = warmup_shape

        self.last_decrease = -1
        self.last_increase = -1

        self.metric_log = []
        self.mode_worse = None  # the worse value for the chosen mode
        self.reset()
    
    def _reset(self):
        self.worst = self.mode_worse
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def _initialize(self, optimizer: Union[Optimizer, list[Optimizer]]):
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        if isinstance(self.min_lr, list):
            assert len(optimizer) == len(self.min_lr), (
                "if `min_lr` is a list, its length should match the length of `optimizer`. "
                f"Expected: {len(optimizer)}, got: {len(self.min_lr)}"
            )
        self._last_lr = [optim._lr for optim in optimizer]
        if self.n_warmup_steps > 0:
            self.warmup_rates = self._get_warmup_rates()
            init_lr = self.warmup_rates[0].tolist()
            for i, optim in enumerate(optimizer):
                optim._lr = init_lr[i]
            self._last_lr = init_lr

    def _get_warmup_rates(self):
        n = self.n_warmup_steps
        warmup_shape = self.warmup_shape
        scale = self.warmup_scale
        warmup_start = scale*np.asarray(self._last_lr)
        warmup_stop = np.asarray(self._last_lr)

        if warmup_shape == 'linear':
            warmup_rates = np.linspace(warmup_start, warmup_stop, n+1)
        if self.warmup_shape == 'exp':
            warmup_rates = np.logspace(
                np.log10(warmup_start), np.log10(warmup_stop), n+1)
        elif self.warmup_shape == 'gaussian':
            mu = np.float32(n)
            x = np.arange(mu + 1)
            # solve for sigma s.t. warmup_rates[0] = warmup_start
            sigma = np.sqrt(-mu**2.0 / (2.0*np.log(warmup_start/warmup_stop)))
            warmup_rates = warmup_stop[None,:]*np.exp((-(x[:,None]-mu)**2.0)/(2.0*sigma[None,:]**2.0))
        else:
            raise ValueError(f"Warmup shape {self.warmup_shape} not recognized")

        return warmup_rates

    def _update_metric_log(self, metrics):
        while len(self.metric_log) >= max(max(self.decrease_patience, self.increase_patience), 1):
            self.metric_log.pop(0)
        self.metric_log.append(metrics)
    
    def _step(
        self, 
        optimizer: Union[Optimizer, list[Optimizer]], 
        metrics: float,
    ):
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        
        if int(self._count[0]) <= self.n_warmup_steps:
            warmup_lr = self.warmup_rates[int(self._count[0])].tolist()
            for i, optim in enumerate(optimizer):
                optim._lr = warmup_lr[i]
            self._last_lr = warmup_lr
            return
        
        current = float(metrics)
        decreased_rate = False
        if self.do_decrease_rate:
            if int(self._count[0]) > self.last_decrease + self.decrease_patience:
                metric_window = self.metric_log[-self.decrease_patience:]
                compare_func = np.greater if self.mode == "min" else np.less
                if np.all(compare_func(current, metric_window)):
                    self._reduce_lr(optimizer)
                    self.last_decrease = int(self._count[0])
                    decreased_rate = True
        if not decreased_rate and self.do_increase_rate:
            if int(self._count[0]) > self.last_increase + self.increase_patience:
                metric_window = self.metric_log[-self.increase_patience:]
                compare_func = np.less if self.mode == "min" else np.greater
                if np.all(compare_func(metric_window[1:] + [current], metric_window)):
                    self._increase_lr(optimizer)
                    self.last_increase = int(self._count[0])        

        self._last_lr = [optim._lr for optim in optimizer]

    def _reduce_lr(self, optimizer: Union[Optimizer, list[Optimizer]]):
        if not isinstance(optimizer, list):
            optimizer = [optimizer]

        for i, optim in enumerate(optimizer):
            min_lr = self.min_lr[i] if isinstance(self.min_lr, list) else self.min_lr
            old_lr = optim._lr
            new_lr = max(old_lr * self.decrease_factor, min_lr)
            if old_lr - new_lr > self.eps:
                optim._lr = new_lr
                if self._verbose:
                    step_str = str(int(self._count[0]))
                    print(f'Step {step_str}: reducing learning rate of optimizer {i} to {new_lr:.4e}.')
                    
    def _increase_lr(self, optimizer: Union[Optimizer, list[Optimizer]]):
        if not isinstance(optimizer, list):
            optimizer = [optimizer]

        for i, optim in enumerate(optimizer):
            old_lr = optim._lr
            new_lr = old_lr * self.increase_factor # no upper limit, for now
            if new_lr - old_lr > self.eps:
                optim._lr = new_lr
                if self._verbose:
                    step_str = str(int(self._count[0]))
                    print(f'Step {step_str}: increasing learning rate of optimizer {i} to {new_lr:.4e}.')


class AdaptiveGradNormClip:
    def __init__(
        self,
        do_adaptive_clipping: bool = True,
        sliding_window_len: int = 128,
        percentile: float = 0.95,
        init_clip_val: float = 1e12,
        max_clip_val: float = 1e12,
        verbose: bool = False,
    ):
        super().__init__()
        self.step = 0
        self.do_adaptive_clipping = do_adaptive_clipping
        self.sliding_window_len = sliding_window_len
        self.percentile = percentile
        self.max_clip_val = max_clip_val
        self.grad_norm_log = []
        self.verbose = verbose

        self.clip_val = init_clip_val if self.do_adaptive_clipping else max_clip_val
    
    def update(self, grad_norm: float):
        if self.do_adaptive_clipping:
            if self.step < self.sliding_window_len:
                # First fill up an entire "window" of values
                self.grad_norm_log.append(grad_norm)
            else:
                # Once the window is full, overwrite the oldest value
                idx = np.mod(self.step, self.sliding_window_len)
                self.grad_norm_log[idx] = grad_norm

            proposed_clip_val = \
                np.percentile(self.grad_norm_log, self.percentile)

            self.clip_val = min(proposed_clip_val, self.max_clip_val)

        self.step += 1