"""A context projection against one snapshot of available per-card memory.

Pure host arithmetic: the architecture supplies its existing allocation recipe
and room, so placement, auto context and a picker report share one calculation.
"""
import math


class ContextFit:
    def __init__(self, native_ctx, devices, slot_mib, room_mib, project, step):
        self.native_ctx, self.step = int(native_ctx), int(step)
        self.devices, self.room_mib = tuple(devices), tuple(float(v) for v in room_mib)
        self.slot_mib, self._project = float(slot_mib), project
        if self.native_ctx < 1 or self.step < 1:
            raise ValueError('context fit: positive native context and step required')
        if not self.devices or len(self.devices) != len(self.room_mib):
            raise ValueError('context fit: one memory budget per device required')
        if not all(math.isfinite(v) for v in self.room_mib) or not math.isfinite(self.slot_mib) or self.slot_mib < 0:
            raise ValueError('context fit: finite memory budgets and nonnegative slot size required')

    def need_mib(self, ctx):
        if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx < 1:
            raise ValueError('context fit: positive integer context required')
        need = tuple(float(v) for v in self._project(ctx))
        if len(need) != len(self.room_mib) or not all(math.isfinite(v) and v >= 0 for v in need):
            raise ValueError('context fit: finite nonnegative projection for every device required')
        return need

    def fits(self, ctx):
        return all(n <= r for n, r in zip(self.need_mib(ctx), self.room_mib))

    def maximum(self):
        """Largest allowed auto step or native endpoint, or None if no candidate fits."""
        if self.fits(self.native_ctx): return self.native_ctx
        below = (self.native_ctx - 1) // self.step * self.step
        return next((c for c in range(below, 0, -self.step) if self.fits(c)), None)

    def report(self):
        maximum = self.maximum()
        return dict(native_ctx=self.native_ctx, max_ctx=maximum, step=self.step,
                    room_mib=list(self.room_mib), slot_mib=self.slot_mib,
                    need_mib=None if maximum is None else list(self.need_mib(maximum)))
