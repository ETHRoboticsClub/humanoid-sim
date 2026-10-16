"""Integer-nanosecond time and demand arrival.

Every time, delay and window in this package is an integer number of nanoseconds. Physics steps are
never counted: a step covers ``[t0_ns, t0_ns + period_ns)`` and anything that happens inside it is
weighted by how much of the step it covers.
"""

from humanoid_sim.timing.arrival import ArrivalQueue
from humanoid_sim.timing.clock import NS_PER_MS, NS_PER_S, NS_PER_US, ns_from_s, s_from_ns

__all__ = ["ArrivalQueue", "NS_PER_MS", "NS_PER_S", "NS_PER_US", "ns_from_s", "s_from_ns"]
