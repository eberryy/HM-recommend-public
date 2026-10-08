"""Current process/system counters, distinct from historical peak RSS."""
import ctypes
import os
from .p42e_resources import memory


def snapshot():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
    c=Counters();c.cb=ctypes.sizeof(c)
    handle=ctypes.windll.kernel32.GetCurrentProcess
    handle.restype=ctypes.c_void_p
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle()),ctypes.byref(c),c.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return dict(pid=os.getpid(),current_working_gib=c.working/2**30,
        peak_working_gib=c.peak/2**30,private_commit_gib=c.private/2**30,
        system_available_gib=memory().available/2**30)
