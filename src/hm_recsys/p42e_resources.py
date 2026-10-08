"""Read-only Win32 resource counters, no additional dependency."""
import ctypes


def memory():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    value=Status(); value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('GlobalMemoryStatusEx failed')
    return value


def rss():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    c=Counters(); c.cb=ctypes.sizeof(c)
    handle=ctypes.windll.kernel32.GetCurrentProcess
    handle.restype=ctypes.c_void_p
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle()),ctypes.byref(c),c.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return int(c.peak)
