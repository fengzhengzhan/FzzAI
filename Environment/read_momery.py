import os

from dependencies import *


class ReadMemory:
    def __init__(self, name_process):
        """
        kernel32.ReadProcessMemory是一个 Windows API 函数，用于读取指定进程的内存区域中的数据。
        这个函数只能用于读取其他进程的内存，如果你想要读取当前进程的内存，应该使用标准的内存读取函数，如 memcpy 或 std::copy。
        Parameters:
            hProcess：要读取内存的进程的句柄。
            lpBaseAddress：要读取的内存区域的起始地址。该参数是一个指向被读取内存区域的指针，必须指向进程的虚拟地址空间中可读取的内存页。
            lpBuffer：指向缓冲区的指针，该缓冲区将包含读取的数据。
            nSize：要读取的字节数。
            lpNumberOfBytesRead：指向变量的指针，该变量将接收读取的字节数。如果该值为 NULL，则忽略该值。
        Return:
            ReadProcessMemory 函数返回一个布尔值，指示是否成功读取指定进程的内存区域中的数据。

        """
        self.psapi = ctypes.WinDLL('Psapi.dll')
        self.kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        # self.kernel32 = ctypes.WinDLL('kernel32.dll')
        # self.kernal32 = ctypes.windll.LoadLibrary(r"C:\\Windows\\System32\\kernel32.dll")

        self.PROCESS_QUERY_INFORMATION = 0x0400
        self.PROCESS_VM_READ = 0x0010
        self.LIST_MODULES_ALL = 0x03

        window_handle = win32gui.FindWindow(None, name_process)
        self.pid = win32process.GetWindowThreadProcessId(window_handle)[1]

        openProcess = self.kernel32.OpenProcess
        openProcess.argtypes = ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD
        openProcess.restype = ctypes.wintypes.HANDLE
        self.process_handle = openProcess(0x1F0FFF, False, self.pid)  # 第一个参数是权限参数，0x1F0FFF高权限
        # self.process_handle = win32api.openProcess(0x1F0FFF, False, self.pid)
        # openProcess(0x10, False, self.pid)

        self.readProcessMemory = self.kernel32.ReadProcessMemory
        self.readProcessMemory.argtypes = ctypes.wintypes.HANDLE, ctypes.wintypes.LPCVOID, ctypes.wintypes.LPVOID, \
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
        self.readProcessMemory.restype = ctypes.wintypes.BOOL

        self.closeHandle = self.kernel32.CloseHandle
        self.closeHandle.argtypes = [ctypes.wintypes.HANDLE]
        self.closeHandle.restype = ctypes.wintypes.BOOL

        # get dll address
        self.hProcess = self.kernel32.OpenProcess(
            self.PROCESS_QUERY_INFORMATION | self.PROCESS_VM_READ, False, self.pid)
        self.hModule = self.enumProcessModulesEx(self.hProcess)
        self.map_dll = {}
        for i in self.hModule:
            temp = win32process.GetModuleFileNameEx(self.process_handle, i.value)
            # projlog(DEBUG, "{} : {} ".format(str(temp), str(hex(i.value))))
            self.map_dll[os.path.basename(temp)] = i.value

    def __del__(self):
        self.closeHandle(self.process_handle)

    ######################
    # 获得运行状态
    ######################
    def travelProcess(self):
        array_name_process = []

        def __enum_windows(hwnd, lParam):
            if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
                array_name_process.append({hwnd: win32gui.GetWindowText(hwnd)})

        win32gui.EnumWindows(__enum_windows, 0)
        # for np in array_name_process:
        #     print(np)
        return array_name_process

    def travelDll(self):
        return self.map_dll

    def enumProcesses(self):
        buf_count = 256
        while True:
            buf = (ctypes.wintypes.DWORD * buf_count)()
            buf_size = ctypes.sizeof(buf)
            res_size = ctypes.wintypes.DWORD()
            if not self.psapi.EnumProcesses(ctypes.byref(buf), buf_size, ctypes.byref(res_size)):
                raise OSError('EnumProcesses failed')
            if res_size.value >= buf_size:
                buf_count *= 2
                continue
            count = res_size.value // (buf_size // buf_count)
            return buf[:count]

    def enumProcessModulesEx(self, hProcess):
        buf_count = 256
        while True:
            buf = (ctypes.wintypes.HMODULE * buf_count)()
            buf_size = ctypes.sizeof(buf)
            needed = ctypes.wintypes.DWORD()
            if not self.psapi.EnumProcessModulesEx(
                    hProcess, ctypes.byref(buf), buf_size, ctypes.byref(needed), self.LIST_MODULES_ALL):
                raise OSError('EnumProcessModulesEx failed')
            if buf_size < needed.value:
                buf_count = needed.value // (buf_size // buf_count)
                continue
            count = needed.value // (buf_size // buf_count)
            return map(ctypes.wintypes.HMODULE, buf[:count])

    def getModuleFileNameEx(self, hProcess, hModule):
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        nSize = ctypes.wintypes.DWORD()
        if not self.psapi.GetModuleFileNameExW(hProcess, hModule, ctypes.byref(buf), ctypes.byref(nSize)):
            raise OSError('GetModuleFileNameEx failed')
        return buf.value

    def getProcessModules(self, pid):
        hProcess = self.kernel32.OpenProcess(self.PROCESS_QUERY_INFORMATION | self.PROCESS_VM_READ, False, pid)
        if not hProcess:
            raise OSError('Could not open PID %s' % pid)
        try:
            return [self.getModuleFileNameEx(hProcess, hModule) for hModule in self.enumProcessModulesEx(hProcess)]
        finally:
            self.kernel32.CloseHandle(hProcess)

    ######################
    # 访问内存
    ######################
    def __gainValueFromOneaddrX32(self, address):
        data = ctypes.c_long()
        self.kernel32.ReadProcessMemory(int(self.process_handle), address, ctypes.byref(data), 4, None)
        return data.value

    def __gainValueFromOneaddrX64(self, address):
        data = ctypes.c_ulonglong()
        bytes_read = ctypes.c_ulonglong()
        result = self.readProcessMemory(
            self.process_handle, address, ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytes_read))
        e = ctypes.get_last_error()

        projlog(DEBUG, 'result: {}, err code: {}, bytes_read: {}'.format(result, e, bytes_read.value))
        projlog(INFO, 'data: {:016X}h'.format(data.value))
        return hex(data.value)

    def gainValueFromOneaddr(self, address):
        """根据地址长度调用相应的函数，短地址调用X32，长地址调用X64"""
        return self.__gainValueFromOneaddrX64(address)

    def __gainValueFromMuladdrX32(self, library_address, baseoffset_address: int, array_offset: list):
        base_address = library_address + baseoffset_address
        data = ctypes.c_long()
        self.kernel32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(data), 4, None)
        for each_offset in array_offset:
            self.kernel32.ReadProcessMemory(
                int(self.process_handle), data.value + each_offset, ctypes.byref(data), 4, None)
        return data.value

    def __gainValueFromMuladdrX64(self, library_address, baseoffset_address: int, array_offset: list):
        base_address = library_address + baseoffset_address
        projlog(DEBUG, "base_address: {}".format(hex(library_address + baseoffset_address)))

        data = ctypes.c_ulonglong()
        bytes_read = ctypes.c_ulonglong()
        self.readProcessMemory(self.process_handle, base_address,
                               ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytes_read))
        for each_offset in array_offset:
            self.readProcessMemory(self.process_handle, data.value + each_offset,
                                   ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytes_read))

        projlog(DEBUG, 'bytes_read: {}'.format(bytes_read.value))
        projlog(INFO, 'data: {:016X}h'.format(data.value))
        return hex(data.value)

    def gainValueFromMuladdr(self, library_name: str, baseoffset_address: int, array_offset: list):
        if library_name not in self.map_dll:
            projlog(DEBUG, self.map_dll)
            projlog(ERROR, "Unable to find the dynamic link library, please check the library name.")

        return self.__gainValueFromMuladdrX64(self.map_dll[library_name], baseoffset_address, array_offset)


if __name__ == '__main__':
    # score = ReadMemory("Super Hexagon")
    # base_address = 0x00691048  # 0x00691048  0x00694B00  0x00694B8C
    # offset_address = 0x2988
    score = ReadMemory("Hollow Knight")
    print(score.travelDll())

    # 获得角色血量
    # print(score.gainValueFromMuladdr("mono-2.0-bdwgc.dll", 0x00497DE8, [0x90, 0xE08, 0x48, 0x70, 0x68, 0x218, 0x190]))
    # print(score.gainValueFromMuladdr("mono-2.0-bdwgc.dll", 0x00497DE8, [0x190, 0x218, 0x68, 0x70, 0x48, 0xE08, 0x90]))
    # print(score.gainValueFromMuladdr("UnityPlayer.dll", 0x019B8BE0, [0x10, 0x88, 0x28, 0x150, 0x68, 0x218, 0x190]))
    print(score.gainValueFromMuladdr("UnityPlayer.dll", 0x019D7CF0, [0x10, 0x100, 0x28, 0x68, 0x218, 0x190]))

    # print(score.gainValueFromOneaddr(0x1D78A0F3190))

    # base_address = 0x00416D60  # 0x00691048  0x00694B00  0x00694B8C
    # offset_address = 0x00000190
    # while True:
    #     print(score.gainValue(base_address, offset_address))
    # print(round(float(int(score.gainValue(base_address, offset_address)) / 60), 2))
