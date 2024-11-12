#Python3 version of hugo24's snippet
import winreg

REG_PATH = r"Control Panel\Mouse"

def set_reg(name, value):
    try:
        winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_PATH)
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, 
                                       winreg.KEY_WRITE)
        winreg.SetValueEx(registry_key, name, 0, winreg.REG_SZ, value)
        winreg.CloseKey(registry_key)
        return True
    except WindowsError:
        return False

def get_reg(name):
    try:
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0,
                                       winreg.KEY_READ)
        value, regtype = winreg.QueryValueEx(registry_key, name)
        winreg.CloseKey(registry_key)
        return value
    except WindowsError:
        return None

#Example MouseSensitivity
#Read value 
#print (get_reg('MouseSensitivity'))

#Set Value 1/20 (will just write the value to reg, the changed mouse val requires a win re-log to apply*)
#set_reg('MouseSensitivity', str(10))

#*For instant apply of SystemParameters like the mouse speed on-write, you can use win32gui/SPI
#http://docs.activestate.com/activepython/3.4/pywin32/win32gui__SystemParametersInfo_meth.html