import winreg

REG_PATH = r"Control Panel\Mouse"
name="MouseSensitivity"
registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0,
                                winreg.KEY_READ)
value, regtype = winreg.QueryValueEx(registry_key, name)
winreg.CloseKey(registry_key)
print(value)