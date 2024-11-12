import winreg

# REG_PATH = r"Control Panel\Mouse"
# name="MouseSensitivity"
# registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0,
#                                 winreg.KEY_READ)
# value, regtype = winreg.QueryValueEx(registry_key, name)
# winreg.CloseKey(registry_key)

REG_PATH=r'Software\Classes\CLSID\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}'
winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_PATH)

REG_PATH=r'Software\Classes\CLSID\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}\InprocServer32'
winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_PATH)
registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_WRITE)

name='' # 其實就是(預設值),也是回答影片中的問題
value=''
winreg.SetValueEx(registry_key, name, 0, winreg.REG_SZ, value)
winreg.CloseKey(registry_key)