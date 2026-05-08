@echo off

set DIR=%~dp0

set PATH=%DIR%system\python;%DIR%system\python\Scripts;%PATH%
set MODEL_CHECKSUM_DISABLED=1
set U2NET_HOME=%DIR%webui\models

cd %~dp0webui

"%DIR%system\python\python.exe" rembgui.py

:done
pause