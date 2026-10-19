
@echo off
setlocal
cd /d "%~dp0"

where javac >nul 2>nul
if errorlevel 1 (
  echo Java JDK 17 or newer is required. Install a JDK and reopen this window.
  pause
  exit /b 1
)

if not exist out mkdir out

(for /r src %%f in (*.java) do @echo %%f) > .sources.list
echo Main.java>>.sources.list

javac --release 17 --add-modules jdk.httpserver -encoding UTF-8 -d out @.sources.list
set "BUILD_RESULT=%ERRORLEVEL%"
del .sources.list

if not "%BUILD_RESULT%"=="0" (
  echo Build failed. Review the compiler errors above.
  pause
  exit /b 1
)

java --add-modules jdk.httpserver -Dvectorvault.root="%CD%" -cp out Main
pause
