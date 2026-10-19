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
if not exist out-test mkdir out-test
(for /r src %%f in (*.java) do @echo %%f) > .sources.list
echo Main.java>>.sources.list
javac --release 17 --add-modules jdk.httpserver -encoding UTF-8 -d out @.sources.list
del .sources.list
if errorlevel 1 exit /b 1
javac --release 17 --add-modules jdk.httpserver -encoding UTF-8 -cp out -d out-test tests\VectorVaultSelfTest.java
if errorlevel 1 exit /b 1
set "WORK=%TEMP%\vectorvault-test-%RANDOM%"
mkdir "%WORK%"
pushd "%WORK%"
java --add-modules jdk.httpserver -cp "%~dp0out;%~dp0out-test" VectorVaultSelfTest
set "RESULT=%ERRORLEVEL%"
popd
rmdir /s /q "%WORK%"
if not "%RESULT%"=="0" exit /b %RESULT%
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0tests\integration_smoke.py"
  if errorlevel 1 exit /b 1
) else (
  echo Python not found; skipped optional HTTP/Ollama-protocol integration smoke test.
)
echo All available tests passed.
