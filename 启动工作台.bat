@echo off
rem Chinese-named entry point -> real script is the pure-ASCII start_workbench.bat
rem (bat content must stay ASCII: cmd parses the file with the GBK code page and
rem  UTF-8 Chinese bytes would split the command lines)
call "%~dp0start_workbench.bat" %*
