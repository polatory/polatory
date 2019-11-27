@echo off
setlocal

cd %~dp0
powershell -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File .run.ps1 %*
