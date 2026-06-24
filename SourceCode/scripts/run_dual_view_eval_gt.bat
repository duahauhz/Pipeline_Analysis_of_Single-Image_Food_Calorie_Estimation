@echo off
REM Re-run the GT-based dual-view evaluation on ECUSTFD.
REM No GPU / YOLOv13n required -- uses ground-truth bbox labels.
REM
REM Usage: just double-click this file, or run from any terminal:
REM   scripts\run_dual_view_eval_gt.bat
REM
REM The script auto-detects the project root as the parent of this folder.
setlocal

set "PROJECT_ROOT=%~dp0..\"
pushd "%PROJECT_ROOT%"

set "PY="
where py >nul 2>&1
if not errorlevel 1 (
    set "PY=py"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PY=python"
    ) else (
        echo [ERROR] Neither 'py' nor 'python' found on PATH.
        popd
        exit /b 1
    )
)

"%PY%" -u scripts\eval_calorie_dual_view_gt.py ^
    --labels-root datasets\ECUSTFD\labels ^
    --density-json data\density_processed.json ^
    --output runs\dual_view_eval_gt

if errorlevel 1 (
    echo.
    echo [ERROR] Dual-view (GT) evaluation failed.
    popd
    exit /b 1
)

popd
echo.
echo [OK] Dual-view (GT) evaluation finished. See runs\dual_view_eval_gt\.
endlocal
exit /b 0