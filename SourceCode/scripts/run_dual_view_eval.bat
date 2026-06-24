@echo off
REM Run the dual-view (top + side) calorie evaluation with YOLOv13n.
REM Requires:  weights\yolov13n_ecustfd_best.pt + datasets\ECUSTFD\images\test
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

"%PY%" -u scripts\eval_calorie_dual_view.py ^
    --source datasets\ECUSTFD\images\test ^
    --weights weights\yolov13n_ecustfd_best.pt ^
    --density-json data\density_processed.json ^
    --output runs\dual_view_eval

if errorlevel 1 (
    echo.
    echo [ERROR] Dual-view evaluation failed. Check logs above.
    popd
    exit /b 1
)

popd
echo.
echo [OK] Dual-view evaluation finished. See runs\dual_view_eval\.
endlocal
exit /b 0