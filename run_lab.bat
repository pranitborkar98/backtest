@echo off
REM =====================================================================
REM  run_lab.bat - One-click Quant Backtesting Lab pipeline
REM  Location: put this file inside D:\backtest (same folder as the .py files)
REM
REM  What it does, in order:
REM    1. Sanity check (python available, engine scripts present)
REM    2. run_backtest.py  -> loops every date x every engine, each engine
REM                            writes predictions_<YYYY-MM-DD>_<engine>.json
REM    3. engine_analyzer.py --aggregate -> leaderboard + analysis_detail.csv
REM
REM  Usage:
REM    run_lab.bat                     default: last 30 days, all engines
REM    run_lab.bat 60                  backtest last 60 days
REM    run_lab.bat 2024-09-01 2024-10-31   backtest an explicit date range
REM
REM  To exclude an engine (e.g. v31.py vs prediction_engine_v31.py share the
REM  'v31' label), edit the ENGINES list inside run_backtest.py.
REM =====================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "DAYS=30"
set "BT_ARGS="

if "%~1"=="" goto :defaults
REM --- single numeric arg = number of days -----------------------------
if not "%~2"=="" goto :range
set "DAYS=%~1"
set "BT_ARGS=--days %~1"
goto :defaults

:range
REM --- two args = start and end dates ----------------------------------
set "BT_ARGS=--start %~1 --end %~2"

:defaults
echo.
echo ============================================================
echo   QUANT BACKTESTING LAB
echo   Folder : %CD%
echo   Mode   : run_backtest.py %BT_ARGS%
echo ============================================================
echo.

REM -------------------- [1/3] sanity checks ----------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found on PATH. Install Python or fix PATH.
    goto :fail
)

for %%E in (run_backtest.py engine_analyzer.py all_markets_history.json) do (
    if not exist "%%E" (
        echo [ERROR] Required file missing: %%E
        echo         Copy your data and scripts into this folder first.
        goto :fail
    )
)

REM optional: clear old prediction files so results are fresh
set /p "CLEAN=Delete existing predictions_*.json before running? (y/N): "
if /i "!CLEAN!"=="y" del /q predictions_*.json 2>nul

echo.
echo ------------------------------------------------------------
echo [1/3] Generating dated prediction files (this takes ~10 min)...
echo ------------------------------------------------------------
python run_backtest.py %BT_ARGS%
if errorlevel 1 (
    echo.
    echo [WARN] run_backtest.py reported errors - see backtest_errors.log
)

echo.
echo ------------------------------------------------------------
echo [2/3] Counting generated prediction files...
echo ------------------------------------------------------------
set /a CNT=0
for %%F in (predictions_*.json) do set /a CNT+=1
echo Found !CNT! prediction file(s) in %CD%
if "!CNT!"=="0" (
    echo [ERROR] No predictions_*.json were produced. Check backtest_errors.log
    goto :fail
)

echo.
echo ------------------------------------------------------------
echo [3/3] Aggregating scores: leaderboard + analysis_detail.csv...
echo ------------------------------------------------------------
python engine_analyzer.py --aggregate
if errorlevel 1 (
    echo [ERROR] engine_analyzer.py failed.
    goto :fail
)

echo.
echo ============================================================
echo   DONE. Files to look at:
echo     - predictions_*.json      ^(!CNT! dated prediction files^)
echo     - analysis_detail.csv     ^(open in Excel, pivot by engine/category^)
echo     - backtest_errors.log     ^(only if some engine runs failed^)
echo ============================================================
pause
exit /b 0

:fail
echo.
echo *** PIPELINE STOPPED - fix the error above and re-run. ***
pause
exit /b 1
