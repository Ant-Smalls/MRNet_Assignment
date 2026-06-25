#!/bin/bash
# run_dashboard_update.sh

echo "Checking if explainability background jobs are still running..."
if pgrep -f "python code/src/explainability.py" > /dev/null; then
    echo "The explainability scripts are STILL RUNNING."
    echo "Please wait for them to finish before regenerating the dashboard."
    exit 1
fi

echo "Explainability jobs are done! Regenerating the Triage Dashboard..."
source .venv/bin/activate
python code/src/generate_triage_dashboard.py
echo "Dashboard regenerated successfully at job_outputs/triage_dashboard.html"
