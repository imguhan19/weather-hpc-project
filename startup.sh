#!/bin/bash
# Activate Azure virtual environment if present
if [ -d "/home/site/wwwroot/antenv" ]; then
    source /home/site/wwwroot/antenv/bin/activate
fi

# Fallback to python3 / python
PORT="${PORT:-8080}"
python3 -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0 || python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0
