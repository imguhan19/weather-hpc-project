#!/bin/bash
# Move to project root directory
cd /home/site/wwwroot

# Activate virtual environment if present
if [ -d "/home/site/wwwroot/antenv" ]; then
    source /home/site/wwwroot/antenv/bin/activate
fi

# Run Streamlit on Port 8000 (Azure App Service Default Port)
python3 -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
