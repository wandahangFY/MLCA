#!/bin/bash
python flask_app.py &
python -m http.server 8888
