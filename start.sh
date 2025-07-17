#!/bin/bash

# Start the Flask app with Gunicorn
exec gunicorn app:app \
  --bind 0.0.0.0:10000 \
  --workers 4
