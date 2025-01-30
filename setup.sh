#!/bin/bash

# Clean up existing containers and images
docker-compose down
docker system prune -af

# Create directory structure
mkdir -p frontend/src/{components/{Charts,Trading},pages,services} frontend/public

# Create necessary files
cat > frontend/public/index.html << 'EOL'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Quantum Trading Platform" />
    <title>Quantum Trading</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap" />
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOL

# Copy package.json
cp package.json frontend/

# Install dependencies
cd frontend
npm install

# Go back to root
cd ..

# Build and start containers
docker-compose up --build -d 