@echo off

REM Clean up Docker
docker-compose down
docker system prune -af

REM Create frontend directory structure
mkdir frontend\src\components\Charts
mkdir frontend\src\components\Trading
mkdir frontend\src\pages
mkdir frontend\src\services
mkdir frontend\public

REM Create React files
echo import React from 'react';> frontend\src\index.js
echo import ReactDOM from 'react-dom/client';>> frontend\src\index.js
echo import App from './App';>> frontend\src\index.js
echo import { ThemeProvider } from '@mui/material/styles';>> frontend\src\index.js
echo import theme from './theme';>> frontend\src\index.js
echo.>> frontend\src\index.js
echo const root = ReactDOM.createRoot(document.getElementById('root'));>> frontend\src\index.js
echo root.render(>> frontend\src\index.js
echo   ^<React.StrictMode^>>> frontend\src\index.js
echo     ^<ThemeProvider theme={theme}^>>> frontend\src\index.js
echo       ^<App /^>>> frontend\src\index.js
echo     ^</ThemeProvider^>>> frontend\src\index.js
echo   ^</React.StrictMode^>>> frontend\src\index.js
echo );>> frontend\src\index.js

REM Copy package.json
copy package.json frontend\

REM Install dependencies
cd frontend
call npm install

REM Go back and start Docker
cd ..
docker-compose up --build 