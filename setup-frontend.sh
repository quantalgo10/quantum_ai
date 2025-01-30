#!/bin/bash

# Create directory structure
mkdir -p frontend/src/{components/{Charts,Trading},pages,services}
mkdir -p frontend/public

# Create pages
for page in Trading Portfolio Analysis Settings; do
  cat > "frontend/src/pages/${page}.js" << EOL
import React from 'react';
import { Grid, Paper, Typography } from '@mui/material';

function ${page}() {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">${page} Dashboard</Typography>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ${page};
EOL
done

# Create component index files
echo "export { default as CandlestickChart } from './CandlestickChart';
export { default as LineChart } from './LineChart';" > frontend/src/components/Charts/index.js

echo "export { default as MetricsCard } from './MetricsCard';
export { default as PositionsTable } from './PositionsTable';" > frontend/src/components/Trading/index.js

# Run these commands
chmod +x setup-frontend.sh
./setup-frontend.sh 