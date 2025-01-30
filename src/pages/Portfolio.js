import React from 'react';
import { Grid, Paper, Typography } from '@mui/material';

function Portfolio() {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Portfolio Overview</Typography>
          {/* Add portfolio components here */}
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Portfolio; 