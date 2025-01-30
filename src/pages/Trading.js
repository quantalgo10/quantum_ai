import React from 'react';
import { Grid, Paper, Typography } from '@mui/material';

function Trading() {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Trading Dashboard</Typography>
          {/* Add trading components here */}
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Trading; 