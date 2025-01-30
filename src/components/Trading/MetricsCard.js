import React from 'react';
import { Box, Typography, Grid } from '@mui/material';

function MetricsCard({ data }) {
  if (!data) return null;

  return (
    <Grid container spacing={2}>
      <Grid item xs={6}>
        <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="subtitle2" color="text.secondary">
            Total Value
          </Typography>
          <Typography variant="h6">
            ₹{Number(data.net).toLocaleString()}
          </Typography>
        </Box>
      </Grid>
      <Grid item xs={6}>
        <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="subtitle2" color="text.secondary">
            Available Margin
          </Typography>
          <Typography variant="h6">
            ₹{Number(data.available_margin).toLocaleString()}
          </Typography>
        </Box>
      </Grid>
    </Grid>
  );
}

export default MetricsCard; 