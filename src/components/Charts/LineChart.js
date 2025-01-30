import React from 'react';
import { Line } from 'recharts';
import { Box } from '@mui/material';

function LineChart({ data }) {
  if (!data) return null;

  return (
    <Box sx={{ width: '100%', height: 300 }}>
      <Line
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      />
    </Box>
  );
}

export default LineChart; 