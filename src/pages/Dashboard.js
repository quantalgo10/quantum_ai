import React, { useState, useEffect } from 'react';
import { Grid, Paper, Typography } from '@mui/material';
import { LineChart, CandlestickChart } from '../components/Charts';
import { MetricsCard, PositionsTable } from '../components/Trading';

function Dashboard() {
  const [marketData, setMarketData] = useState(null);
  const [portfolio, setPortfolio] = useState(null);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [marketResponse, portfolioResponse] = await Promise.all([
        fetch('/api/market-data'),
        fetch('/api/portfolio')
      ]);
      
      const marketData = await marketResponse.json();
      const portfolio = await portfolioResponse.json();
      
      setMarketData(marketData);
      setPortfolio(portfolio);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Market Overview
          </Typography>
          <CandlestickChart data={marketData} />
        </Paper>
      </Grid>
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Portfolio Summary
          </Typography>
          <MetricsCard data={portfolio} />
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Open Positions
          </Typography>
          <PositionsTable data={portfolio?.positions} />
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Dashboard; 