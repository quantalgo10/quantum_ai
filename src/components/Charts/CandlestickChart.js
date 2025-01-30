import React from 'react';
import { createChart } from 'lightweight-charts';
import { Box } from '@mui/material';

function CandlestickChart({ data }) {
  const chartContainerRef = React.useRef();

  React.useEffect(() => {
    if (!data || !chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: '#1E1E1E' },
        textColor: '#DDD',
      },
      grid: {
        vertLines: { color: '#2B2B2B' },
        horzLines: { color: '#2B2B2B' },
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#00FF9D',
      downColor: '#FF4B4B',
      borderVisible: false,
      wickUpColor: '#00FF9D',
      wickDownColor: '#FF4B4B',
    });

    candlestickSeries.setData(data);

    const handleResize = () => {
      chart.applyOptions({
        width: chartContainerRef.current.clientWidth,
      });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data]);

  return <Box ref={chartContainerRef} sx={{ width: '100%', height: 400 }} />;
}

export default CandlestickChart; 