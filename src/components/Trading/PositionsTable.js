import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';

function PositionsTable({ data }) {
  if (!data || data.length === 0) {
    return <div>No positions found</div>;
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Entry Price</TableCell>
            <TableCell align="right">Current Price</TableCell>
            <TableCell align="right">P&L</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((position) => (
            <TableRow key={position.symbol}>
              <TableCell>{position.symbol}</TableCell>
              <TableCell align="right">{position.quantity}</TableCell>
              <TableCell align="right">₹{position.entryPrice}</TableCell>
              <TableCell align="right">₹{position.currentPrice}</TableCell>
              <TableCell 
                align="right"
                sx={{ 
                  color: position.pnl >= 0 ? 'success.main' : 'error.main'
                }}
              >
                ₹{position.pnl}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default PositionsTable; 