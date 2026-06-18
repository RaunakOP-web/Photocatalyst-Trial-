const fs = require('fs');
const path = require('path');

module.exports = (req, res) => {
  const csvPath = path.join(__dirname, '..', 'data', 'results', 'publication_gnn_benchmark_table.csv');
  fs.readFile(csvPath, 'utf8', (err, data) => {
    if (err) {
      res.status(500).json({ error: 'Unable to read CSV' });
      return;
    }
    const lines = data.trim().split('\n');
    const headers = lines[0].split(',');
    const rows = lines.slice(1).map(line => {
      const values = line.split(',');
      const obj = {};
      headers.forEach((h, i) => {
        obj[h.trim()] = values[i] ? values[i].trim() : '';
      });
      return obj;
    });
    res.status(200).json(rows);
  });
};
