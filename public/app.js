// app.js – handles tab navigation, data loading, and rendering charts/tables

document.addEventListener('DOMContentLoaded', () => {
  // Tab navigation
  const tabButtons = document.querySelectorAll('.tab-btn');
  const sections = document.querySelectorAll('.tab-section');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.getAttribute('data-target');
      sections.forEach(sec => {
        if (sec.id === target) sec.classList.remove('hidden');
        else sec.classList.add('hidden');
      });
    });
  });

  // Load Top‑20 metrics
  fetch('/api/top20')
    .then(r => r.json())
    .then(data => {
      // Populate DataTable
      $('#top20Table').DataTable({
        data: data,
        columns: Object.keys(data[0] || {}).map(k => ({title: k, data: k})),
        paging: false,
        searching: true,
        info: false,
        autoWidth: true,
      });

      // Simple bar chart – assume a column named "Metric" and "Score"
      const labels = data.map(d => d['Catalyst'] || d['Name'] || d['ID']);
      const scores = data.map(d => parseFloat(d['Score'] || d['Metric'] || d[Object.keys(d)[1]]));
      const ctx = document.getElementById('top20Chart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Score',
            data: scores,
            backgroundColor: 'rgba(100,149,237,0.7)',
            borderColor: 'rgba(100,149,237,1)',
            borderWidth: 1,
          }]
        },
        options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}}
      });
    })
    .catch(console.error);

  // Load GNN benchmark data
  fetch('/api/gnn_benchmark')
    .then(r => r.json())
    .then(data => {
      $('#gnnTable').DataTable({
        data: data,
        columns: Object.keys(data[0] || {}).map(k => ({title: k, data: k})),
        paging: false,
        searching: true,
        info: false,
        autoWidth: true,
      });

      // Bar chart for LOGO‑CV_R2 (assuming column name "LOGO‑CV_R2")
      const labels = data.map(d => d['Model'] || d['Name'] || d['ID']);
      const logoCv = data.map(d => parseFloat(d['LOGO‑CV_R2'] || d['LOGO-CV_R2'] || d['LOGO_CV_R2'] || 0));
      const spearman = data.map(d => parseFloat(d['Spearman'] || 0));
      const ctx = document.getElementById('gnnChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {label: 'LOGO‑CV R²', data: logoCv, backgroundColor: 'rgba(255,99,132,0.7)'},
            {label: 'Spearman', data: spearman, backgroundColor: 'rgba(54,162,235,0.7)'}
          ]
        },
        options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {position: 'top'}}}
      });
    })
    .catch(console.error);
});
