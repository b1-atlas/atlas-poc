const express = require('express');
const app = express();
const path = require('path');
const args = process.argv;

app.all(/\/api\/.*?/i, (req, res, next) => {
  res.header('Access-Control-Allow-Methods', 'GET, PUT, POST, DELETE, OPTIONS');
  res.header(
    'Access-Control-Allow-Headers',
    'X-Requested-With, Content-type, Accept, Content-Length, Authorization, X-Access-Token, X-Key, id-token, host',
  );
  res.header('Access-Control-Allow-Origin', 'http://app.eric.com'); //*
  res.header('Access-Control-Allow-Credentials', 'true');
  res.header('Content-Type', 'application/json;charset=utf-8');
  console.log('Mock API:', req.method, req.path);
  // console.log('Cookie:', req.headers['cookie']);
  if (req.method === 'OPTIONS') {
    res.status(200).end();
  } else {
    next();
  }
});

app.use('/', express.static(path.join(__dirname, 'build')));

app.get('*', (req, res) => {
  console.log(req.method, req.path, req.headers['user-agent']);
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

const port = args[2] || '3000';
const host = args[3] || '0.0.0.0';
app.listen(port, host, () => {
  console.log(`Starting up http-server, serving ${host}:${port}`);
});
