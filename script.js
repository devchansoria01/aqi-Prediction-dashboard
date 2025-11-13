function showPage(pageId) {
    // Hide all pages
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => {
        page.classList.remove('active');
    });

    // Remove active class from all nav links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.classList.remove('active');
    });

    // Show selected page
    const selectedPage = document.getElementById(pageId);
    if (selectedPage) {
        selectedPage.classList.add('active');
    }

    // Add active class to clicked nav link
    const activeLink = document.querySelector(`[data-page="${pageId}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }

    // Scroll to top
    window.scrollTo(0, 0);
}

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    showPage('transformer');
});


// LSTM Model - 72 Hour Predictions
const lstmPredictions = [
  { time: "2025-10-24 00:00", aqi: 260.190002 },
  { time: "2025-10-24 01:00", aqi: 262.700012 },
  { time: "2025-10-24 02:00", aqi: 253.190002 },
  { time: "2025-10-24 03:00", aqi: 258.589996 },
  { time: "2025-10-24 04:00", aqi: 249.229996 },
  { time: "2025-10-24 05:00", aqi: 250.729996 },
  { time: "2025-10-24 06:00", aqi: 253.289993 },
  { time: "2025-10-24 07:00", aqi: 250.770004 },
  { time: "2025-10-24 08:00", aqi: 250.630005 },
  { time: "2025-10-24 09:00", aqi: 247.020004 },
  { time: "2025-10-24 10:00", aqi: 251.990005 },
  { time: "2025-10-24 11:00", aqi: 244.630005 },
  { time: "2025-10-24 12:00", aqi: 242.740005 },
  { time: "2025-10-24 13:00", aqi: 251.380005 },
  { time: "2025-10-24 14:00", aqi: 246.520004 },
  { time: "2025-10-24 15:00", aqi: 241.429993 },
  { time: "2025-10-24 16:00", aqi: 249.669998 },
  { time: "2025-10-24 17:00", aqi: 242.399994 },
  { time: "2025-10-24 18:00", aqi: 250.020004 },
  { time: "2025-10-24 19:00", aqi: 252.610001 },
  { time: "2025-10-24 20:00", aqi: 234.910004 },
  { time: "2025-10-24 21:00", aqi: 239.820007 },
  { time: "2025-10-24 22:00", aqi: 245.419998 },
  { time: "2025-10-24 23:00", aqi: 232.110001 },
  { time: "2025-10-25 00:00", aqi: 243.940002 },
  { time: "2025-10-25 01:00", aqi: 236.610001 },
  { time: "2025-10-25 02:00", aqi: 236.789993 },
  { time: "2025-10-25 03:00", aqi: 230.389999 },
  { time: "2025-10-25 04:00", aqi: 234.360001 },
  { time: "2025-10-25 05:00", aqi: 231.580002 },
  { time: "2025-10-25 06:00", aqi: 232.580002 },
  { time: "2025-10-25 07:00", aqi: 234.520004 },
  { time: "2025-10-25 08:00", aqi: 238.609995 },
  { time: "2025-10-25 09:00", aqi: 232.190002 },
  { time: "2025-10-25 10:00", aqi: 231.399994 },
  { time: "2025-10-25 11:00", aqi: 219.740005 },
  { time: "2025-10-25 12:00", aqi: 228.849994 },
  { time: "2025-10-25 13:00", aqi: 233.130005 },
  { time: "2025-10-25 14:00", aqi: 220.960007 },
  { time: "2025-10-25 15:00", aqi: 223.610001 },
  { time: "2025-10-25 16:00", aqi: 239.470001 },
  { time: "2025-10-25 17:00", aqi: 219.059998 },
  { time: "2025-10-25 18:00", aqi: 220.610001 },
  { time: "2025-10-25 19:00", aqi: 221.990005 },
  { time: "2025-10-25 20:00", aqi: 226.609998 },
  { time: "2025-10-25 21:00", aqi: 226.059998 },
  { time: "2025-10-25 22:00", aqi: 218.220001 },
  { time: "2025-10-25 23:00", aqi: 227.279999 },
  { time: "2025-10-26 00:00", aqi: 231.960007 },
  { time: "2025-10-26 01:00", aqi: 222.880005 },
  { time: "2025-10-26 02:00", aqi: 224.710007 },
  { time: "2025-10-26 03:00", aqi: 221.259995 },
  { time: "2025-10-26 04:00", aqi: 235.240005 },
  { time: "2025-10-26 05:00", aqi: 221.289993 },
  { time: "2025-10-26 06:00", aqi: 220.619995 },
  { time: "2025-10-26 07:00", aqi: 212.929993 },
  { time: "2025-10-26 08:00", aqi: 207.880005 },
  { time: "2025-10-26 09:00", aqi: 216.479996 },
  { time: "2025-10-26 10:00", aqi: 218.600006 },
  { time: "2025-10-26 11:00", aqi: 212.139999 },
  { time: "2025-10-26 12:00", aqi: 212.500002 },
  { time: "2025-10-26 13:00", aqi: 217.089996 },
  { time: "2025-10-26 14:00", aqi: 216.330002 },
  { time: "2025-10-26 15:00", aqi: 216.250000 },
  { time: "2025-10-26 16:00", aqi: 220.490005 },
  { time: "2025-10-26 17:00", aqi: 217.570007 },
  { time: "2025-10-26 18:00", aqi: 211.729996 },
  { time: "2025-10-26 19:00", aqi: 220.559998 },
  { time: "2025-10-26 20:00", aqi: 203.330002 },
  { time: "2025-10-26 21:00", aqi: 207.700006 },
  { time: "2025-10-26 22:00", aqi: 210.889999 }
];

// Function to populate the LSTM forecast table
function populateLSTMTable() {
  const tableBody = document.querySelector('#lstm-forecast tbody');
  if (!tableBody) return;
  
  tableBody.innerHTML = '';
  
  // Display first 24 hours (72 hours = 3 days, showing 24 per page)
  lstmPredictions.slice(0, 72).forEach(prediction => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${prediction.time}</td>
      <td class="aqi-value">${prediction.aqi.toFixed(2)}</td>
    `;
    tableBody.appendChild(row);
  });
}

// Call this function when the page loads or when LSTM tab is clicked
document.addEventListener('DOMContentLoaded', populateLSTMTable);


// Random Forest Model - 72 Hour Predictions
const randomForestPredictions = [
  { time: "2025-10-24 00:00", aqi: 214.49 },
  { time: "2025-10-24 01:00", aqi: 212.65 },
  { time: "2025-10-24 02:00", aqi: 211.51 },
  { time: "2025-10-24 03:00", aqi: 209.47 },
  { time: "2025-10-24 04:00", aqi: 208.21 },
  { time: "2025-10-24 05:00", aqi: 206.23 },
  { time: "2025-10-24 06:00", aqi: 205.48 },
  { time: "2025-10-24 07:00", aqi: 205.40 },
  { time: "2025-10-24 08:00", aqi: 204.16 },
  { time: "2025-10-24 09:00", aqi: 204.68 },
  { time: "2025-10-24 10:00", aqi: 204.35 },
  { time: "2025-10-24 11:00", aqi: 203.55 },
  { time: "2025-10-24 12:00", aqi: 203.74 },
  { time: "2025-10-24 13:00", aqi: 203.16 },
  { time: "2025-10-24 14:00", aqi: 203.46 },
  { time: "2025-10-24 15:00", aqi: 201.95 },
  { time: "2025-10-24 16:00", aqi: 202.35 },
  { time: "2025-10-24 17:00", aqi: 202.53 },
  { time: "2025-10-24 18:00", aqi: 203.27 },
  { time: "2025-10-24 19:00", aqi: 202.15 },
  { time: "2025-10-24 20:00", aqi: 202.40 },
  { time: "2025-10-24 21:00", aqi: 203.08 },
  { time: "2025-10-24 22:00", aqi: 203.35 },
  { time: "2025-10-24 23:00", aqi: 204.06 },
  { time: "2025-10-25 00:00", aqi: 203.76 },
  { time: "2025-10-25 01:00", aqi: 204.49 },
  { time: "2025-10-25 02:00", aqi: 204.23 },
  { time: "2025-10-25 03:00", aqi: 203.90 },
  { time: "2025-10-25 04:00", aqi: 203.97 },
  { time: "2025-10-25 05:00", aqi: 202.96 },
  { time: "2025-10-25 06:00", aqi: 203.28 },
  { time: "2025-10-25 07:00", aqi: 203.01 },
  { time: "2025-10-25 08:00", aqi: 203.40 },
  { time: "2025-10-25 09:00", aqi: 203.03 },
  { time: "2025-10-25 10:00", aqi: 203.00 },
  { time: "2025-10-25 11:00", aqi: 201.85 },
  { time: "2025-10-25 12:00", aqi: 201.55 },
  { time: "2025-10-25 13:00", aqi: 202.02 },
  { time: "2025-10-25 14:00", aqi: 201.04 },
  { time: "2025-10-25 15:00", aqi: 199.72 },
  { time: "2025-10-25 16:00", aqi: 199.09 },
  { time: "2025-10-25 17:00", aqi: 196.84 },
  { time: "2025-10-25 18:00", aqi: 194.24 },
  { time: "2025-10-25 19:00", aqi: 192.08 },
  { time: "2025-10-25 20:00", aqi: 190.14 },
  { time: "2025-10-25 21:00", aqi: 187.41 },
  { time: "2025-10-25 22:00", aqi: 185.57 },
  { time: "2025-10-25 23:00", aqi: 183.35 },
  { time: "2025-10-26 00:00", aqi: 181.03 },
  { time: "2025-10-26 01:00", aqi: 179.36 },
  { time: "2025-10-26 02:00", aqi: 178.02 },
  { time: "2025-10-26 03:00", aqi: 176.99 }
];

// Function to populate the Random Forest forecast table
function populateRandomForestTable() {
  const tableBody = document.querySelector('#random-forest-forecast tbody');
  if (!tableBody) return;
  
  tableBody.innerHTML = '';
  
  // Display first 24 hours (modify slice if you want different range)
  randomForestPredictions.slice(0, 59).forEach(prediction => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${prediction.time}</td>
      <td class="aqi-value">${prediction.aqi.toFixed(2)}</td>
    `;
    tableBody.appendChild(row);
  });
}

// Call this function when the page loads or when Random Forest tab is clicked
document.addEventListener('DOMContentLoaded', populateRandomForestTable);










// Transformer Model - 72 Hour Predictions
const transformerPredictions = [
  { time: "2025-10-24 00:00", aqi: 276.929993 },
  { time: "2025-10-24 01:00", aqi: 282.559998 },
  { time: "2025-10-24 02:00", aqi: 281.420013 },
  { time: "2025-10-24 03:00", aqi: 283.429993 },
  { time: "2025-10-24 04:00", aqi: 283.519989 },
  { time: "2025-10-24 05:00", aqi: 284.209991 },
  { time: "2025-10-24 06:00", aqi: 284.779991 },
  { time: "2025-10-24 07:00", aqi: 288.700012 },
  { time: "2025-10-24 08:00", aqi: 291.410004 },
  { time: "2025-10-24 09:00", aqi: 291.480011 },
  { time: "2025-10-24 10:00", aqi: 296.170013 },
  { time: "2025-10-24 11:00", aqi: 296.529999 },
  { time: "2025-10-24 12:00", aqi: 291.059998 },
  { time: "2025-10-24 13:00", aqi: 294.619995 },
  { time: "2025-10-24 14:00", aqi: 292.910004 },
  { time: "2025-10-24 15:00", aqi: 291.529995 },
  { time: "2025-10-24 16:00", aqi: 291.690002 },
  { time: "2025-10-24 17:00", aqi: 294.940002 },
  { time: "2025-10-24 18:00", aqi: 295.970001 },
  { time: "2025-10-24 19:00", aqi: 299.470001 },
  { time: "2025-10-24 20:00", aqi: 298.489990 },
  { time: "2025-10-24 21:00", aqi: 302.780012 },
  { time: "2025-10-24 22:00", aqi: 301.040009 },
  { time: "2025-10-24 23:00", aqi: 305.510010 },
  { time: "2025-10-25 00:00", aqi: 302.899994 },
  { time: "2025-10-25 01:00", aqi: 304.160004 },
  { time: "2025-10-25 02:00", aqi: 304.529999 },
  { time: "2025-10-25 03:00", aqi: 303.020007 },
  { time: "2025-10-25 04:00", aqi: 303.950012 },
  { time: "2025-10-25 05:00", aqi: 307.829987 },
  { time: "2025-10-25 06:00", aqi: 302.839996 },
  { time: "2025-10-25 07:00", aqi: 306.880005 },
  { time: "2025-10-25 08:00", aqi: 310.059998 },
  { time: "2025-10-25 09:00", aqi: 310.739990 },
  { time: "2025-10-25 10:00", aqi: 310.399994 },
  { time: "2025-10-25 11:00", aqi: 313.299988 },
  { time: "2025-10-25 12:00", aqi: 311.230011 },
  { time: "2025-10-25 13:00", aqi: 312.170013 },
  { time: "2025-10-25 14:00", aqi: 314.549988 },
  { time: "2025-10-25 15:00", aqi: 311.130005 },
  { time: "2025-10-25 16:00", aqi: 314.440002 },
  { time: "2025-10-25 17:00", aqi: 315.470001 },
  { time: "2025-10-25 18:00", aqi: 315.699002 },
  { time: "2025-10-25 19:00", aqi: 315.680002 },
  { time: "2025-10-25 20:00", aqi: 318.049999 },
  { time: "2025-10-25 21:00", aqi: 316.540009 },
  { time: "2025-10-25 22:00", aqi: 315.519989 },
  { time: "2025-10-25 23:00", aqi: 316.079987 },
  { time: "2025-10-26 00:00", aqi: 317.179993 },
  { time: "2025-10-26 01:00", aqi: 320.720001 },
  { time: "2025-10-26 02:00", aqi: 319.690002 },
  { time: "2025-10-26 03:00", aqi: 319.279999 },
  { time: "2025-10-26 04:00", aqi: 321.929009 },
  { time: "2025-10-26 05:00", aqi: 319.359985 },
  { time: "2025-10-26 06:00", aqi: 322.180006 },
  { time: "2025-10-26 07:00", aqi: 319.820313 },
  { time: "2025-10-26 08:00", aqi: 320.290009 },
  { time: "2025-10-26 09:00", aqi: 318.559996 },
  { time: "2025-10-26 10:00", aqi: 318.760010 },
  { time: "2025-10-26 11:00", aqi: 320.049988 },
  { time: "2025-10-26 12:00", aqi: 322.640002 },
  { time: "2025-10-26 13:00", aqi: 319.529999 },
  { time: "2025-10-26 14:00", aqi: 318.440002 },
  { time: "2025-10-26 15:00", aqi: 320.399994 },
  { time: "2025-10-26 16:00", aqi: 320.660004 },
  { time: "2025-10-26 17:00", aqi: 317.359985 },
  { time: "2025-10-26 18:00", aqi: 316.589996 },
  { time: "2025-10-26 19:00", aqi: 316.350006 },
  { time: "2025-10-26 20:00", aqi: 314.279999 },
  { time: "2025-10-26 21:00", aqi: 313.959991 },
  { time: "2025-10-26 22:00", aqi: 316.739990 },
  { time: "2025-10-26 23:00", aqi: 314.359985 }
];

// Function to populate the Transformer forecast table
function populateTransformerTable() {
  const tableBody = document.querySelector('#transformer-forecast tbody');
  if (!tableBody) return;
  
  tableBody.innerHTML = '';
  
  // Display first 24 hours (modify slice if you want different range)
  transformerPredictions.slice(0, 72).forEach(prediction => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${prediction.time}</td>
      <td class="aqi-value">${prediction.aqi.toFixed(2)}</td>
    `;
    tableBody.appendChild(row);
  });
}

// Call this function when the page loads or when Transformer tab is clicked
document.addEventListener('DOMContentLoaded', populateTransformerTable);










// Transformer4Cities Model - 72 Hour Predictions (Jharia Location)
const transformer4CitiesPredictions = [
  { time: "2025-10-24 00:00", aqi: 264.529999 },
  { time: "2025-10-24 01:00", aqi: 263.410004 },
  { time: "2025-10-24 02:00", aqi: 266.589996 },
  { time: "2025-10-24 03:00", aqi: 268.019989 },
  { time: "2025-10-24 04:00", aqi: 265.500000 },
  { time: "2025-10-24 05:00", aqi: 271.010010 },
  { time: "2025-10-24 06:00", aqi: 270.410004 },
  { time: "2025-10-24 07:00", aqi: 273.709991 },
  { time: "2025-10-24 08:00", aqi: 273.899994 },
  { time: "2025-10-24 09:00", aqi: 273.660004 },
  { time: "2025-10-24 10:00", aqi: 275.589996 },
  { time: "2025-10-24 11:00", aqi: 274.039999 },
  { time: "2025-10-24 12:00", aqi: 273.720001 },
  { time: "2025-10-24 13:00", aqi: 274.869995 },
  { time: "2025-10-24 14:00", aqi: 279.829987 },
  { time: "2025-10-24 15:00", aqi: 276.779999 },
  { time: "2025-10-24 16:00", aqi: 275.160004 },
  { time: "2025-10-24 17:00", aqi: 275.910004 },
  { time: "2025-10-24 18:00", aqi: 273.190002 },
  { time: "2025-10-24 19:00", aqi: 276.359995 },
  { time: "2025-10-24 20:00", aqi: 278.489990 },
  { time: "2025-10-24 21:00", aqi: 275.660004 },
  { time: "2025-10-24 22:00", aqi: 275.789988 },
  { time: "2025-10-24 23:00", aqi: 276.980011 },
  { time: "2025-10-25 00:00", aqi: 274.200012 },
  { time: "2025-10-25 01:00", aqi: 272.859985 },
  { time: "2025-10-25 02:00", aqi: 276.279999 },
  { time: "2025-10-25 03:00", aqi: 272.440002 },
  { time: "2025-10-25 04:00", aqi: 262.929993 },
  { time: "2025-10-25 05:00", aqi: 268.489990 },
  { time: "2025-10-25 06:00", aqi: 267.869995 },
  { time: "2025-10-25 07:00", aqi: 267.769989 },
  { time: "2025-10-25 08:00", aqi: 264.989990 },
  { time: "2025-10-25 09:00", aqi: 260.429993 },
  { time: "2025-10-25 10:00", aqi: 258.880005 },
  { time: "2025-10-25 11:00", aqi: 256.040009 },
  { time: "2025-10-25 12:00", aqi: 257.059998 },
  { time: "2025-10-25 13:00", aqi: 257.130002 },
  { time: "2025-10-25 14:00", aqi: 254.980005 },
  { time: "2025-10-25 15:00", aqi: 252.839996 },
  { time: "2025-10-25 16:00", aqi: 252.830002 },
  { time: "2025-10-25 17:00", aqi: 247.640007 },
  { time: "2025-10-25 18:00", aqi: 246.919998 },
  { time: "2025-10-25 19:00", aqi: 246.639999 },
  { time: "2025-10-25 20:00", aqi: 245.929993 },
  { time: "2025-10-25 21:00", aqi: 245.929993 },
  { time: "2025-10-25 22:00", aqi: 247.939996 },
  { time: "2025-10-25 23:00", aqi: 239.720001 }
];

// Function to populate the Transformer4Cities forecast table
function populateTransformer4CitiesTable() {
  const tableBody = document.querySelector('#transformer4cities-forecast tbody');
  if (!tableBody) return;
  
  tableBody.innerHTML = '';
  
  transformer4CitiesPredictions.slice(0, 24).forEach(prediction => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${prediction.time}</td>
      <td class="aqi-value">${prediction.aqi.toFixed(2)}</td>
    `;
    tableBody.appendChild(row);
  });
}

document.addEventListener('DOMContentLoaded', populateTransformer4CitiesTable);











const comparisonData = [
      { time: "2025-10-24 00:00", randomForest: 214.49, lstm: 260.190002, transformer4Cities: 264.529999 },
      { time: "2025-10-24 01:00", randomForest: 212.65, lstm: 262.700012, transformer4Cities: 263.410004 },
      { time: "2025-10-24 02:00", randomForest: 211.51, lstm: 253.190002, transformer4Cities: 266.589996 },
      { time: "2025-10-24 03:00", randomForest: 209.47, lstm: 258.589996, transformer4Cities: 268.019989 },
      { time: "2025-10-24 04:00", randomForest: 208.21, lstm: 249.229996, transformer4Cities: 265.500000 },
      { time: "2025-10-24 05:00", randomForest: 206.23, lstm: 250.729996, transformer4Cities: 271.010010 },
      { time: "2025-10-24 06:00", randomForest: 205.48, lstm: 253.289993, transformer4Cities: 270.410004 },
      { time: "2025-10-24 07:00", randomForest: 205.40, lstm: 250.770004, transformer4Cities: 273.709991 },
      { time: "2025-10-24 08:00", randomForest: 204.16, lstm: 250.630005, transformer4Cities: 273.899994 },
      { time: "2025-10-24 09:00", randomForest: 204.68, lstm: 247.020004, transformer4Cities: 273.660004 },
      { time: "2025-10-24 10:00", randomForest: 204.35, lstm: 251.990005, transformer4Cities: 275.589996 },
      { time: "2025-10-24 11:00", randomForest: 203.55, lstm: 244.630005, transformer4Cities: 274.039999 },
      { time: "2025-10-24 12:00", randomForest: 203.74, lstm: 242.740005, transformer4Cities: 273.720001 },
      { time: "2025-10-24 13:00", randomForest: 203.16, lstm: 251.380005, transformer4Cities: 274.869995 },
      { time: "2025-10-24 14:00", randomForest: 203.46, lstm: 246.520004, transformer4Cities: 279.829987 },
      { time: "2025-10-24 15:00", randomForest: 201.95, lstm: 241.429993, transformer4Cities: 276.779999 },
      { time: "2025-10-24 16:00", randomForest: 202.35, lstm: 249.669998, transformer4Cities: 275.160004 },
      { time: "2025-10-24 17:00", randomForest: 202.53, lstm: 242.399994, transformer4Cities: 275.910004 },
      { time: "2025-10-24 18:00", randomForest: 203.27, lstm: 250.020004, transformer4Cities: 273.190002 },
      { time: "2025-10-24 19:00", randomForest: 202.15, lstm: 252.610001, transformer4Cities: 276.359995 },
      { time: "2025-10-24 20:00", randomForest: 202.40, lstm: 234.910004, transformer4Cities: 278.489990 },
      { time: "2025-10-24 21:00", randomForest: 203.08, lstm: 239.820007, transformer4Cities: 275.660004 },
      { time: "2025-10-24 22:00", randomForest: 203.35, lstm: 245.419998, transformer4Cities: 275.789988 },
      { time: "2025-10-24 23:00", randomForest: 204.06, lstm: 232.110001, transformer4Cities: 276.980011 }
    ];



    function populateComparisonTable() {
      const tableBody = document.querySelector('#comparison-forecast tbody');
      if (!tableBody) return;
      tableBody.innerHTML = '';
      comparisonData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${row.time}</td>
          <td class="aqi-value">${row.randomForest.toFixed(2)}</td>
          <td class="aqi-value">${row.lstm.toFixed(2)}</td>
          <td class="aqi-value">${row.transformer4Cities.toFixed(2)}</td>
        `;
        tableBody.appendChild(tr);
      });
    }

    document.addEventListener('DOMContentLoaded', populateComparisonTable);
