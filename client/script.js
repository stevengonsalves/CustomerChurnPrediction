const resultDiv = document.getElementById('report');
const barGraphImg = document.getElementById('graph'); // Bar graph
const lineGraphImg = document.getElementById('lineGraph'); // Line graph

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please upload a CSV file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
    });

    if (response.ok) {
        const result = await response.json();

        // Clear previous results
        resultDiv.innerHTML = '';

        // Render predictions
        result.results.forEach(item => {
            const prediction = document.createElement('h1');
            prediction.style.fontWeight = 'bold';
            prediction.style.color = item.prediction === 'Churn' ? 'red' : 'green';
            prediction.innerText = `Churn Prediction: ${item.prediction}`;

            const reason = document.createElement('p');
            reason.innerText = item.reason;
            reason.style.fontSize = '16px';

            resultDiv.appendChild(prediction);
            resultDiv.appendChild(reason);
        });

        // Display bar graph
        barGraphImg.src = result.bar_graph;
        barGraphImg.style.display = 'block';

        // Display line graph
        lineGraphImg.src = result.line_graph;
        lineGraphImg.style.display = 'block';
    } else {
        const error = await response.json();
        alert(`Error: ${error.error}`);
    }
});
