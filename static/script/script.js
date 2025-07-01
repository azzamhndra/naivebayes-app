document.getElementById('file').addEventListener('change', function(event) {
    var fileName = event.target.files[0].name;
    var fileNameDisplay = document.querySelector('.file-name');
    fileNameDisplay.textContent = fileName;
});

document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const overlay = document.getElementById('overlay');
    const loadingDiv = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');

    form.addEventListener('submit', function () {
        overlay.style.display = 'block';
        loadingDiv.style.display = 'flex';
        loadingText.style.display = 'block';

        setTimeout(() => {
            overlay.style.opacity = '1';
            loadingDiv.style.opacity = '1';
            loadingText.style.opacity = '1';
        }, 10);
    });
});
document.getElementById('manual-classification-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Mencegah form untuk refresh halaman secara default

    // Ambil input teks dari form
    const manualText = document.getElementById('manual-text').value.trim();

    if (manualText === "") {
        alert("Teks input tidak boleh kosong!");
        return;
    }

    // Menampilkan loading atau pesan sementara jika diperlukan
    const classificationResult = document.getElementById('classification-result');
    const sentimentResult = document.getElementById('sentimen-result');
    const confidenceScore = document.getElementById('confidence-score');
    classificationResult.style.display = "none"; // Sembunyikan hasil klasifikasi sebelumnya
    sentimentResult.textContent = "Loading..."; // Menampilkan pesan loading
    confidenceScore.textContent = ""; // Reset confidence score

    // Kirimkan data teks ke server menggunakan AJAX
    const xhr = new XMLHttpRequest();
    xhr.open('POST', 'klasifikasi-manual', true); // Endpoint API untuk klasifikasi manual
    xhr.setRequestHeader('Content-Type', 'application/json'); // Set header content type ke JSON

    // Kirimkan data teks dalam format JSON
    xhr.send(JSON.stringify({ text: manualText }));

    // Tangani respon dari server
    xhr.onload = function() {
        if (xhr.status === 200) {
            // Ambil hasil klasifikasi dari response
            const response = JSON.parse(xhr.responseText);
            const sentimen = response.sentimen; // Asumsikan server mengirimkan hasil prediksi sebagai 'sentimen'
            const confidence = response.confidence_score; // Ambil confidence score dari response
            
            // Tampilkan hasil klasifikasi
            sentimentResult.textContent = sentimen;
            confidenceScore.textContent = confidence ? confidence.toFixed(2) : "N/A"; // Tampilkan confidence score
            classificationResult.style.display = "block"; // Menampilkan hasil klasifikasi
        } else {
            // Tangani error jika ada
            alert("Terjadi kesalahan dalam klasifikasi. Silakan coba lagi.");
        }
    };
});

document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const loadingDiv = document.getElementById('loading');

    form.addEventListener('submit', function () {
        loadingDiv.style.display = 'block';
    });
});
