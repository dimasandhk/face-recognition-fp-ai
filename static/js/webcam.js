const video = document.getElementById('video');
const captureButton = document.getElementById('captureButton');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
        video.play();
    })
    .catch((error) => {
        console.error("Error accessing webcam: ", error);
        resultDiv.innerHTML = "<p style='color: red;'>Unable to access the webcam.</p>";
    });

captureButton.addEventListener('click', () => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/png');

    const formData = new FormData();
    formData.append('image', dataURLtoBlob(imageData));

    fetch('/recognize', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<p style='color: red;'>Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `<p style='color: green;'>Recognition Result: ${data.result}</p>`;
        }
    })
    .catch(error => {
        console.error("Error during recognition: ", error);
        resultDiv.innerHTML = "<p style='color: red;'>An error occurred during recognition.</p>";
    });
});

function dataURLtoBlob(dataURL) {
    const byteString = atob(dataURL.split(',')[1]);
    const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
}