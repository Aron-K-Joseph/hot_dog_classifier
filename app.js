// Reference HTML elements
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("image-input");
const statusText = document.getElementById("status");
const resultsScreen = document.getElementById("results-screen");
const splashPage = document.getElementById("splash-page");

const resultImage = document.getElementById("result-image");
const resultText = document.getElementById("result-text");
const tryAgainBtn = document.getElementById("try-again");

// Handle form submit
form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
        alert("You must upload an image first!");
        return;
    }

    // Show "loading" message
    statusText.classList.remove("hidden");

    // Prepare file for backend
    const formData = new FormData();
    formData.append("file", file);

    // Send to Flask server
    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
    });

    const data = await response.json();

    // Hide loading text
    statusText.classList.add("hidden");

    // Show results screen
    splashPage.classList.add("hidden");
    resultsScreen.classList.remove("hidden");

    // Display preview image
    resultImage.src = URL.createObjectURL(file);

    // Display prediction text
    resultText.textContent = `Result: ${data.prediction}`;
});

// Reset UI
tryAgainBtn.addEventListener("click", () => {
    resultsScreen.classList.add("hidden");
    splashPage.classList.remove("hidden");
    fileInput.value = "";
});
