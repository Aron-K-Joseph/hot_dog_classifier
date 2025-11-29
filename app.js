// Grab DOM elements
const splashPage = document.getElementById("splash-page");
const resultsScreen = document.getElementById("results-screen");

const uploadForm = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const statusText = document.getElementById("status");

const resultImage = document.getElementById("result-image");
const resultIcon = document.getElementById("result-icon");
const resultText = document.getElementById("result-text");
const resultConfidence = document.getElementById("result-confidence");
const resultEaten = document.getElementById("result-eaten");

const tryAgainBtn = document.getElementById("try-again");

// just to see the result page without a real model yet.
function mockClassify() {
  const isHotDog = Math.random() > 0.5;
  const confidence = 0.6 + Math.random() * 0.39; // 60–99%
  const eatenPercent = isHotDog ? Math.floor(Math.random() * 101) : 0; // 0–100%

  return { isHotDog, confidence, eatenPercent };
}

// HANDLE FORM SUBMIT 
uploadForm.addEventListener("submit", (e) => {
  e.preventDefault(); // don't reload the page

  const file = imageInput.files[0];
  if (!file) return;

  // Show "analyzing..." text 
  statusText.classList.remove("hidden");
  statusText.textContent = "Consulting the forces of Egypt... identifying...";

  // Show a preview of the image on the results page
  const objectURL = URL.createObjectURL(file);
  resultImage.src = objectURL;

  // Simulate model compute time
  setTimeout(() => {
    const { isHotDog, confidence, eatenPercent } = mockClassify();

    if (isHotDog) {
      resultIcon.textContent = "✅";
      resultText.textContent = "The Pharaohs declare: HOT DOG.";
    } else {
      resultIcon.textContent = "❌";
      resultText.textContent = "The omens say: NOT a hot dog.";
    }

    const confPct = Math.round(confidence * 100);
    resultConfidence.textContent = `Confidence: ${confPct}%`;
    resultEaten.textContent = `Portion devoured: ${eatenPercent}%`;

    // Hide splash page, show results page
    splashPage.classList.add("hidden");
    resultsScreen.classList.remove("hidden");

    // Hide the analyzing text 
    statusText.classList.add("hidden");
  }, 800); // 0.8s fake delay
});

// Submit another image button 
tryAgainBtn.addEventListener("click", () => {
  // Reset file input and result text
  imageInput.value = "";
  resultImage.src = "";
  resultIcon.textContent = "";
  resultText.textContent = "";
  resultConfidence.textContent = "";
  resultEaten.textContent = "";

  // Go back to splash page
  resultsScreen.classList.add("hidden");
  splashPage.classList.remove("hidden");
});
