document.addEventListener("DOMContentLoaded", function () {
    const toggle = document.getElementById("modeToggle");
    const modeInput = document.getElementById("modeInput");
    const modeLabel = document.getElementById("modeLabel");

    if (!toggle || !modeInput || !modeLabel) {
        console.warn("⚠️ Switch elements not found in DOM.");
        return;
    }

    function updateModeLabel() {
        const mode = toggle.checked ? "lemmas" : "tokens";
        modeInput.value = mode;
        modeLabel.textContent = mode;
    }

    updateModeLabel();

    toggle.addEventListener("change", updateModeLabel);
});
