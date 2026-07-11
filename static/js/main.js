// Predict button + page-wait animation.
// This is a normal server-rendered form (full page reload), so we can't
// know exactly when the response lands — we just show the spinner the
// moment the user clicks, and it naturally disappears when the new
// page (with results) finishes loading.
document.addEventListener("DOMContentLoaded", function () {
  var form = document.querySelector("form.search");
  var btn = document.querySelector(".predict-btn");
  var overlay = document.getElementById("page-overlay");

  if (form && btn) {
    form.addEventListener("submit", function () {
      btn.classList.add("shrinking", "loading");
      if (overlay) overlay.classList.add("active");
    });
  }
});
