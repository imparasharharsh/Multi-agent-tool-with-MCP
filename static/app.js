/**
 * Multi-Agent Trip Planner - Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  const tripForm = document.getElementById("trip-form");
  const queryInput = document.getElementById("query-input");
  const submitBtn = document.getElementById("submit-btn");
  const loadingBox = document.getElementById("loading-box");
  const resultsCard = document.getElementById("results-card");
  const planContent = document.getElementById("plan-content");
  const intentPill = document.getElementById("intent-pill");
  const copyBtn = document.getElementById("copy-btn");

  const stepRouter = document.getElementById("step-router");
  const stepFlight = document.getElementById("step-flight");
  const stepHotel = document.getElementById("step-hotel");
  const stepItinerary = document.getElementById("step-itinerary");

  let currentPlanMarkdown = "";

  // Helper to reset step indicators
  function resetSteps() {
    [stepRouter, stepFlight, stepHotel, stepItinerary].forEach((el) => {
      el.classList.remove("active", "completed");
    });
  }

  // Set active step
  function setStep(stepEl, state) {
    if (state === "active") {
      stepEl.classList.remove("completed");
      stepEl.classList.add("active");
    } else if (state === "completed") {
      stepEl.classList.remove("active");
      stepEl.classList.add("completed");
    }
  }

  // Handle Quick Chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.getAttribute("data-query");
      queryInput.focus();
    });
  });

  // Simple Markdown Parser for tables, headings, bold, lists
  function parseMarkdown(md) {
    if (!md) return "<p>No output generated.</p>";

    let html = md
      // Escaping
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Headers
    html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, "<blockquote>$1</blockquote>");

    // Bold & Italics
    html = html.replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/gim, "<em>$1</em>");

    // Horizontal Rule
    html = html.replace(/^---$/gim, "<hr style='border:0; border-top:1px solid rgba(255,255,255,0.1); margin:20px 0;'>");

    // Simple Tables
    html = html.replace(/\|(.+)\|/g, function (match) {
      if (match.includes("---")) {
        return ""; // skip separator row
      }
      const cells = match
        .split("|")
        .filter((c, i, arr) => i > 0 && i < arr.length - 1);
      const isHeader = match.includes("Category") || match.includes("Activity") || match.includes("Airline") || match.includes("Hotel");
      const tag = isHeader ? "th" : "td";
      const row = cells.map((c) => `<${tag}>${c.trim()}</${tag}>`).join("");
      return `<tr>${row}</tr>`;
    });

    // Wrap tables
    html = html.replace(/(<tr>.*?<\/tr>(\s*<tr>.*?<\/tr>)*)/gis, "<table>$1</table>");

    // Unordered Lists
    html = html.replace(/^\s*[-•]\s+(.*$)/gim, "<li>$1</li>");
    html = html.replace(/(<li>.*?<\/li>(\s*<li>.*?<\/li>)*)/gis, "<ul>$1</ul>");

    // Paragraph breaks
    html = html.replace(/\n\n+/g, "<p></p>");

    return html;
  }

  // Handle Form Submit
  tripForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // UI state
    submitBtn.disabled = true;
    loadingBox.style.display = "block";
    resultsCard.style.display = "none";
    resetSteps();
    setStep(stepRouter, "active");

    // Simulate flow animation
    const progressTimer = setTimeout(() => {
      setStep(stepRouter, "completed");
      setStep(stepFlight, "active");
    }, 1500);

    const progressTimer2 = setTimeout(() => {
      setStep(stepFlight, "completed");
      setStep(stepHotel, "active");
    }, 3500);

    const progressTimer3 = setTimeout(() => {
      setStep(stepHotel, "completed");
      setStep(stepItinerary, "active");
    }, 5500);

    try {
      const response = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      clearTimeout(progressTimer);
      clearTimeout(progressTimer2);
      clearTimeout(progressTimer3);

      const data = await response.json();

      if (data.status === "success") {
        currentPlanMarkdown = data.final_plan || "Plan generated successfully.";

        // Update intent pill
        const intent = data.intent || "full_trip";
        if (intent === "hotel_only") {
          intentPill.textContent = "🏨 Hotel Stay Only";
          setStep(stepRouter, "completed");
          setStep(stepHotel, "completed");
        } else if (intent === "flight_only") {
          intentPill.textContent = "✈️ Flight Options Only";
          setStep(stepRouter, "completed");
          setStep(stepFlight, "completed");
        } else {
          intentPill.textContent = "🗺️ Complete Master Itinerary";
          [stepRouter, stepFlight, stepHotel, stepItinerary].forEach((el) => {
            setStep(el, "completed");
          });
        }

        // Render formatted HTML
        planContent.innerHTML = parseMarkdown(currentPlanMarkdown);
        resultsCard.style.display = "block";
      } else {
        alert("Error: " + (data.message || "Failed to process request."));
      }
    } catch (err) {
      alert("Network error connecting to trip planner backend: " + err.message);
    } finally {
      submitBtn.disabled = false;
      loadingBox.style.display = "none";
    }
  });

  // Copy Plan to Clipboard
  copyBtn.addEventListener("click", () => {
    if (!currentPlanMarkdown) return;
    navigator.clipboard.writeText(currentPlanMarkdown).then(() => {
      const originalText = copyBtn.innerHTML;
      copyBtn.innerHTML = "✓ Copied to Clipboard";
      setTimeout(() => {
        copyBtn.innerHTML = originalText;
      }, 2000);
    });
  });
});
