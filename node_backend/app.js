const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();

// Enable CORS for all routes, which is good practice for a frontend server
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Import prediction routes
const predictRoutes = require("./routes/predict");
app.use("/api/predict", predictRoutes);

// Serve static files (css, js, images) from the 'public/static' directory
app.use("/static", express.static(path.join(__dirname, "public/static")));

// --- Serve main HTML pages ---
const templatesPath = path.join(__dirname, "public/templates");

app.get("/", (req, res) => {
  res.sendFile(path.join(templatesPath, "index.html"));
});

app.get("/about-us", (req, res) => {
  res.sendFile(path.join(templatesPath, "about-us.html"));
});

app.get("/feedback", (req, res) => {
  res.sendFile(path.join(templatesPath, "feedback.html"));
});

app.get("/feedback_submitted", (req, res) => {
  res.sendFile(path.join(templatesPath, "feedback_submitted.html"));
});

app.get("/motivation", (req, res) => {
  res.sendFile(path.join(templatesPath, "motivation.html"));
});

app.get("/model-insight", (req, res) => {
  res.sendFile(path.join(templatesPath, "model-insight.html"));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Node frontend server running at http://localhost:${PORT}`);
});
