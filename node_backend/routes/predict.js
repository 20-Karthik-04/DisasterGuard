const express = require("express");
const axios = require("axios");
const router = express.Router();

router.post("/", async (req, res) => {
  try {
    const response = await axios.post("http://127.0.0.1:5000/predict", {
      tweet: req.body.tweet,
    });
    res.json(response.data);
  } catch (error) {
    console.error("Error calling Python microservice:", error.message);
    res.status(500).json({ error: "Prediction service unavailable" });
  }
});

module.exports = router;
