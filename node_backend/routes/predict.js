const express = require("express");
const axios = require("axios");
const router = express.Router();

// Enhanced prediction endpoint with better error handling
router.post("/", async (req, res) => {
  try {
    const { tweet } = req.body;
    
    if (!tweet || typeof tweet !== 'string' || tweet.trim().length < 3) {
      return res.status(400).json({ 
        error: "Invalid tweet provided. Tweet must be at least 3 characters long." 
      });
    }

    const response = await axios.post("http://127.0.0.1:4000/analyze", {
      tweet: tweet.trim(),
    }, {
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'application/json'
      }
    });

    res.json(response.data);
  } catch (error) {
    console.error("Error calling Python microservice:", error.message);
    
    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ 
        error: "Prediction service is currently unavailable. Please try again later." 
      });
    } else if (error.response) {
      res.status(error.response.status).json({ 
        error: error.response.data.error || "Error processing request" 
      });
    } else {
      res.status(500).json({ 
        error: "Internal server error occurred during prediction" 
      });
    }
  }
});

// Batch prediction endpoint for multiple tweets
router.post("/batch", async (req, res) => {
  try {
    const { tweets } = req.body;
    
    if (!Array.isArray(tweets) || tweets.length === 0) {
      return res.status(400).json({ 
        error: "Invalid tweets array provided" 
      });
    }

    if (tweets.length > 100) {
      return res.status(400).json({ 
        error: "Maximum 100 tweets allowed per batch request" 
      });
    }

    const results = [];
    for (let i = 0; i < tweets.length; i++) {
      try {
        const response = await axios.post("http://127.0.0.1:4000/analyze", {
          tweet: tweets[i].trim(),
        }, {
          timeout: 10000,
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        results.push({
          index: i,
          tweet: tweets[i],
          result: response.data
        });
      } catch (error) {
        results.push({
          index: i,
          tweet: tweets[i],
          error: "Failed to process this tweet"
        });
      }
    }

    res.json({ results });
  } catch (error) {
    console.error("Error in batch prediction:", error.message);
    res.status(500).json({ 
      error: "Internal server error occurred during batch prediction" 
    });
  }
});

module.exports = router;
