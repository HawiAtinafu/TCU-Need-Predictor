// React frontend component for the Transitional Care Predictor application

// Import necessary dependencies
import React, { useState } from "react";
import axios from "axios"; // Library for making HTTP requests
import "./style.css"; // External CSS file for styling
import Logo from './MCHS_PrimaryLogo_Black.jpg'; // Importing Mayo Clinic logo image

// Functional component definition
export default function App() {
  // State variables to manage form input, errors, error messages, and prediction result
  const [eid, setEID] = useState("");
  const [error, setError] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  // Function to handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault(); // Prevent default form submission behavior
    setError(null); // Clear any previous errors

    try {
      if (!eid.trim()) {
        // Validate input: Ensure Encounter ID is not empty
        setError("Please provide a valid Visit Encounter ID");
        setPredictionResult(null); // Clear prediction result
        return;
      }

      // Send POST request to backend server for prediction
      const response = await axios.post(
        "http://localhost:5000/predict", // Backend server endpoint
        { id: eid }, // Data payload containing Encounter ID
        { headers: { "Content-Type": "application/json" } } // Request headers
      );
      
      // Check response for prediction result
      if (response.data.probability !== undefined) {
        setPredictionResult(response.data.probability); // Set prediction result
      } else {
        setPredictionResult(null); // Clear prediction result
        setPredictionResult(response.data); // Set prediction result from response
      }
    } catch (error) {
      // Handle errors
      setPredictionResult(null); // Clear prediction result
      setError("Invalid Encounter ID"); // Set error message
      setErrorMessage(error.message); // Set error message
      console.error("Error:", error.message); // Log error message to console
    }
  };

  // JSX structure for rendering UI elements
  return (
    <>
      {/* Header section with Mayo Clinic logo */}
      <header>
        <img src={Logo} alt="Mayo Clinic Logo" />
      </header>

      {/* Main container */}
      <div className="container">
        <br />
        <br />

        {/* Input field and form */}
        <div className="inputField" id="inputField">
          <div className="head">
            <h1 className="enter">Transitional Care Predictor</h1>
          </div>
          <p className="msg">Enter Patient Encounter ID</p>
          <div className="form">
            <form onSubmit={handleSubmit}>
              <input
                type="text"
                value={eid}
                onChange={(e) => setEID(e.target.value)}
                placeholder="Enter Visit Encounter ID"
              />
              <br />
              <br />
              <button type="submit" className="submit">
                Submit
              </button>
            </form>
          </div>

          {/* Display error messages if present */}
          {error && <p className="error">{error}</p>}
          {error && <p className="error">{errorMessage}</p>}

          {predictionResult !== null && (
            <p className="result">
              Prediction Result:{" "}
              <span
                className="boldResult"
                style={{
                  color:
                    predictionResult >= 0 && predictionResult < 40
                      ? "#16FF00"
                      : predictionResult >= 40 && predictionResult <= 59
                      ? "yellow"
                      : "#F94C10"
                }}
              >
                {predictionResult} %
              </span>
            </p>
          )}
        </div>
      </div>

      {/* Footer section */}
      <footer>
        <p>Made with ©Mayo Clinic Health System 2024</p>
      </footer>
    </>
  );
}
