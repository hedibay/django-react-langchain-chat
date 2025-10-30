import React, { useState } from 'react';
import './DogClassifier.css';

const DogClassifier = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [predictions, setPredictions] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setError(null);
            setPredictions(null);
        }
    };

    const handleClassify = async () => {
        if (!selectedFile) {
            setError('Please select an image first');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('species', 'dog');

            const response = await fetch('http://127.0.0.1:8000/api/dog-classify/', {
                method: 'POST',
                body: formData,
                mode: 'cors',
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setPredictions(data.predictions);
        } catch (err) {
            setError(`Classification failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setSelectedFile(null);
        setPredictions(null);
        setError(null);
        setLoading(false);
    };

    return (
        <div className="dog-classifier">
            <h2>🐕 Dog Breed Classifier</h2>
            
            <div className="upload-area">
                <input
                    type="file"
                    id="image-upload"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="file-input"
                />
                <label htmlFor="image-upload" className="upload-label">
                    {selectedFile ? selectedFile.name : '📁 Choose Dog Image'}
                </label>
            </div>

            <div className="actions">
                <button
                    onClick={handleClassify}
                    disabled={!selectedFile || loading}
                    className="classify-btn"
                >
                    {loading ? '🔄 Analyzing...' : '🔍 Identify Breed'}
                </button>
                <button onClick={handleReset} className="reset-btn">
                    🗑️ Clear
                </button>
            </div>

            {error && (
                <div className="error">
                    ❌ {error}
                </div>
            )}

            {predictions && (
                <div className="results">
                    <h3>🎯 Results</h3>
                    {predictions.map((prediction, index) => (
                        <div key={index} className="result-item">
                            <div className="rank">#{index + 1}</div>
                            <div className="breed">{prediction.breed}</div>
                            <div className="probability">{prediction.probability}%</div>
                            <div className="bar">
                                <div 
                                    className="fill"
                                    style={{ width: `${prediction.probability}%` }}
                                ></div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default DogClassifier;
