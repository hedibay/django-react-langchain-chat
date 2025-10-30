import React, { useState } from 'react';
import './ObjectDetector.css';

const ObjectDetector = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [detections, setDetections] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [settings, setSettings] = useState({
        min_confidence: 0.35,
        max_results: 50,
        class_filter: '',
        nms_iou: 0.45
    });

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setError(null);
            setDetections(null);
            
            // Create preview
            const reader = new FileReader();
            reader.onload = (e) => {
                setImagePreview(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleDetect = async () => {
        if (!selectedFile) {
            setError('Please select an image first');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('min_confidence', settings.min_confidence.toString());
            formData.append('max_results', settings.max_results.toString());
            formData.append('nms_iou', settings.nms_iou.toString());
            
            if (settings.class_filter.trim()) {
                formData.append('class_filter', settings.class_filter);
            }

            const response = await fetch('http://127.0.0.1:8000/api/detect/', {
                method: 'POST',
                body: formData,
                mode: 'cors',
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setDetections(data);
        } catch (err) {
            setError(`Detection failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setSelectedFile(null);
        setImagePreview(null);
        setDetections(null);
        setError(null);
        setLoading(false);
    };

    const drawDetections = () => {
        if (!imagePreview || !detections) return null;

        return (
            <div className="detection-overlay">
                <img src={imagePreview} alt="Detection" className="detection-image" />
                <svg className="detection-svg" viewBox={`0 0 ${detections.image_size[0]} ${detections.image_size[1]}`}>
                    {detections.detections.map((detection, index) => (
                        <g key={index}>
                            <rect
                                x={detection.bbox[0]}
                                y={detection.bbox[1]}
                                width={detection.bbox[2]}
                                height={detection.bbox[3]}
                                fill="none"
                                stroke="#00ff00"
                                strokeWidth="2"
                            />
                            <text
                                x={detection.bbox[0]}
                                y={detection.bbox[1] - 5}
                                fill="#00ff00"
                                fontSize="14"
                                fontWeight="bold"
                            >
                                {detection.class_name} ({Math.round(detection.confidence * 100)}%)
                            </text>
                        </g>
                    ))}
                </svg>
            </div>
        );
    };

    return (
        <div className="object-detector">
            <h2>🔍 Object Detector</h2>
            
            <div className="upload-area">
                <input
                    type="file"
                    id="image-upload"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="file-input"
                />
                <label htmlFor="image-upload" className="upload-label">
                    {selectedFile ? selectedFile.name : '📁 Choose Image'}
                </label>
            </div>

            <div className="settings-panel">
                <h3>Detection Settings</h3>
                <div className="settings-grid">
                    <div className="setting-item">
                        <label>Min Confidence:</label>
                        <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={settings.min_confidence}
                            onChange={(e) => setSettings({...settings, min_confidence: parseFloat(e.target.value)})}
                        />
                        <span>{settings.min_confidence}</span>
                    </div>
                    <div className="setting-item">
                        <label>Max Results:</label>
                        <input
                            type="number"
                            min="1"
                            max="100"
                            value={settings.max_results}
                            onChange={(e) => setSettings({...settings, max_results: parseInt(e.target.value)})}
                        />
                    </div>
                    <div className="setting-item">
                        <label>Class Filter (comma-separated):</label>
                        <input
                            type="text"
                            placeholder="person, car, dog"
                            value={settings.class_filter}
                            onChange={(e) => setSettings({...settings, class_filter: e.target.value})}
                        />
                    </div>
                    <div className="setting-item">
                        <label>NMS IoU:</label>
                        <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={settings.nms_iou}
                            onChange={(e) => setSettings({...settings, nms_iou: parseFloat(e.target.value)})}
                        />
                        <span>{settings.nms_iou}</span>
                    </div>
                </div>
            </div>

            <div className="actions">
                <button
                    onClick={handleDetect}
                    disabled={!selectedFile || loading}
                    className="detect-btn"
                >
                    {loading ? '🔄 Detecting...' : '🔍 Detect Objects'}
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

            {detections && (
                <div className="results">
                    <h3>🎯 Detection Results</h3>
                    <div className="detection-stats">
                        <p>Found {detections.detections.length} objects</p>
                        <p>Processing time: {detections.processing_time}s</p>
                        <p>Image size: {detections.image_size[0]}x{detections.image_size[1]}</p>
                    </div>
                    
                    {imagePreview && (
                        <div className="detection-visualization">
                            {drawDetections()}
                        </div>
                    )}
                    
                    <div className="detection-list">
                        {detections.detections.map((detection, index) => (
                            <div key={index} className="detection-item">
                                <div className="detection-info">
                                    <span className="class-name">{detection.class_name}</span>
                                    <span className="confidence">{Math.round(detection.confidence * 100)}%</span>
                                </div>
                                <div className="bbox-info">
                                    BBox: [{detection.bbox.join(', ')}]
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ObjectDetector;

