import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import DogClassifier from './DogClassifier';
import ObjectDetector from './ObjectDetector';
import './App.css';

const StreamLangchain = () => {
    // State to store the input from the user
    const [input, setInput] = useState('');
    const [mode, setMode] = useState('chat'); // 'chat' | 'cv_critic' | 'career_mapper' | 'dog_classifier' | 'object_detector'
    const [job, setJob] = useState(''); // Optional job description
    const [pdfFile, setPdfFile] = useState(null);
    const [tutor, setTutor] = useState({ language: '', lesson_mode: 'casual' });
    const [tutorState, setTutorState] = useState(null);
    const [careerProfile, setCareerProfile] = useState({ current_title: '', industry: '', skills: '', education: '', goals: '', detail: 'summary' });
    const [careerRoles, setCareerRoles] = useState([]);
    // State to store the responses/messages
    const [responses, setResponses] = useState([]);
    // Ref to manage the WebSocket connection
    const ws = useRef(null);
    // Ref to scroll to the latest message
    const messagesEndRef = useRef(null);
    // Maximum number of attempts to reconnect
    const [reconnectAttempts, setReconnectAttempts] = useState(0);
    const maxReconnectAttempts = 5;
    // Connection status
    const [isConnected, setIsConnected] = useState(false);

    // Function to setup the WebSocket connection and define event handlers
    const setupWebSocket = () => {
        ws.current = new WebSocket('ws://127.0.0.1:8000/ws/chat/');
        let ongoingStream = null; // To track the ongoing stream's ID

        ws.current.onopen = () => {
            console.log("WebSocket connected!");
            setReconnectAttempts(0); // Reset reconnect attempts on successful connection
            setIsConnected(true);
        };

        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            let sender = data.name || "Assistant";

            // Handle different types of events from the WebSocket
            if (data.event === 'on_parser_start') {
                // When a new stream starts
                ongoingStream = { id: data.run_id, content: '' };
                setResponses(prevResponses => [...prevResponses, { sender, message: '', id: data.run_id }]);
            } else if (data.event === 'on_parser_stream' && ongoingStream && data.run_id === ongoingStream.id) {
                // During a stream, appending new chunks of data
                setResponses(prevResponses => prevResponses.map(msg =>
                    msg.id === data.run_id ? { ...msg, message: msg.message + data.data.chunk } : msg));
            } else if (data.event === 'error') {
                // Handle error responses
                setResponses(prevResponses => [...prevResponses, { sender: "System", message: `Error: ${data.text}` }]);
            } else if (data.event === 'career_roles') {
                setCareerRoles(data.roles || []);
            } else if (data.event === 'language_tutor_state') {
                setTutorState(data.state || null);
            }
        };

        ws.current.onerror = (event) => {
            console.error("WebSocket error observed:", event);
        };

        ws.current.onclose = (event) => {
            console.log(`WebSocket is closed now. Code: ${event.code}, Reason: ${event.reason}`);
            setIsConnected(false);
            handleReconnect();
        };
    };

    // Function to handle reconnection attempts with exponential backoff
    const handleReconnect = () => {
        if (reconnectAttempts < maxReconnectAttempts) {
            let timeout = Math.pow(2, reconnectAttempts) * 1000; // Exponential backoff
            setTimeout(() => {
                setupWebSocket(); // Attempt to reconnect
            }, timeout);
        } else {
            console.log("Max reconnect attempts reached, not attempting further reconnects.");
        }
    };

    // Effect hook to setup and cleanup the WebSocket connection
    useEffect(() => {
        setupWebSocket(); // Setup WebSocket on component mount

        return () => {
            if (ws.current.readyState === WebSocket.OPEN) {
                ws.current.close(); // Close WebSocket on component unmount
            }
        };
    }, []);

    // Reset file-specific state when switching modes
    useEffect(() => {
        if (mode === 'chat') {
            setPdfFile(null);
        }
    }, [mode]);

    // Effect hook to auto-scroll to the latest message
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [responses]);

    // Function to render each message
    const renderMessage = (response, index) => (
        <div key={index} className={`message ${response.sender}`}>
            <strong>{response.sender}</strong>
            <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {response.message}
                </ReactMarkdown>
            </div>
        </div>
    );

    // Handler for input changes
    const handleInputChange = (e) => {
        setInput(e.target.value);
    };

    // Handler for form submission
    const handleSubmit = (e) => {
        e.preventDefault();
            const trimmed = input.trim();

            // Optional slash commands to change mode quickly
            const lower = trimmed.toLowerCase();
            if (lower === '/mode cv' || lower === '/mode cv_critic') {
                setMode('cv_critic');
                setResponses(prev => [...prev, { sender: 'System', message: 'Mode set to CV Critic' }]);
                setInput('');
                return;
            }
            if (lower === '/mode chat') {
                setMode('chat');
                setResponses(prev => [...prev, { sender: 'System', message: 'Mode set to General Chat' }]);
                setInput('');
                return;
            }

            if (mode === 'cv_critic') {
                const userMessage = { sender: "You", message: pdfFile ? 'Uploading PDF for critique...' : 'Requesting CV critique...' };
                setResponses(prevResponses => [...prevResponses, userMessage]);

                if (pdfFile) {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const base64 = reader.result; // data:*/*;base64,...
                        ws.current.send(JSON.stringify({ type: 'cv_critic_pdf', pdf_base64: base64, job }));
                    };
                    reader.readAsDataURL(pdfFile);
                } else {
                    if (!trimmed) {
                        setResponses(prev => [...prev, { sender: 'System', message: 'Please paste resume text or upload a PDF.' }]);
                        return;
                    }
                    ws.current.send(JSON.stringify({ type: 'cv_critic', resume: trimmed, job }));
                }
                setInput('');
                return;
            }

            if (mode === 'career_mapper') {
                const profilePayload = {
                    current_title: careerProfile.current_title,
                    industry: careerProfile.industry,
                    skills: careerProfile.skills.split(',').map(s => s.trim()).filter(Boolean),
                    education: careerProfile.education,
                    goals: careerProfile.goals,
                };
                setResponses(prev => [...prev, { sender: 'You', message: 'Generate career path plan' }]);
                ws.current.send(JSON.stringify({ type: 'career_mapper', profile: profilePayload, detail: careerProfile.detail }));
                return;
            }

            if (mode === 'language_tutor') {
                const trimmed = input.trim();
                if (!trimmed) return;
                setResponses(prev => [...prev, { sender: 'You', message: trimmed }]);
                ws.current.send(JSON.stringify({ type: 'language_tutor_message', message: trimmed }));
                setInput('');
                return;
            }

            if (mode === 'career_mapper') {
                const profilePayload = {
                    current_title: careerProfile.current_title,
                    industry: careerProfile.industry,
                    skills: careerProfile.skills.split(',').map(s => s.trim()).filter(Boolean),
                    education: careerProfile.education,
                    goals: careerProfile.goals,
                };
                setResponses(prev => [...prev, { sender: 'You', message: 'Generate career path plan' }]);
                ws.current.send(JSON.stringify({ type: 'career_mapper', profile: profilePayload, detail: careerProfile.detail }));
                return;
            }

            if (!trimmed) {
                return;
            }
            const userMessage = { sender: "You", message: trimmed };
            setResponses(prevResponses => [...prevResponses, userMessage]);
            ws.current.send(JSON.stringify({ type: 'chat', message: trimmed }));
            setInput('');
    };

    return (
        <div className="chat-container">
            <div className="connection-status">
                <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                    {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                </span>
            </div>
            <div className="mode-panel">
                <label>
                    Mode:
                    <select value={mode} onChange={(e) => { const next = e.target.value; console.log('Mode:', next); setMode(next); }}>
                        <option value="chat">General Chat</option>
                        <option value="cv_critic">CV Critic</option>
                        <option value="career_mapper">Career Path Mapper</option>
                        <option value="language_tutor">Language Tutor</option>
                        <option value="dog_classifier">Dog Breed Classifier</option>
                        <option value="object_detector">Object Detector</option>
                    </select>
                </label>
                {mode === 'cv_critic' && (
                    <>
                        <textarea
                            placeholder="Optional: paste job description here to tailor the critique"
                            value={job}
                            onChange={(e) => setJob(e.target.value)}
                            rows={4}
                            style={{ width: '100%', marginTop: 8 }}
                        />
                        <div style={{ marginTop: 8 }}>
                            <label>
                                Upload PDF resume (optional):
                                <input
                                    type="file"
                                    accept="application/pdf"
                                    onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
                                    style={{ display: 'block', marginTop: 4 }}
                                />
                            </label>
                            {pdfFile && (
                                <small>Selected: {pdfFile.name} ({Math.round(pdfFile.size/1024)} KB)</small>
                            )}
                        </div>
                    </>
                )}
                {mode === 'career_mapper' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8, maxWidth: 760 }}>
                        <input placeholder="Current Title" value={careerProfile.current_title} onChange={(e) => setCareerProfile(p => ({...p, current_title: e.target.value}))} />
                        <input placeholder="Industry" value={careerProfile.industry} onChange={(e) => setCareerProfile(p => ({...p, industry: e.target.value}))} />
                        <input placeholder="Skills (comma-separated)" value={careerProfile.skills} onChange={(e) => setCareerProfile(p => ({...p, skills: e.target.value}))} />
                        <input placeholder="Education" value={careerProfile.education} onChange={(e) => setCareerProfile(p => ({...p, education: e.target.value}))} />
                        <input placeholder="Career Goals" value={careerProfile.goals} onChange={(e) => setCareerProfile(p => ({...p, goals: e.target.value}))} />
                        <label>
                            Detail:
                            <select value={careerProfile.detail} onChange={(e) => setCareerProfile(p => ({...p, detail: e.target.value}))}>
                                <option value="summary">Quick Summary</option>
                                <option value="deep">In-Depth Plan</option>
                            </select>
                        </label>
                        {careerRoles.length > 0 && (
                            <div style={{ gridColumn: '1 / -1' }}>
                                <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                                    {careerRoles.map((r, idx) => (
                                        <button key={idx} type="button" onClick={() => ws.current.send(JSON.stringify({ type: 'career_mapper_role_detail', role: r }))} style={{ borderRadius: 16, padding: '6px 10px' }}>{r}</button>
                                    ))}
                                </div>
                            </div>
                        )}
                        <div style={{ gridColumn: '1 / -1' }}>
                            <button type="button" onClick={() => {
                                const profilePayload = {
                                    current_title: careerProfile.current_title,
                                    industry: careerProfile.industry,
                                    skills: careerProfile.skills.split(',').map(s => s.trim()).filter(Boolean),
                                    education: careerProfile.education,
                                    goals: careerProfile.goals,
                                };
                                setResponses(prev => [...prev, { sender: 'You', message: 'Generate career path plan' }]);
                                ws.current.send(JSON.stringify({ type: 'career_mapper', profile: profilePayload, detail: careerProfile.detail }));
                            }}>Generate Plan</button>
                        </div>
                    </div>
                )}
                {mode === 'language_tutor' && (
                    <div style={{ display: 'grid', gap: 8, marginTop: 8, maxWidth: 760 }}>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            <input placeholder="Target Language (e.g., Spanish)" value={tutor.language} onChange={(e) => setTutor(t => ({ ...t, language: e.target.value }))} />
                            <label>
                                Tutor Mode:
                                <select value={tutor.lesson_mode} onChange={(e) => setTutor(t => ({ ...t, lesson_mode: e.target.value }))}>
                                    <option value="casual">Casual</option>
                                    <option value="formal">Formal</option>
                                    <option value="exam">Exam Prep</option>
                                </select>
                            </label>
                            <button type="button" onClick={() => {
                                if (!tutor.language.trim()) {
                                    setResponses(prev => [...prev, { sender: 'System', message: 'Please choose a target language.' }]);
                                    return;
                                }
                                setResponses(prev => [...prev, { sender: 'You', message: `Start ${tutor.language} tutor (${tutor.lesson_mode})` }]);
                                ws.current.send(JSON.stringify({ type: 'language_tutor_start', language: tutor.language.trim(), lesson_mode: tutor.lesson_mode }));
                            }}>Start Tutor</button>
                            <button type="button" disabled={!input.trim()} onClick={() => {
                                const trimmed = input.trim();
                                setResponses(prev => [...prev, { sender: 'You', message: trimmed }]);
                                ws.current.send(JSON.stringify({ type: 'language_tutor_message', message: trimmed }));
                                setInput('');
                            }}>Send Message</button>
                            <button type="button" disabled={!input.trim()} onClick={() => {
                                const trimmed = input.trim();
                                setResponses(prev => [...prev, { sender: 'You', message: `Answer: ${trimmed}` }]);
                                ws.current.send(JSON.stringify({ type: 'language_tutor_submit', answer: trimmed, exercise: '' }));
                                setInput('');
                            }}>Submit Answer</button>
                        </div>
                        {tutorState && (
                            <div className="markdown-body" style={{ fontSize: 14 }}>
                                <p>Language: <strong>{tutorState.language || '-'}</strong> | Mode: <strong>{tutorState.mode || '-'}</strong> | Level: <strong>{tutorState.level || '-'}</strong></p>
                                <p>Accuracy: {tutorState.answers_correct}/{tutorState.answers_total} ({tutorState.accuracy_percent}%)</p>
                                {tutorState.learned_words?.length > 0 && (
                                    <p>Learned: {tutorState.learned_words.join(', ')}</p>
                                )}
                            </div>
                        )}
                    </div>
                )}
                {mode === 'dog_classifier' && (
                    <DogClassifier />
                )}
                {mode === 'object_detector' && (
                    <ObjectDetector />
                )}
            </div>
            {mode !== 'dog_classifier' && mode !== 'object_detector' && (
                <>
                    <div className="messages-container">
                        {responses.map((response, index) => renderMessage(response, index))}
                        <div ref={messagesEndRef} /> {/* Invisible element to help scroll into view */}
                    </div>
            <form onSubmit={handleSubmit} className="input-form">
                {mode === 'cv_critic' ? (
                    <textarea
                        value={input}
                        onChange={handleInputChange}
                        placeholder="Paste your resume text here for critique"
                        disabled={!isConnected}
                        rows={6}
                    />
                ) : (
                    <input
                        type="text"
                        value={input}
                        onChange={handleInputChange}
                        placeholder={mode === 'language_tutor' ? 'Type in the target language or answer an exercise...' : 'Type your message here...'}
                        disabled={!isConnected}
                    />
                )}
                <button type="submit" disabled={!isConnected}>Send</button>
            </form>
                </>
            )}
        </div>
    );
};

export default StreamLangchain;