import React, { useState, useEffect } from 'react';
import Field from './Field';

export default function Dashboard() {
    const [trace, setTrace] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [progress, setProgress] = useState(0); // 0.0 to 1.0
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        let animationFrame;
        if (isPlaying) {
            const animate = () => {
                setProgress(prev => {
                    if (prev >= 1) {
                        setIsPlaying(false);
                        return 1;
                    }
                    return prev + 0.005; // Speed of animation
                });
                animationFrame = requestAnimationFrame(animate);
            };
            animationFrame = requestAnimationFrame(animate);
        }
        return () => cancelAnimationFrame(animationFrame);
    }, [isPlaying]);

    useEffect(() => {
        fetch('/trace.json')
            .then(res => res.json())
            .then(data => setTrace(data))
            .catch(err => console.error("Failed to load trace:", err));
    }, []);

    if (!trace.length) return <div className="p-10 text-center">Loading Play Data...</div>;

    const currentItem = trace[currentIndex];

    const togglePlay = () => {
        if (progress >= 1) setProgress(0); // Restart if at end
        setIsPlaying(!isPlaying);
    };

    const handleNext = () => {
        if (currentIndex < trace.length - 1) {
            setCurrentIndex(currentIndex + 1);
            setProgress(0);
        }
    };

    const handlePrev = () => {
        if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
            setProgress(0);
        }
    };

    // Check for Summary Slide
    if (currentItem.type === 'summary') {
        const { best_iteration, score, play, critique, rationale } = currentItem;
        return (
            <div className="min-h-screen bg-gray-900 text-white font-sans p-6 flex flex-col items-center justify-center">
                <div className="max-w-4xl w-full bg-white text-black rounded-lg shadow-2xl p-10 border-4 border-black">
                    <h1 className="text-5xl font-black mb-6 text-center tracking-tighter uppercase text-black">
                        The Winning Play
                    </h1>

                    <div className="flex justify-between items-center mb-8 border-b-2 border-gray-200 pb-6">
                        <div>
                            <p className="text-gray-500 text-sm uppercase tracking-widest font-bold">Concept Name</p>
                            <h2 className="text-4xl font-bold">{play.play_name}</h2>
                        </div>
                        <div className="text-right">
                            <p className="text-gray-500 text-sm uppercase tracking-widest font-bold">Formation</p>
                            <h2 className="text-3xl font-mono">{play.personnel_formation}</h2>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-8 mb-8">
                        <div className="bg-blue-50 p-6 rounded-lg border border-blue-100">
                            <h3 className="text-xl font-bold text-blue-900 mb-2">Why it Won</h3>
                            <p className="text-lg leading-relaxed">{rationale}</p>
                            <div className="mt-4 pt-4 border-t border-blue-200">
                                <span className="font-bold text-blue-800">Iteration #{best_iteration}</span>
                                <span className="mx-2">|</span>
                                <span className="font-bold text-green-700">Composite Score: {score.toFixed(2)}</span>
                            </div>
                        </div>

                        <div className="grid grid-rows-2 gap-4">
                            <div className="bg-gray-100 p-4 rounded flex flex-col justify-center items-center">
                                <span className="text-4xl font-black">{critique.expected_yards}</span>
                                <span className="text-sm uppercase tracking-wide text-gray-600">Expected Yards</span>
                            </div>
                            <div className="bg-gray-100 p-4 rounded flex flex-col justify-center items-center">
                                <span className="text-4xl font-black">{critique.predictability_score.toFixed(4)}</span>
                                <span className="text-sm uppercase tracking-wide text-gray-600">Predictability</span>
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={handlePrev}
                        className="w-full py-4 bg-black text-white font-bold text-lg rounded hover:bg-gray-800 transition-colors uppercase tracking-widest"
                    >
                        Review Process
                    </button>
                </div>
            </div>
        );
    }

    const { play, critique, rationale, iteration } = currentItem;



    const Scoreboard = () => (
        <div className="flex w-full mb-4 shadow-2xl rounded overflow-hidden font-bold font-mono border-2 border-black">
            {/* Patriots (Away/Home?) - Let's put PHI (Left) NE (Right) style */}
            <div className="flex-1 bg-[#002244] text-white p-4 flex flex-col items-center justify-center border-r-2 border-white">
                <div className="text-3xl">NE</div>
                <div className="text-5xl">0</div>
            </div>

            {/* Middle Section: White BG, Black Text */}
            <div className="flex-[2] bg-white text-black p-2 flex flex-col items-center justify-center">
                <div className="text-black text-3xl mb-2 font-black uppercase font-sans tracking-tight">1st & 10</div>
                <div className="text-black text-xl font-bold">BALL ON 35</div>
                <div className="text-gray-500 text-sm mt-1">Q1 15:00</div>
                <div className="text-xs text-gray-500 mt-2 tracking-widest uppercase">Iteration {iteration} / {trace.length}</div>
            </div>

            <div className="flex-1 bg-[#4F2683] text-white p-4 flex flex-col items-center justify-center border-l-2 border-white">
                <div className="text-3xl">MIN</div>
                <div className="text-5xl">0</div>
            </div>
        </div>
    );

    // Helper to format bullet points
    const FormatBullets = ({ text }) => {
        if (!text) return "No rationale.";
        // Split by sentences, remove empty
        const sentences = text.split('. ').filter(s => s.trim().length > 0);
        return (
            <ul className="list-disc pl-5 space-y-2">
                {sentences.map((s, i) => (
                    <li key={i} className="text-sm text-gray-700 leading-relaxed">
                        {s.trim()}{s.endsWith('.') ? '' : '.'}
                    </li>
                ))}
            </ul>
        );
    };

    return (
        <div className="min-h-screen bg-white text-black font-sans p-6">

            {/* Top Banner Scoreboard */}
            <Scoreboard />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-6">

                {/* Left Column: Visuals */}
                <div className="lg:col-span-2 flex flex-col gap-4">
                    <Field play={currentItem} progress={progress} />

                    {/* Timeline Slider & Play Button */}
                    <div className="bg-gray-100 p-4 rounded-lg border-2 border-black flex gap-4 items-center">
                        <button
                            onClick={togglePlay}
                            className="w-12 h-12 bg-black text-white rounded-full flex items-center justify-center hover:bg-gray-800 transition shadow-lg"
                        >
                            {isPlaying ? (
                                <span className="block w-4 h-4 bg-white rounded-sm"></span> // Pause Icon
                            ) : (
                                <span className="block w-0 h-0 border-t-8 border-t-transparent border-l-12 border-l-white border-b-8 border-b-transparent ml-1"></span> // Play Icon
                            )}
                        </button>

                        <div className="flex-1">
                            <label className="block text-sm font-bold mb-2 uppercase tracking-widest">run play simulation</label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.005"
                                value={progress}
                                onChange={(e) => {
                                    setProgress(parseFloat(e.target.value));
                                    setIsPlaying(false); // Stop playing if user drags
                                }}
                                className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-black"
                            />
                            <div className="flex justify-between text-xs mt-1 font-mono text-gray-500">
                                <span>PRE-SNAP</span>
                                <span>PLAY ACTION</span>
                                <span>END PLAY</span>
                            </div>
                        </div>
                    </div>

                    {/* Navigation */}
                    <div className="flex justify-between">
                        <button
                            onClick={handlePrev}
                            disabled={currentIndex === 0}
                            className="px-6 py-3 border-2 border-black font-bold hover:bg-black hover:text-white disabled:opacity-20 transition-colors uppercase"
                        >
                            &larr; Previous Iteration
                        </button>
                        <button
                            onClick={handleNext}
                            disabled={currentIndex === trace.length - 1}
                            className="px-6 py-3 border-2 border-black font-bold hover:bg-black hover:text-white disabled:opacity-20 transition-colors uppercase"
                        >
                            Next Iteration &rarr;
                        </button>
                    </div>
                </div>

                {/* Right Column: Analytics (Black/White Style) */}
                <div className="space-y-6">

                    {/* Play Header */}
                    <div className="border-b-4 border-black pb-4">
                        <h2 className="text-3xl font-black uppercase italic">{play.play_name}</h2>
                        <p className="font-mono text-gray-600 mt-1">{'>'} {play.personnel_formation}</p>
                    </div>

                    {/* Scores */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 border-2 border-black text-center">
                            <div className="text-xs uppercase font-bold text-gray-500">Predictability</div>
                            <div className="text-3xl font-black">{critique.predictability_score}</div>
                            <div className={`text-xs font-bold mt-1 ${critique.predictability_score < 0.1 ? 'text-green-600' : 'text-red-600'}`}>
                                {critique.predictability_score < 0.1 ? 'LOW (GOOD)' : 'HIGH (BAD)'}
                            </div>
                        </div>

                        <div className="p-4 border-2 border-black text-center">
                            <div className="text-xs uppercase font-bold text-gray-500">Exp. Yards</div>
                            <div className="text-3xl font-black">{critique.expected_yards}</div>
                            <div className={`text-xs font-bold mt-1 ${critique.expected_yards > 5.0 ? 'text-green-600' : 'text-red-600'}`}>
                                {critique.expected_yards > 5.0 ? 'HIGH (GOOD)' : 'LOW (BAD)'}
                            </div>
                        </div>
                    </div>

                    {/* Analysis */}
                    <div className="space-y-6">
                        <div>
                            <h3 className="font-black bg-black text-white inline-block px-2 py-1 mb-2 uppercase text-sm">Design Intent</h3>
                            <p className="text-sm border-l-2 border-gray-300 pl-4 py-1 italic text-gray-700">
                                "{play.ey_ps_analysis}"
                            </p>
                        </div>

                        <div>
                            <h3 className="font-black bg-black text-white inline-block px-2 py-1 mb-2 uppercase text-sm">Post-Mortem</h3>
                            <div className="border-l-2 border-black pl-4 py-1">
                                <FormatBullets text={rationale} />
                            </div>
                        </div>
                    </div>

                    {/* Route Table */}
                    <div className="text-xs font-mono border-t-2 border-black pt-4">
                        {Object.entries(play.route_responsibilities).map(([pos, route]) => (
                            <div key={pos} className="flex justify-between py-1 border-b border-gray-200">
                                <span className="font-bold">{pos}</span>
                                <span>{route}</span>
                            </div>
                        ))}
                    </div>

                </div>
            </div>
        </div>
    );
}
