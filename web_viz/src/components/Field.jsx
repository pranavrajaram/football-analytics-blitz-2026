import React, { useEffect, useState } from 'react';
import { DEFENSE, FORMATIONS } from '../data/formations';
import { getRoutePath } from '../data/routes';

export default function Field({ play, progress }) {
    // Parse play data
    const formationName = play.play.personnel_formation.split(',').pop().trim();
    // e.g. "Trips Right" from "11 Personnel, Gun Trips Right"

    const offenseCoords = FORMATIONS[formationName] || FORMATIONS["Trips Right"]; // Fallback
    const defenseCoords = DEFENSE["Cover 2"]; // Static for now

    // Routes
    const routes = play.play.route_responsibilities || {};

    return (
        <div className="w-full max-w-4xl mx-auto bg-green-600 p-2 rounded-lg shadow-xl relative aspect-video overflow-hidden border-4 border-white">
            {/* Field SVG */}
            {/* ViewBox: X(-30 to 30), Y(-45 to 10). Total H=55. 
                -45 is Endzone Back Line. -35 is Endzone Front Line. 0 is LOS.
            */}
            <svg viewBox="-30 -45 60 55" className="w-full h-full">
                {/* Adjusted ViewBox to show deep field to Endzone */}

                <defs>
                    <pattern id="grass" width="10" height="10" patternUnits="userSpaceOnUse">
                        <rect width="10" height="10" fill="#22c55e" /> {/* Lighter Green (green-500) */}
                        <path d="M0 10 L10 0" stroke="#4ade80" strokeWidth="0.5" opacity="0.3" />
                    </pattern>
                </defs>

                {/* Field Area */}
                <rect x="-30" y="-45" width="60" height="60" fill="url(#grass)" />

                {/* Top Endzone (Vikings) */}
                {/* Starts at -45, Height 10. Ends at -35 line */}
                <rect x="-30" y="-45" width="60" height="10" fill="#4F2683" opacity="0.9" />
                <text x="0" y="-39" textAnchor="middle" fill="#FFC62F" fontSize="4" fontWeight="bold">VIKINGS</text>

                {/* Line of Scrimmage */}
                <line x1="-30" y1="0" x2="30" y2="0" stroke="blue" strokeWidth="0.5" strokeDasharray="1 1" />

                {/* Endzone Line at -35 */}
                <line x1="-30" y1="-35" x2="30" y2="-35" stroke="white" strokeWidth="0.8" />

                {/* Sideline Markers (No lines across field, just invisible markers or hash marks if needed) */}
                {/* User asked to "Get rid of the white lines on the field" */}

                {/* Yard numbers / Hashes could be added here if needed, but keeping clean per request */}

                {/* --- ROUTES --- */}
                {Object.entries(routes).map(([pos, routeName]) => {
                    const start = offenseCoords[pos];
                    if (!start) return null;

                    // Determine side for route logic
                    // If X < 0 (Left), side='L'. If X > 0 (Right), side='R'.
                    // Center is usually 'L' bias or check generic.
                    const side = start.x < 0 ? 'L' : 'R';
                    const pathData = getRoutePath(routeName, side);

                    return (
                        <g key={`route-${pos}`} transform={`translate(${start.x}, ${start.y})`}>
                            <path
                                d={pathData}
                                fill="none"
                                stroke="yellow"
                                strokeWidth="0.5"
                                strokeDasharray="1 0.5"
                                opacity="0.6"
                            />
                        </g>
                    );
                })}

                {/* --- DEFENSE (X) --- */}
                {/* --- DEFENSE (X) --- */}
                <g style={{ opacity: Math.max(0, 1 - progress * 3) }}>
                    {Object.entries(defenseCoords).map(([pos, coord]) => (
                        <g key={pos} transform={`translate(${coord.x}, ${coord.y * -1})`}>
                            {/* Circle with Red Outline, White Fill */}
                            <circle r="2.0" fill="white" stroke="#ef4444" strokeWidth="0.5" />
                            {/* Position Text */}
                            <text textAnchor="middle" dominantBaseline="middle" fontSize="1.5" fill="black" fontWeight="bold">{pos}</text>
                        </g>
                    ))}
                </g>

                {/* --- OFFENSE (O) --- */}
                {Object.entries(offenseCoords).map(([pos, coord]) => {
                    // Filter: Only render if in routes OR if it's the QB
                    if (!routes[pos] && pos !== 'QB') return null;

                    // Check if this player has a route to animate
                    const routeName = routes[pos];
                    const side = coord.x < 0 ? 'L' : 'R';
                    const pathData = routeName ? getRoutePath(routeName, side) : null;

                    return (
                        <g key={pos} transform={`translate(${coord.x}, ${coord.y})`}>
                            <g style={pathData ? {
                                offsetPath: `path('${pathData}')`,
                                offsetDistance: `${progress * 100}%`,
                                offsetRotate: '0deg' /* Prevent rotation */
                            } : {}}>
                                {/* Bigger circle, Lighter Blue Stroke */}
                                <circle r="2.0" fill="white" stroke="#3b82f6" strokeWidth="0.5" />
                                <text textAnchor="middle" dominantBaseline="middle" fontSize="1.8" fill="black" fontWeight="bold">{pos}</text>
                            </g>
                        </g>
                    );
                })}
            </svg>
        </div>
    );
}
