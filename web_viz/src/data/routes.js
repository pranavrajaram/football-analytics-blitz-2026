// Route SVG Paths
// Relative to Receiver Start [0,0]
// Y+ is Downfield
// X+ is Right

// Helper to flip X for left-side receivers if needed?
// Actually simpler to just define standard routes and flip in component if needed.
// But standard "Post" from Left implies going Right. "Post" from Right implies going Left.
// So routes need to be context aware or we define "Inside/Outside".
// Let's assume standard definitions are "From Left". We can flip X for Right side?
// No, "Post" always means "To Goalpost".
// If on Left, Post goes +X +Y.
// If on Right, Post goes -X +Y.

export const getRoutePath = (routeName, side = 'L') => {
    const r = routeName.toLowerCase();
    const dir = side === 'L' ? 1 : -1; // 1 means Move Right (Inside for L), -1 means Move Left (Inside for R)

    // Basic Primitives
    // Y- is UP (Standard SVG)
    if (r.includes('go') || r.includes('streak') || r.includes('fly') || r.includes('fade') || r.includes('vertical')) {
        return `M 0 0 L 0 -40`;
    }
    if (r.includes('post')) {
        // Up 10, then angle Inside 45 deg
        return `M 0 0 L 0 -8 L ${45 * dir} -40`;
    }
    if (r.includes('corner') || r.includes('flag')) {
        // Up 10, then angle Outside 45 deg
        return `M 0 0 L 0 -10 L ${-15 * dir} -40`;
    }
    if (r.includes('out')) {
        // Up 5, then flat Out
        return `M 0 0 L 0 -5 L ${-10 * dir} -5`;
    }
    if (r.includes('dig') || r.includes('in')) {
        // Up 10, then flat In
        return `M 0 0 L 0 -10 L ${10 * dir} -10`;
    }
    if (r.includes('drag') || r.includes('shallow')) {
        // Angle In immediately
        return `M 0 0 Q ${2 * dir} -1 ${15 * dir} -1`;
    }
    if (r.includes('slant')) {
        return `M 0 0 L ${5 * dir} -5`;
    }
    if (r.includes('hitch') || r.includes('curl') || r.includes('hook') || r.includes('stop')) {
        // Up 10, come back
        return `M 0 0 L 0 -12 L ${2 * dir} -10`;
    }
    if (r.includes('flat')) {
        return `M 0 0 L ${-5 * dir} -2`;
    }
    if (r.includes('seam')) {
        return `M 0 0 L 0 -40`;
    }
    if (r.includes('cross')) {
        // Curve like "Top Left of Circle" (Quarter Circle turn)
        // Start Vertical (0,0), Curve to Horizontal (10*dir, -10), drift to (40*dir, -12)
        // No sharp corner.
        return `M 0 0 Q 0 -10 ${20 * dir} -10 L ${33 * dir} -12`;
    }

    // Comeback: Go deep (15y), cut back outside (5y width, back to 12y depth)
    if (r.includes('comeback')) {
        return `M 0 0 L 0 -15 L ${-5 * dir} -12`;
    }

    // Default: Just a short go
    return `M 0 0 L 0 -5`;
};
