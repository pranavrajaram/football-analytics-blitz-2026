// Offense Coordinates (Relative to Ball at [0,0])
// X: Horizontal (+Right, -Left)
// Y: Vertical (+Downfield?, No, usually +Up is standard but we want +Forward).
// Let's say Y+ is Downfield (Forward). 
// So LOS is Y=0. QB is Y=-5.

// Helper to ensure all keys exist
const BASE = { QB: { x: 0, y: 5 } };

export const FORMATIONS = {
    "Spread": { // 2x2. L1,L2, R1,R2.
        ...BASE,
        "L1": { x: -22, y: 0 }, "L2": { x: -14, y: 0 }, "L3": { x: -4, y: 5 }, "L4": { x: -4, y: 5 },
        "R1": { x: 22, y: 0 }, "R2": { x: 14, y: 0 }, "R3": { x: 4, y: 5 }, "R4": { x: 4, y: 5 }
    },
    "Trips Right": { // L1. R1,R2,R3.
        ...BASE,
        "L1": { x: -22, y: 0 }, "L2": { x: -14, y: 0 }, "L3": { x: -4, y: 5 }, "L4": { x: -4, y: 5 },
        "R1": { x: 24, y: 0 }, "R2": { x: 18, y: 0 }, "R3": { x: 12, y: 0 }, "R4": { x: 4, y: 5 }
    },
    "Trips Left": { // L1,L2,L3. R1.
        ...BASE,
        "L1": { x: -24, y: 0 }, "L2": { x: -18, y: 0 }, "L3": { x: -12, y: 0 }, "L4": { x: -4, y: 5 },
        "R1": { x: 22, y: 0 }, "R2": { x: 14, y: 0 }, "R3": { x: 4, y: 5 }, "R4": { x: 4, y: 5 }
    },
    "Bunch Right": {
        ...BASE,
        "L1": { x: -22, y: 0 }, "L2": { x: -4, y: 5 }, "L3": { x: -4, y: 5 }, "L4": { x: -4, y: 5 },
        "R1": { x: 16, y: 0 }, "R2": { x: 18, y: 1 }, "R3": { x: 14, y: 1 }, "R4": { x: 4, y: 5 }
    },
    "Bunch Left": {
        ...BASE,
        "L1": { x: -16, y: 0 }, "L2": { x: -18, y: 1 }, "L3": { x: -14, y: 1 }, "L4": { x: -4, y: 5 },
        "R1": { x: 22, y: 0 }, "R2": { x: 4, y: 5 }, "R3": { x: 4, y: 5 }, "R4": { x: 4, y: 5 }
    },
    "Slot Right": {
        ...BASE,
        "L1": { x: -22, y: 0 }, "L2": { x: -14, y: 0 }, "L3": { x: -4, y: 5 }, "L4": { x: -4, y: 5 },
        "R1": { x: 22, y: 0 }, "R2": { x: 14, y: 0 }, "R3": { x: 4, y: 5 }, "R4": { x: 4, y: 5 }
    },
    "Slot Left": {
        ...BASE,
        "L1": { x: -22, y: 0 }, "L2": { x: -14, y: 0 }, "L3": { x: -4, y: 5 }, "L4": { x: -4, y: 5 },
        "R1": { x: 22, y: 0 }, "R2": { x: 14, y: 0 }, "R3": { x: 4, y: 5 }, "R4": { x: 4, y: 5 }
    },
    "Balanced": {
        ...BASE,
        "L1": { x: -22, y: 0 }, "L2": { x: -6, y: 0 }, "L3": { x: -4, y: 5 }, "L4": { x: -4, y: 5 },
        "R1": { x: 22, y: 0 }, "R2": { x: 6, y: 0 }, "R3": { x: 4, y: 5 }, "R4": { x: 4, y: 5 }
    },
    "Goal Line": {
        ...BASE,
        "L1": { x: -12, y: 0 }, "L2": { x: -6, y: 0 }, "L3": { x: -2, y: 2 }, "L4": { x: -2, y: 2 },
        "R1": { x: 12, y: 0 }, "R2": { x: 6, y: 0 }, "R3": { x: 2, y: 2 }, "R4": { x: 2, y: 2 }
    }
};

export const DEFENSE = {
    "Cover 2": {
        "CB1": { x: -24, y: 5 },
        "CB2": { x: 24, y: 5 },
        "S1": { x: -12, y: 15 }, // Deep Half
        "S2": { x: 12, y: 15 },  // Deep Half
        "LB1": { x: -4, y: 5 },
        "LB2": { x: 4, y: 5 },
        "LB3": { x: 0, y: 5 }
    }
};
