# Land-Cover-and-Population-Dynamics

# Significant GPP Change Mask (2010–2020)

## What it shows:
Areas where the difference in Gross Primary Production (GPP) between 2010 and 2020 exceeds a certain threshold (200 in this script).

## Interpretation:
- Brightly colored (blue/red) regions indicate places that have undergone substantial vegetation productivity changes (could be loss or gain).
- Dark/maroon regions show minimal change or masked (no-data) areas.

---

# Population Difference (2010–2020)

## What it shows:
The difference in population density between 2010 and 2020.

## Interpretation:
- Purple/teal shading highlights where population has increased or decreased over time.
- Larger positive differences mean higher population growth in that location.

---

# Urbanization/Deforestation Hotspots (Boolean)

## What it shows:
A simple logical AND between:
1. Areas of significant GPP change.
2. Areas of notable population increase.

## Interpretation:
- **White (True) pixels** represent potential urbanization or deforestation hotspots.
- These are locations where a large jump in population density overlaps with a major drop in vegetation productivity.

---

# Hotspot Intensity (Weighted & Normalized)

## What it shows:
- The boolean hotspot map (above) multiplied by the actual population difference and then normalized for visualization.

## Interpretation:
- **Bright or white areas** on this map highlight where both population growth and vegetation change coincide.
- Intensity is scaled by how large the population jump is.
- Essentially, bigger population gains yield a stronger hotspot signal, emphasizing locations of potentially greater human impact.
