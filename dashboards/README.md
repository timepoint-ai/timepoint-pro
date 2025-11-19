# Timepoint Dashboard

Beautiful, interactive visualization dashboard for Timepoint simulation runs.

![Dashboard Preview](https://img.shields.io/badge/Status-Beta-yellow)
![Quarto](https://img.shields.io/badge/Quarto-1.4+-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)

## Features

### ✅ Implemented

- **Run Selector** - Browse and load recent simulation runs from SQLite database
- **Metadata Cards** - View run statistics, costs, timing, and status
- **Timeline Visualization** - Interactive chronological navigation with vis-timeline
- **Network Graph** - Entity relationship visualization with Cytoscape.js
- **Mechanism Coverage** - Radar chart showing which mechanisms were used (ECharts)
- **Executive Summary** - Display narrative summaries from simulation runs
- **Beautiful UI** - Modern glassmorphism design with gradients and smooth animations

### 🚧 Stubbed (Coming Soon)

- **PORTAL Mode Visualization** - Backward causality paths with pivot points
- **Run Comparison** - Side-by-side analysis of multiple simulations
- **Fountain Viewer** - Embedded screenplay reader with scene navigation
- **Search & Filter** - Full-text search and timepoint filtering
- **Export Tools** - PNG/SVG exports and PDF reports

## Quick Start

### 1. Install Quarto

**Option A: Homebrew (requires Xcode license acceptance)**
```bash
brew install quarto
```

**Option B: Direct installer**
```bash
# Package is already downloaded in project root
sudo installer -pkg quarto-1.4.549-macos.pkg -target /

# Verify installation
quarto --version
```

### 2. Preview Dashboard

```bash
cd dashboards
quarto preview
```

This will:
1. Load recent runs from `metadata/runs.db`
2. Parse narrative JSON from `datasets/*/narrative_*.json`
3. Render interactive visualizations
4. Open dashboard in your browser at http://localhost:XXXX

### 3. Build Static Site

```bash
quarto render
```

Output goes to `_site/` directory - can be deployed to any static host.

## Architecture

### File Structure

```
dashboards/
├── _quarto.yml          # Project configuration
├── index.qmd            # Main dashboard
├── about.qmd            # About page
├── utils.py             # Python data loaders (SQLite + JSON)
├── styles.css           # Custom CSS styling
├── README.md            # This file
└── components/          # Future modular components (empty for now)
```

### Data Flow

```
SQLite (runs.db)
    ↓
Python (utils.py) → TimepointDataLoader
    ↓
JSON exports → Quarto/Observable
    ↓
Visualizations (vis-timeline, Cytoscape, ECharts)
```

### Key Components

#### `utils.py` - Data Loading Layer

```python
from utils import get_recent_runs, load_run, get_most_recent_run

# Get 20 most recent runs
runs = get_recent_runs(limit=20)

# Load full data for specific run
run_data = load_run("run_20251103_105234_3a83d054")

# Get most recent run ID
latest = get_most_recent_run()
```

**Functions:**
- `TimepointDataLoader.get_recent_runs(limit)` - Query SQLite for recent runs
- `TimepointDataLoader.load_narrative(run_id)` - Parse narrative JSON
- `TimepointDataLoader.load_screenplay(run_id)` - Load Fountain screenplay
- `TimepointDataLoader.get_run_summary(run_id)` - Comprehensive run data

#### `index.qmd` - Main Dashboard

Uses Observable JS for reactive visualizations:

1. **Python Cell** - Loads data via `utils.py`
2. **Observable Cells** - Create interactive components:
   - Run selector dropdown
   - Metadata and stats cards
   - Timeline with vis-timeline
   - Network graph with Cytoscape.js
   - Metrics with ECharts

#### `styles.css` - Visual Design

- **Color Palette**: Deep purple (#6366f1) + cyan (#06b6d4) gradients
- **Typography**: Inter (UI) + JetBrains Mono (code)
- **Effects**: Glassmorphism cards, smooth hover animations
- **Responsive**: Mobile-friendly with collapsible grids

## Visualization Libraries

### vis-timeline (Timeline)

```javascript
import {Timeline} from "https://esm.sh/vis-timeline@7.7.3"
```

Interactive timeline showing:
- Timepoint markers on chronological axis
- Event descriptions on hover
- Zoom and pan controls
- Customizable date ranges

### Cytoscape.js (Network Graph)

```javascript
cytoscape = require("https://esm.sh/cytoscape@3.26.0")
```

Entity relationship network with:
- Nodes = entities (sized by importance)
- Edges = relationships/interactions
- Physics-based layout (COSE algorithm)
- Interactive zoom/pan/drag

### ECharts (Charts & Metrics)

```javascript
echarts = require("https://esm.sh/echarts@5.4.3")
```

Mechanism coverage radar chart showing:
- Which mechanisms were used
- Frequency of mechanism invocation
- Comparative mechanism usage

## Usage Examples

### Load Specific Run

```bash
# Edit index.qmd, change this line:
selected_run_id = "run_20251103_105234_3a83d054"

# Render
quarto render
```

### Query Recent Runs in Python

```python
from utils import TimepointDataLoader

loader = TimepointDataLoader()
runs = loader.get_recent_runs(limit=10)

for run in runs:
    print(f"{run['run_id']}: {run['template_id']} - ${run['cost_usd']:.2f}")
```

### Export Run Data

```python
from utils import TimepointDataLoader

loader = TimepointDataLoader()
run_json = loader.export_for_observable("run_20251103_105234_3a83d054")

# Use in Observable JS
print(run_json)
```

## Customization

### Change Theme

Edit `_quarto.yml`:

```yaml
format:
  html:
    theme:
      - darkly  # Change to: cosmo, flatly, darkly, etc.
      - styles.css
```

### Add New Visualization

1. Create `components/my_viz.qmd`
2. Add Observable JS code block
3. Import required libraries via ESM
4. Reference in `index.qmd` via tabs

### Modify Color Palette

Edit `styles.css`:

```css
:root {
  --primary-purple: #6366f1;  /* Change to your color */
  --primary-cyan: #06b6d4;    /* Change to your color */
  /* ... */
}
```

## Troubleshooting

### Quarto Not Found

```bash
# Check installation
which quarto

# If not found, install:
brew install quarto
# or
sudo installer -pkg quarto-1.4.549-macos.pkg -target /
```

### No Data Loading

```bash
# Verify database exists
ls -lh ../metadata/runs.db

# Check Python imports
cd dashboards
python3.10 -c "from utils import get_recent_runs; print(get_recent_runs())"
```

### Visualizations Not Rendering

1. Check browser console for JavaScript errors
2. Verify ESM imports are loading (check network tab)
3. Ensure data has correct structure (`currentRun?.narrative?.timepoints`)

### Styling Not Applied

```bash
# Clear Quarto cache
rm -rf _site _freeze

# Rebuild
quarto render
```

## Development

### Local Development with Live Reload

```bash
cd dashboards
quarto preview --port 8080
```

Edit files and dashboard auto-refreshes.

### Testing Data Loader

```bash
cd dashboards
python3.10 -m pytest utils.py -v  # (if tests added)

# Manual testing
python3.10 -c "
from utils import *
runs = get_recent_runs(5)
for r in runs:
    print(r['run_id'], r['template_id'])
"
```

## Deployment

### GitHub Pages

```bash
quarto publish gh-pages
```

### Netlify

```bash
quarto publish netlify
```

### Static Host

```bash
quarto render
# Upload _site/ directory to any static host
```

## Future Enhancements

See stubbed features in `index.qmd`:

1. **PORTAL Mode Viz** - Backward causality arrows, pivot highlighting
2. **Run Comparison** - Side-by-side timelines/networks for 2+ runs
3. **Fountain Viewer** - Embedded screenplay reader
4. **Search/Filter** - Entity search, timepoint filtering
5. **Export Tools** - PNG/SVG/PDF exports

Pull requests welcome!

## Credits

- **Framework**: [Quarto](https://quarto.org/) (Observable JS + Python)
- **Timeline**: [vis-timeline](https://visjs.github.io/vis-timeline/)
- **Network**: [Cytoscape.js](https://js.cytoscape.org/)
- **Charts**: [Apache ECharts](https://echarts.apache.org/)
- **Fonts**: Inter + JetBrains Mono

---

**Note**: Dashboard requires Quarto installation. The package `quarto-1.4.549-macos.pkg` has been pre-downloaded in project root for convenience.
