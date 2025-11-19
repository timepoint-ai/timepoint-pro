# Timepoint Dashboard Quick Start

## ✅ Fixed and Working!

The dashboard is now fully configured and tested. All Quarto/Python/Jupyter issues have been resolved.

## 🚀 Usage

### 1. Refresh Data (Optional)

Update the dashboard with your latest simulation runs:

```bash
cd /Users/seanmcdonald/Documents/GitHub/timepoint-daedalus/dashboards
python3.10 load_data.py
```

Output:
```
Loading Timepoint simulation data...
Found 20 recent runs
✓ Exported data to dashboard_data.json
✓ Selected run: run_20251103_132107_551ae766
  - Template: portal_timepoint_unicorn
  - Has narrative: False
  - Mechanisms: 5
```

### 2. View the Dashboard

**Option A: Preview with live reload**

```bash
quarto preview
```

Opens browser at http://localhost:XXXX with auto-refresh on file changes.

**Option B: Render static site**

```bash
quarto render
```

Then open `_site/index.html` in your browser.

**Option C: Open pre-rendered HTML**

```bash
open _site/index.html
```

## 📊 What You'll See

### Main Dashboard (`index.html`)

1. **Run Selector** - Dropdown showing 20 most recent runs
   - Format: `template_id (date) - $cost`
   - Currently defaults to most recent run

2. **Metadata Panel** (left card)
   - Run ID
   - Template
   - Status (colored: green=completed, yellow=running, red=failed)
   - Cost (USD)
   - Start/end timestamps

3. **Stats Panel** (right card)
   - Entity count
   - Timepoint count
   - Causal mode (PEARL, PORTAL, DIRECTORIAL, etc.)
   - Mechanism count

4. **Executive Summary**
   - Narrative overview from simulation JSON

5. **Visualization Tabs**
   - **Timeline**: Interactive vis-timeline chronological view
   - **Network**: Cytoscape.js entity relationship graph
   - **Metrics**: ECharts radar chart of mechanism coverage
   - **Screenplay**: Stubbed Fountain viewer (shows raw content)

### About Page (`about.html`)

- System overview
- Feature list (implemented + stubbed)
- Technical stack details
- All 17 mechanism descriptions

## 🔧 Configuration Fixed

### Issues Resolved

1. ✅ **Jupyter kernel conflict** - Configured to use `python3.10` instead of `python3.13`
2. ✅ **CSS layer error** - Moved `styles.css` from theme to css option
3. ✅ **Observable JS imports** - Removed problematic CSS import, added to HTML head
4. ✅ **Data loading** - Switched from Python execution to pre-generated JSON

### Files Modified

- `_quarto.yml` - Added `jupyter: python3.10`, fixed CSS configuration
- `index.qmd` - Changed to FileAttachment for JSON loading, removed CSS import
- `load_data.py` - New script to pre-generate dashboard data

## 📁 Generated Files

```
_site/
├── index.html              # Main dashboard (99KB)
├── about.html              # About page (44KB)
├── dashboard_data.json     # Pre-loaded run data (12KB)
├── styles.css              # Custom styling (6.3KB)
├── search.json             # Site search index
└── site_libs/              # Quarto libraries
```

## 🎨 Customization

### Change Data

Edit `load_data.py` to:
- Change number of runs loaded (`limit=20`)
- Filter which runs to show
- Add additional data fields

Then run:
```bash
python3.10 load_data.py
quarto render
```

### Change Styling

Edit `styles.css` variables:
```css
:root {
  --primary-purple: #6366f1;  /* Change to your color */
  --primary-cyan: #06b6d4;    /* Change to your color */
  --dark-bg: #0f172a;         /* Background color */
}
```

Then:
```bash
quarto render
```

### Add Visualizations

Edit `index.qmd` Observable JS sections to add new charts/graphs.

## 🚧 Next Steps

### Stubbed Features Ready to Implement

All have UI placeholders with TODO comments in `index.qmd`:

1. **PORTAL Mode Visualization** (lines 396-399)
   - Backward causality arrows
   - Pivot point highlighting
   - Plausibility heatmaps

2. **Run Comparison** (lines 401-404)
   - Side-by-side timelines
   - Diff visualization
   - Cost comparison charts

3. **Fountain Viewer** (lines 359-387)
   - Parse .fountain files
   - Render as formatted screenplay
   - Scene navigation

4. **Search & Filter** (lines 406-409)
   - Full-text search
   - Date range filtering
   - Entity/mechanism filtering

5. **Export Tools** (lines 411-414)
   - PNG/SVG exports
   - PDF reports
   - Shareable links

## 💡 Tips

### Performance

- Dashboard loads instantly (static HTML + JSON)
- No server required
- Can deploy to any static host (GitHub Pages, Netlify, etc.)

### Data Freshness

- Run `python3.10 load_data.py` after each simulation batch
- Or automate with cron/scheduler
- Consider adding timestamp to dashboard showing data age

### Development

Use `quarto preview` for live development:
- Auto-refreshes on file save
- Hot module reload for Observable JS
- Shows render errors in browser console

## ✅ Verification

Test the dashboard:

```bash
cd /Users/seanmcdonald/Documents/GitHub/timepoint-daedalus/dashboards

# 1. Generate data
python3.10 load_data.py

# 2. Render site
quarto render

# 3. Open in browser
open _site/index.html
```

You should see:
- ✅ Run selector with 20 runs
- ✅ Metadata and stats cards
- ✅ Mechanism radar chart (5 mechanisms for sample run)
- ✅ Purple/cyan gradient theme
- ✅ Smooth animations on hover

---

**Dashboard Status**: 🟢 Fully Operational

All core features implemented and tested. Ready for production use!
